# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sys
from functools import partial
from typing import Callable
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Sequence

from cuda.lang._ir import ir, ops
from cuda.lang import _mlir as mlir
from cuda.lang._mlir._builtins import _Cursor
import cuda.lang._mlir.extras.types as T
import cuda.lang._ir.type as ir_type
from cuda.lang.compilation import KernelSignature
import cuda.lang._datatype as datatype
from cuda.lang._exception import TileTypeError
from cuda.tile._ir.type import TileTy, PointerTy
from .type_conversion import (
    ir_type_to_mlir_type,
    mlir_constant_of_type,
    convert_dtype,
)


def _require_scalar_type(typ: ir_type.Type):
    if not isinstance(typ, ir_type.TileTy) or typ.shape != ():
        raise TileTypeError(f"Expected scalar type, got {typ=}")
    return typ


def _get_llvm_atomic_binop(
    kind: ops.AtomicRMWKind, dtype: datatype.DType
) -> mlir.llvm.AtomicBinOp:
    signed = datatype.is_signed(dtype)
    is_float = datatype.is_float(dtype)
    match kind:
        case ops.AtomicRMWKind.ADD:
            return mlir.llvm.AtomicBinOp.fadd if is_float else mlir.llvm.AtomicBinOp.add
        case ops.AtomicRMWKind.SUB:
            return mlir.llvm.AtomicBinOp.fsub if is_float else mlir.llvm.AtomicBinOp.sub
        case ops.AtomicRMWKind.AND:
            return mlir.llvm.AtomicBinOp._and
        case ops.AtomicRMWKind.OR:
            return mlir.llvm.AtomicBinOp._or
        case ops.AtomicRMWKind.XOR:
            return mlir.llvm.AtomicBinOp._xor
        case ops.AtomicRMWKind.MIN:
            if is_float:
                return mlir.llvm.AtomicBinOp.fminimum
            return mlir.llvm.AtomicBinOp.min if signed else mlir.llvm.AtomicBinOp.umin
        case ops.AtomicRMWKind.MAX:
            if is_float:
                return mlir.llvm.AtomicBinOp.fmaximum
            return mlir.llvm.AtomicBinOp.max if signed else mlir.llvm.AtomicBinOp.umax
        case ops.AtomicRMWKind.INC:
            return mlir.llvm.AtomicBinOp.uinc_wrap
        case ops.AtomicRMWKind.DEC:
            return mlir.llvm.AtomicBinOp.udec_wrap
        case _:
            raise NotImplementedError(f"Unsupported atomic {kind=} for {dtype=}")


@dataclass(frozen=True)
class _OpForDType:
    bool_op: mlir.Operation | None
    signed_integral_op: mlir.Operation | None
    unsigned_integral_op: mlir.Operation | None
    float_op: mlir.Operation | None


def _get_mlir_op_for_op_and_dtype(
    fn: str, dtype: datatype.ArithmeticDType
) -> Callable[[mlir.Value, mlir.Value], mlir.Value] | None:
    _OPERATIONS = {
        "floordiv": _OpForDType(
            None,
            mlir.arith.add_FloorDivSIOp,
            mlir.arith.add_DivUIOp,
            None,
        ),
        "add": _OpForDType(
            mlir.arith.add_OrIOp,
            mlir.arith.add_AddIOp,
            mlir.arith.add_AddIOp,
            mlir.arith.add_AddFOp,
        ),
        "sub": _OpForDType(
            None,
            mlir.arith.add_SubIOp,
            mlir.arith.add_SubIOp,
            mlir.arith.add_SubFOp,
        ),
        "mul": _OpForDType(
            mlir.arith.add_AndIOp,
            mlir.arith.add_MulIOp,
            mlir.arith.add_MulIOp,
            mlir.arith.add_MulFOp,
        ),
        "truediv": _OpForDType(
            None,
            mlir.arith.add_DivSIOp,
            mlir.arith.add_DivUIOp,
            mlir.arith.add_DivFOp,
        ),
        "xor": _OpForDType(
            mlir.arith.add_XOrIOp,
            mlir.arith.add_XOrIOp,
            mlir.arith.add_XOrIOp,
            None,
        ),
        "or_": _OpForDType(
            mlir.arith.add_OrIOp,
            mlir.arith.add_OrIOp,
            mlir.arith.add_OrIOp,
            None,
        ),
        "and_": _OpForDType(
            mlir.arith.add_AndIOp,
            mlir.arith.add_AndIOp,
            mlir.arith.add_AndIOp,
            None,
        ),
        "c_mod": _OpForDType(
            None,
            mlir.arith.add_RemSIOp,
            mlir.arith.add_RemUIOp,
            None,
        ),
    }

    if fn not in _OPERATIONS:
        return None

    op_for_dtype = _OPERATIONS[fn]

    if datatype.is_boolean(dtype):
        return op_for_dtype.bool_op
    elif datatype.is_integral(dtype):
        if datatype.is_signed(dtype):
            return op_for_dtype.signed_integral_op
        else:
            return op_for_dtype.unsigned_integral_op
    elif datatype.is_float(dtype):
        return op_for_dtype.float_op
    else:
        return None


def _get_mlir_unary_op_for_op_and_dtype(
    fn: str, dtype: datatype.ArithmeticDType
) -> Callable[[mlir.Value], mlir.Value] | None:
    if fn == "pos":
        return lambda operand: operand

    if fn != "neg":
        return None

    if datatype.is_float(dtype):
        return mlir.arith.add_NegFOp

    if datatype.is_integral(dtype):
        mlir_type = ir_type_to_mlir_type(dtype)
        zero = mlir_constant_of_type(mlir_type, 0)
        return lambda operand: mlir.arith.add_SubIOp(lhs=zero, rhs=operand)

    return None


def _get_mlir_comparison_op(
    fn: str, dtype: datatype.ArithmeticDType
) -> Callable[[mlir.Value, mlir.Value], mlir.Value] | None:
    if datatype.is_float(dtype):
        match fn:
            case "lt":
                return partial(
                    mlir.arith.add_CmpFOp, predicate=mlir.arith.CmpFPredicate.OLT
                )
            case "le":
                return partial(
                    mlir.arith.add_CmpFOp, predicate=mlir.arith.CmpFPredicate.OLE
                )
            case "gt":
                return partial(
                    mlir.arith.add_CmpFOp, predicate=mlir.arith.CmpFPredicate.OGT
                )
            case "ge":
                return partial(
                    mlir.arith.add_CmpFOp, predicate=mlir.arith.CmpFPredicate.OGE
                )
            case "eq":
                return partial(
                    mlir.arith.add_CmpFOp, predicate=mlir.arith.CmpFPredicate.OEQ
                )
            case "ne":
                return partial(
                    mlir.arith.add_CmpFOp, predicate=mlir.arith.CmpFPredicate.ONE
                )
            case _:
                return None
    else:
        signed = datatype.is_signed(dtype)
        match fn:
            case "lt":
                pred = (
                    mlir.arith.CmpIPredicate.slt
                    if signed
                    else mlir.arith.CmpIPredicate.ult
                )
                return partial(mlir.arith.add_CmpIOp, predicate=pred)
            case "le":
                pred = (
                    mlir.arith.CmpIPredicate.sle
                    if signed
                    else mlir.arith.CmpIPredicate.ule
                )
                return partial(mlir.arith.add_CmpIOp, predicate=pred)
            case "gt":
                pred = (
                    mlir.arith.CmpIPredicate.sgt
                    if signed
                    else mlir.arith.CmpIPredicate.ugt
                )
                return partial(mlir.arith.add_CmpIOp, predicate=pred)
            case "ge":
                pred = (
                    mlir.arith.CmpIPredicate.sge
                    if signed
                    else mlir.arith.CmpIPredicate.uge
                )
                return partial(mlir.arith.add_CmpIOp, predicate=pred)
            case "eq":
                return partial(
                    mlir.arith.add_CmpIOp, predicate=mlir.arith.CmpIPredicate.eq
                )
            case "ne":
                return partial(
                    mlir.arith.add_CmpIOp, predicate=mlir.arith.CmpIPredicate.ne
                )
            case _:
                return None


# These operations have aggregate results. The RHS's elements are stored to
# the LHS's when lowering Assign operations and are no-ops at the MLIR level.
_NOOP_LOWERINGS = frozenset([
    ops.MakeTensorView,
    ops.ReinterpretPointerAsArray,
])


class IR2MLIR:
    def __init__(
        self, signature: KernelSignature, region: ir.Region, ctx: ir.IRContext
    ):
        self.ctx = ctx
        self.region = region
        self.signature = signature
        self.var_map: dict[str, mlir.Value] = {}
        self.block_map: dict[int, mlir.Block] = {}
        self._module_op: mlir.Operation | None = None
        self._gpu_module_op: mlir.Block | None = None
        self._func_op: mlir.Operation | None = None
        self._current_op: ir.Operation | None = None
        self._seen_foreign_functions: dict[str, mlir.llvm.LLVMFunctionType] = {}

    def __call__(self) -> mlir.Operation:
        self.setup_func_op()
        self.setup_blocks()

        try:
            self.lower_region()
        except Exception:
            if self.ctx.log_ir_on_error:
                highlight_loc = (
                    self._current_op.loc if self._current_op is not None else None
                )
                ir_str = self.region.to_string(highlight_loc=highlight_loc)
                print(
                    f"==== Encountered error converting IR to MLIR: ====\n\n{ir_str}\n\n",
                    file=sys.stderr,
                )
            raise

        return self.module_op

    def lower_region(self):
        for ir_block in self.region.blocks:
            mlir_block = self.block_map[ir_block]
            assert len(mlir_block.args) == len(ir_block.params), (
                f"MLIR block parameters do not match IR block parameters: {ir_block._name}"
            )
            for mlir_arg, ir_arg in zip(mlir_block.args, ir_block.params):
                assert mlir_arg.type == ir_type_to_mlir_type(ir_arg.get_type()), (
                    f"MLIR argument type does not match IR argument type: {ir_arg.name}"
                )
                self.def_var(ir_arg, mlir_arg)

            self.lower_block(ir_block)

    @property
    def module_op(self) -> mlir.Operation:
        assert self._module_op is not None, "MLIR module not initialized"
        return self._module_op

    @property
    def gpu_module_op(self) -> mlir.Block:
        assert self._gpu_module_op is not None, "MLIR GPU module not initialized"
        return self._gpu_module_op

    @property
    def func_op(self) -> mlir.Operation:
        assert self._func_op is not None, "MLIR GPU function not initialized"
        return self._func_op

    @property
    def func_region(self) -> mlir.Region:
        return self.func_op.regions[0]

    def get_var(self, var: ir.Var) -> mlir.Value:
        if self.defined(var):
            return self.var_map[var.name]

        raise KeyError(f"Variable {var.name} not found")

    def def_var(self, var: ir.Var, value: mlir.Value):
        if self.defined(var):
            raise ValueError(f"Variable {var.name} is already defined")

        self.var_map[var.name] = value

    def defined(self, var: ir.Var) -> bool:
        return var.name in self.var_map

    def lower_aggregate_assign(self, dst: ir.Var, src: ir.Var) -> None:
        assert dst.is_aggregate() == src.is_aggregate()
        assert dst.is_aggregate(), (
            f"Expected aggregate operands, got dst={dst.get_type()}, src={src.get_type()}"
        )

        dst_items = dst.get_aggregate().as_tuple()
        src_items = src.get_aggregate().as_tuple()
        assert len(dst_items) == len(src_items), (
            "Aggregate alias shape mismatch while lowering to MLIR: "
            f"{dst.name} has {len(dst_items)} items but {src.name} has {len(src_items)}"
        )

        for dst_item, src_item in zip(dst_items, src_items, strict=True):
            assert dst_item.is_aggregate() is src_item.is_aggregate()
            if dst_item.is_aggregate():
                self.lower_aggregate_assign(dst_item, src_item)
            elif not self.defined(dst_item):
                self.def_var(dst_item, self.get_var(src_item))

        if self.defined(src) and not self.defined(dst):
            self.def_var(dst, self.get_var(src))

    def setup_blocks(self):
        for ir_block in self.region.blocks:
            mlir_block_param_types = tuple(
                ir_type_to_mlir_type(param.get_type()) for param in ir_block.params
            )
            block_param_names = tuple(param.name for param in ir_block.params)
            args = tuple(
                mlir.Value(param_type, param_name)
                for param_type, param_name in zip(
                    mlir_block_param_types, block_param_names
                )
            )
            mlir_block = self.func_region.new_block(args=args, block_id=ir_block._name)
            self.block_map[ir_block] = mlir_block

    def setup_func_op(self):
        with mlir.Block().append_here() as top_block:
            module_region = mlir.Region()
            mlir.add_ModuleOp(
                bodyRegion=module_region,
                extra_attributes=[("gpu.container_module", mlir.UnitAttr())],
            )
        self._module_op = top_block[0]

        with module_region.new_block().append_here() as module_block:
            gpu_module_region = mlir.Region()
            mlir.gpu.add_GPUModuleOp(
                sym_name="kernels",
                targets=mlir.ArrayAttr(value=[mlir.nvvm.NVVMTargetAttr()]),
                bodyRegion=gpu_module_region,
            )
        self._gpu_module_op = module_block[0]

        with gpu_module_region.new_block().append_here():
            body_region = mlir.Region()
            input_types = tuple(
                ir_type_to_mlir_type(param.get_type())
                for param in self.region.blocks[0].params
            )
            function_type = mlir.FunctionType(inputs=input_types, results=())
            mlir.gpu.add_GPUFuncOp(
                function_type=function_type,
                body=body_region,
                extra_attributes=[
                    ("sym_name", mlir.StringAttr(value=self.signature.symbol)),
                    ("gpu.kernel", mlir.UnitAttr()),
                ],
            )
            self._func_op = gpu_module_region.blocks[0].operations[-1]

    def lower_block(self, ir_block: ir.Block):
        mlir_block = self.block_map[ir_block]
        with mlir_block.append_here():
            for ir_operation in ir_block.operations:
                self._current_op = ir_operation

                if isinstance(ir_operation, tuple(_NOOP_LOWERINGS)):
                    continue

                results = self.lower_operation(ir_operation)

                # Aggregate assignments do not materialize an MLIR SSA value
                if isinstance(ir_operation, ops.Assign):
                    continue

                assert len(ir_operation.result_vars) == len(results)
                for lhs, rhs in zip(ir_operation.result_vars, results):
                    self.def_var(lhs, rhs)

    @singledispatchmethod
    def lower_operation(self, operation: ir.Operation) -> Sequence[mlir.Value]:
        raise NotImplementedError(f"Unable to lower {operation=} to MLIR")

    @lower_operation.register
    def lower_comparison(
        self, operation: ops.RawComparisonOperation
    ) -> Sequence[mlir.Value]:
        lhs_type = _require_scalar_type(operation.lhs.get_type()).dtype
        rhs_type = _require_scalar_type(operation.rhs.get_type()).dtype
        if lhs_type != rhs_type:
            raise TileTypeError(
                "Comparison operations can not promote types: "
                f"{lhs_type} and {rhs_type}"
            )

        mlir_op = _get_mlir_comparison_op(operation.fn, lhs_type)
        if mlir_op is None:
            raise NotImplementedError(
                f"Comparison operation {operation.fn} not supported for {lhs_type}"
            )

        lhs = self.get_var(operation.lhs)
        rhs = self.get_var(operation.rhs)

        result = mlir_op(lhs=lhs, rhs=rhs)
        result = mlir.arith.add_ExtSIOp(out_type=T.i8(), in_=result)
        return [result]

    @lower_operation.register
    def lower_assume_bounded(
        self, operation: ops.AssumeBounded
    ) -> Sequence[mlir.Value]:
        return [self.get_var(operation.x)]

    @lower_operation.register
    def lower_assume_div_by(self, operation: ops.AssumeDivBy) -> Sequence[mlir.Value]:
        return [self.get_var(operation.x)]

    @lower_operation.register
    def lower_raw_unary_arith(self, operation: ops.Unary) -> Sequence[mlir.Value]:
        res_dtype = _require_scalar_type(operation.result_var.get_type()).dtype
        mlir_op = _get_mlir_unary_op_for_op_and_dtype(operation.fn, res_dtype)
        if mlir_op is None:
            raise NotImplementedError(
                f"Arithmetic operation {operation.fn} not supported for {res_dtype=}"
            )

        operand = self.get_var(operation.operand)
        result = mlir_op(operand=operand)
        return [result]

    @lower_operation.register
    def lower_raw_binary_arith(
        self, operation: ops.RawBinaryArithmeticOperation
    ) -> Sequence[mlir.Value]:
        lhs_type = _require_scalar_type(operation.lhs.get_type())
        rhs_type = _require_scalar_type(operation.rhs.get_type())
        res_type = _require_scalar_type(operation.result_var.get_type())

        if lhs_type.dtype != rhs_type.dtype:
            raise TileTypeError(
                "Arithmetic operations can not promote types: "
                f"{lhs_type.dtype} and {rhs_type.dtype}"
            )

        res_dtype = res_type.dtype
        lhs = self.get_var(operation.lhs)
        rhs = self.get_var(operation.rhs)

        mlir_op = _get_mlir_op_for_op_and_dtype(operation.fn, res_dtype)
        if mlir_op is None:
            raise NotImplementedError(
                f"Arithmetic operation {operation.fn} not supported for {res_dtype=}"
            )

        result = mlir_op(lhs=lhs, rhs=rhs)
        return [result]

    @lower_operation.register
    def lower_raw_binary_bitwise_arith(
        self, operation: ops.RawBinaryBitwiseOperation
    ) -> Sequence[mlir.Value]:
        return self.lower_raw_binary_arith(operation)

    @lower_operation.register
    def lower_astype(self, operation: ops.TileAsType) -> Sequence[mlir.Value]:
        src = self.get_var(operation.x)
        src_type = _require_scalar_type(operation.x.get_type()).dtype
        dst_type = _require_scalar_type(operation.result_var.get_type()).dtype
        if not isinstance(src_type, datatype.ArithmeticDType) or not isinstance(
            dst_type, datatype.ArithmeticDType
        ):
            raise NotImplementedError(
                f"Expected arithmetic types, got {src_type=} {dst_type=}"
            )
        result = convert_dtype(src_type, dst_type, src)
        return [result]

    @lower_operation.register
    def lower_typed_const(self, operation: ops.TypedConst) -> Sequence[mlir.Value]:
        target_type = operation.result_var.get_type()
        value = operation.value
        match target_type:
            case (
                ir_type.FunctionTy()
                | ir_type.ModuleTy()
                | ir_type.StringTy()
                | ir_type.DTypeConstructor()
                | ir_type.NoneType()
                | ir_type.TypeTy()
                | ir_type.EnumTy()
            ):
                return [value]
            case ir_type.TileTy():
                assert target_type.shape == (), "Tile types must be scalar"
                mlir_type = ir_type_to_mlir_type(target_type.dtype)
                cst = mlir_constant_of_type(mlir_type, value)
                return [cst]
            case _:
                raise NotImplementedError(
                    f"Unable to convert {value=} to MLIR constant of type {target_type=}"
                )

    @lower_operation.register
    def lower_atomic_rmw(self, operation: ops.AtomicRMW) -> Sequence[mlir.Value]:
        pointer = self.get_var(operation.pointer)
        value = self.get_var(operation.value)
        value_dtype = _require_scalar_type(operation.value.get_type()).dtype
        bin_op = _get_llvm_atomic_binop(operation.kind, value_dtype)

        result = mlir.llvm.add_AtomicRMWOp(
            bin_op=bin_op,
            ptr=pointer,
            val=value,
            ordering=operation.memory_order,
        )
        return [result]

    @lower_operation.register
    def lower_atomic_exchange(
        self, operation: ops.AtomicExchange
    ) -> Sequence[mlir.Value]:
        pointer = self.get_var(operation.pointer)
        value = self.get_var(operation.value)
        result = mlir.llvm.add_AtomicRMWOp(
            bin_op=mlir.llvm.AtomicBinOp.xchg,
            ptr=pointer,
            val=value,
            ordering=operation.memory_order,
        )
        return [result]

    @lower_operation.register
    def lower_atomic_cas(self, operation: ops.AtomicCAS) -> Sequence[mlir.Value]:
        pointer = self.get_var(operation.pointer)
        compare = self.get_var(operation.compare)
        value = self.get_var(operation.value)
        pair = mlir.llvm.add_AtomicCmpXchgOp(
            ptr=pointer,
            cmp=compare,
            val=value,
            success_ordering=operation.success_memory_order,
            failure_ordering=operation.failure_memory_order,
        )

        # llvm.cmpxchg returns {old_value, success_flag}, we want the old value.
        old_value = mlir.llvm.add_ExtractValueOp(
            res_type=value.type,
            container=pair,
            position=(0,),
        )
        return [old_value]

    @lower_operation.register
    def lower_branch(self, operation: ops.Branch) -> Sequence[mlir.Value]:
        mlir_block = self.block_map[operation.target]
        mlir_args = tuple(self.get_var(arg) for arg in operation.args)
        mlir.cf.add_BranchOp(
            dest=mlir_block.label,
            destOperands=mlir_args,
        )
        return []

    @lower_operation.register
    def lower_cond_branch(self, operation: ops.CondBranch) -> Sequence[mlir.Value]:
        cond = self.get_var(operation.cond)
        cond = mlir.arith.add_TruncIOp(
            out_type=T.i1(),
            in_=cond,
        )
        true_block = self.block_map[operation.true_target]
        true_args = tuple(self.get_var(arg) for arg in operation.true_args)
        false_block = self.block_map[operation.false_target]
        false_args = tuple(self.get_var(arg) for arg in operation.false_args)
        mlir.cf.add_CondBranchOp(
            condition=cond,
            trueDest=true_block.label,
            falseDest=false_block.label,
            trueDestOperands=true_args,
            falseDestOperands=false_args,
        )
        return []

    @lower_operation.register
    def lower_assign(self, operation: ops.Assign) -> None:
        if operation.result_var.is_aggregate():
            self.lower_aggregate_assign(operation.result_var, operation.value)
        else:
            self.def_var(operation.result_var, self.get_var(operation.value))

    @lower_operation.register
    def lower_return(self, operation: ops.Return) -> Sequence[mlir.Value]:
        assert operation.result_vars == (), "Kernels may not return values"
        mlir.gpu.add_ReturnOp(operands=[])
        return []

    @lower_operation.register
    def lower_printf(self, operation: ops.TilePrintf) -> Sequence[mlir.Value]:
        args = tuple(self.get_var(arg) for arg in operation.args)
        mlir.gpu.add_PrintfOp(format=operation.format, args=args)
        return []

    @lower_operation.register
    def lower_pointer_offset(
        self, operation: ops.PointerOffset
    ) -> Sequence[mlir.Value]:
        pointer_type = _require_scalar_type(operation.pointer.get_type()).dtype
        assert isinstance(pointer_type, ir_type.PointerTy), (
            f"Expected a typed pointer, got {pointer_type}"
        )
        element_type = ir_type_to_mlir_type(pointer_type.pointee_type)

        pointer = self.get_var(operation.pointer)
        offset = self.get_var(operation.offset)
        dynamic_32b_sentinel = -1 << 31
        offset_pointer = mlir.llvm.add_GEPOp(
            res_type=pointer.type,
            base=pointer,
            dynamicIndices=[offset],
            rawConstantIndices=[dynamic_32b_sentinel],
            elem_type=element_type,
        )
        return [offset_pointer]

    @lower_operation.register
    def lower_load_pointer(self, operation: ops.LoadPointer) -> Sequence[mlir.Value]:
        pointer_type = _require_scalar_type(operation.pointer.get_type()).dtype
        assert isinstance(pointer_type, ir_type.PointerTy), (
            f"Expected a typed pointer, got {pointer_type}"
        )
        result_type = ir_type_to_mlir_type(operation.result_var.get_type())
        pointer = self.get_var(operation.pointer)
        result = mlir.llvm.add_LoadOp(
            res_type=result_type,
            addr=pointer,
            alignment=operation.alignment,
            volatile_=operation.volatile,
        )
        return [result]

    @lower_operation.register
    def lower_store_pointer(self, operation: ops.StorePointer) -> Sequence[mlir.Value]:
        pointer = self.get_var(operation.pointer)
        value = self.get_var(operation.value)
        mlir.llvm.add_StoreOp(
            value=value,
            addr=pointer,
            alignment=operation.alignment,
            volatile_=operation.volatile,
        )
        return []

    @lower_operation.register
    def lower_alloc_local_memory(
        self, operation: ops.AllocLocalMemory
    ) -> Sequence[mlir.Value]:
        result_ty = operation.result_var.get_type()
        assert isinstance(result_ty, TileTy)
        assert result_ty.shape == ()
        assert isinstance(result_ty.dtype, PointerTy)

        result_ty_mlir = ir_type_to_mlir_type(result_ty)
        elem_type_mlir = ir_type_to_mlir_type(result_ty.dtype.pointee_type)

        with self.func_region.blocks[0].prepend_here():
            array_size = mlir_constant_of_type(T.i64(), operation.count)
            ptr = mlir.llvm.add_AllocaOp(
                res_type=result_ty_mlir,
                arraySize=array_size,
                elem_type=elem_type_mlir,
                alignment=operation.alignment,
            )
        # It would be nice to emit llvm.intr.lifetime_start() here but MLIR's region-simplify
        # pass doesn't respect it and produces invalid IR as a result.
        # mlir.llvm.add_LifetimeStartOp(ptr=ptr)
        return [ptr]

    @lower_operation.register
    def lower_dealloc_local_memory(self, operation: ops.DeallocLocalMemory):
        # It would be nice to emit llvm.intr.lifetime_end() here but MLIR's region-simplify
        # pass doesn't respect it and produces invalid IR as a result.
        # mlir.llvm.add_LifetimeEndOp(ptr=self.get_var(operation.ptr))
        return ()

    @lower_operation.register
    def lower_alloc_static_shared_memory(
        self, operation: ops.AllocStaticSharedMemory
    ) -> Sequence[mlir.Value]:
        result_ty = operation.result_var.get_type()
        assert isinstance(result_ty, TileTy)
        assert result_ty.shape == ()
        assert isinstance(result_ty.dtype, PointerTy)

        elem_type = ir_type_to_mlir_type(result_ty.dtype.pointee_type)
        global_type = mlir.llvm.LLVMArrayType(
            elementType=elem_type,
            numElements=operation.count,
        )
        sym = f"static_shared_memory_{operation.result_var.name}"
        with self.gpu_module_op.regions[0].blocks[0].prepend_here():
            mlir.llvm.add_GlobalOp(
                global_type=global_type,
                sym_name=sym,
                linkage=mlir.llvm.Linkage.Internal,
                addr_space=ir_type.MemorySpace.SHARED._value_,
                visibility_=mlir.llvm.Visibility.Default,
                initializer=mlir.Region(),
                alignment=operation.alignment,
            )
        base = mlir.llvm.add_AddressOfOp(
            res_type=ir_type_to_mlir_type(result_ty),
            global_name=sym,
        )
        return [base]

    @lower_operation.register
    def lower_get_dyn_shared_memory_base_ptr(self, operation: ops.GetDynSharedMemoryBasePtr
                                             ) -> Sequence[mlir.Value]:
        global_type = mlir.llvm.LLVMArrayType(elementType=T.i8(), numElements=0)
        sym = self.signature.symbol + "$dynamic_shared_memory"
        addr_space = ir_type.MemorySpace.SHARED._value_
        with self.gpu_module_op.regions[0].blocks[0].prepend_here():
            mlir.llvm.add_GlobalOp(
                global_type=global_type,
                sym_name=sym,
                linkage=mlir.llvm.Linkage.External,
                addr_space=addr_space,
                visibility_=mlir.llvm.Visibility.Default,
                initializer=mlir.Region(),
                alignment=ops.GetDynSharedMemoryBasePtr.initial_alignment,
            )
        res_type = ir_type_to_mlir_type(operation.result_var.get_type())
        assert isinstance(res_type, mlir.llvm.LLVMPointerType)
        ptr = mlir.llvm.add_AddressOfOp(res_type=res_type, global_name=sym)
        return [ptr]

    @lower_operation.register
    def lower_syncthreads(self, operation: ops.SyncThreads) -> Sequence[mlir.Value]:
        mlir.nvvm.add_Barrier0Op()
        return [None]

    @lower_operation.register
    def lower_inline_ptx(self, operation: ops.InlinePTX) -> Sequence[mlir.Value]:
        ptx_code = operation.ptx_code
        ro_args = tuple(self.get_var(arg) for arg in operation.read_only_operands)
        rw_args = tuple(self.get_var(arg) for arg in operation.read_write_operands)
        wo_args = tuple(
            ir_type_to_mlir_type(arg) for arg in operation.write_only_operands
        )
        results = mlir.nvvm.add_InlinePtxOp(
            ptxCode=ptx_code,
            readOnlyArgs=ro_args,
            readWriteArgs=rw_args,
            writeOnlyArgs_types=wo_args,
        )
        return tuple(results)

    def _lower_intrinsic_operand(self, operand: ir.Var) -> mlir.Value:
        operand_value = self.get_var(operand)
        operand_type = operand.get_type()
        if isinstance(operand_type, ir_type.VectorTy):
            return operand_value
        if datatype.is_boolean(operand_type.dtype):
            return mlir.arith.add_TruncIOp(out_type=T.i1(), in_=operand_value)
        else:
            return operand_value

    def _lower_intrinsic_result(
        self, results: Sequence[mlir.Value]
    ) -> Sequence[mlir.Value]:
        for result in results:
            if result.type == T.i1():
                yield mlir.arith.add_ExtUIOp(out_type=T.i8(), in_=result)
            else:
                yield result

    def _lower_intrinsic_result_type(
        self, result_types: Sequence[ir_type.Type]
    ) -> Sequence[mlir.Type]:
        for result_type in result_types:
            if result_type == ir_type.make_tile_ty(datatype.bool_, ()):
                yield T.i1()
            else:
                yield ir_type_to_mlir_type(result_type)

    def _extract_aggregate_elements(self, value: mlir.Value) -> Sequence[mlir.Value]:
        assert isinstance(value.type, mlir.llvm.LLVMStructType)
        element_types = value.type.types
        for i, element_type in enumerate(element_types):
            yield mlir.llvm.add_ExtractValueOp(
                res_type=element_type,
                container=value,
                position=[i]
            )

    @lower_operation.register
    def lower_raw_nvvm_intrinsic(
        self, operation: ops.RawNVVMIntrinsic
    ) -> Sequence[mlir.Value]:
        operands = tuple(
            self._lower_intrinsic_operand(operand)
            for operand in operation.operands_
        )
        result_types = tuple(
            self._lower_intrinsic_result_type(
                result.get_type()
                for result in operation.result_vars
            )
        )

        match len(result_types):
            case 0:
                mlir_result_type = None
            case 1:
                mlir_result_type = result_types[0]
            case _:
                mlir_result_type = mlir.llvm.LLVMStructType(types=result_types)

        intrinsic_call_result = mlir.llvm.add_CallIntrinsicOp(
            results_type=mlir_result_type,
            intrin=operation.intrinsic,
            args=operands,
            op_bundle_operands=(),
            op_bundle_sizes=(),
        )

        match len(result_types):
            case 0:
                mlir_values = []
            case 1:
                mlir_values = [intrinsic_call_result]
            case _:
                mlir_values = list(self._extract_aggregate_elements(intrinsic_call_result))

        return tuple(self._lower_intrinsic_result(mlir_values))

    @lower_operation.register
    def lower_raw_where(self, operation: ops.RawWhereOperation) -> Sequence[mlir.Value]:
        cond_i8 = self.get_var(operation.cond)
        cond = mlir.arith.add_TruncIOp(
            out_type=T.i1(),
            in_=cond_i8,
        )
        x = self.get_var(operation.x)
        y = self.get_var(operation.y)
        result = mlir.llvm.add_SelectOp(
            condition=cond,
            trueValue=x,
            falseValue=y,
        )
        return [result]

    @lower_operation.register
    def lower_reinterpret_pointer(
        self, operation: ops.ReinterpretPointer
    ) -> Sequence[mlir.Value]:
        # This operation only exists to reinterpret the type of a pointer
        # at the IR level, but the MLIR representation will be the same either way.
        return [self.get_var(operation.pointer)]

    @lower_operation.register
    def lower_addrspace_cast(
        self, operation: ops.AddrSpaceCast
    ) -> Sequence[mlir.Value]:
        value = self.get_var(operation.pointer)
        ir_result_type = operation.result_var.get_type()
        if ir_result_type.dtype.memory_space.value == value.type.addressSpace:
            return [value]
        mlir_result_type = ir_type_to_mlir_type(ir_result_type)
        result = mlir.llvm.add_AddrSpaceCastOp(
            res_type=mlir_result_type,
            arg=value,
        )
        return [result]

    @lower_operation.register
    def lower_bitshift(
        self, operation: ops.RawBitwiseShiftOperation
    ) -> Sequence[mlir.Value]:
        lhs_type = _require_scalar_type(operation.lhs.get_type())
        rhs_type = _require_scalar_type(operation.rhs.get_type())
        res_type = _require_scalar_type(operation.result_var.get_type())

        for typ, name in (
            (lhs_type, "lhs"),
            (rhs_type, "rhs"),
            (res_type, "result"),
        ):
            if not datatype.is_integral(typ.dtype):
                raise TileTypeError(
                    "Bitwise shift operations require integral scalar "
                    f"types, got {name}={typ.dtype}"
                )

        if lhs_type.dtype != res_type.dtype:
            raise TileTypeError(
                "Bitwise shift operations require the lhs and result dtypes to match: "
                f"{lhs_type.dtype} and {res_type.dtype}"
            )

        lhs = self.get_var(operation.lhs)
        rhs = self.get_var(operation.rhs)

        match operation.fn:
            case "lshift":
                result = mlir.arith.add_ShLIOp(lhs=lhs, rhs=rhs)
            case "rshift":
                shift_op = (
                    mlir.arith.add_ShRSIOp
                    if datatype.is_signed(res_type.dtype)
                    else mlir.arith.add_ShRUIOp
                )
                result = shift_op(lhs=lhs, rhs=rhs)
            case _:
                raise NotImplementedError(
                    f"Bitwise shift operation {operation.fn} not supported"
                )

        return [result]

    @lower_operation.register
    def lower_dummy(
        self, operation: ops.MakeDummy
    ) -> Sequence[mlir.Value]:
        result_type = ir_type_to_mlir_type(operation.result_var.get_type())
        dummy = mlir_constant_of_type(result_type, 0)
        return [dummy]

    @lower_operation.register
    def lower_foreign_function(
        self, operation: ops.ForeignFunction
    ) -> Sequence[mlir.Value]:
        function_name = operation.function_name
        operands = tuple(self.get_var(operand) for operand in operation.operands_)
        operand_types = tuple(
            ir_type_to_mlir_type(operand.get_type()) for operand in operation.operands_
        )

        has_result = len(operation.result_vars) > 0
        result_type = (
            ir_type_to_mlir_type(operation.result_var.get_type())
            if has_result
            else None
        )
        function_type = mlir.llvm.LLVMFunctionType(
            returnType=result_type if result_type else mlir.llvm.LLVMVoidType(),
            params=operand_types,
            varArg=False,
        )
        if prev_type := self._seen_foreign_functions.get(function_name):
            if prev_type != function_type:
                raise TileTypeError(
                    f"Tried calling foreign function {function_name} with type {function_type} "
                    f"but it has already been declared with type {prev_type!r}"
                )
        else:
            mlir.llvm.add_LLVMFuncOp(
                sym_name=function_name,
                linkage=mlir.llvm.Linkage.External,
                body=mlir.Region(),
                function_type=function_type,
            )
            func_op = _Cursor.current()._block.operations.pop()
            self.gpu_module_op.regions[0].blocks[0].operations.insert(0, func_op)
            self._seen_foreign_functions[function_name] = function_type

        call = mlir.llvm.add_CallOp(
            result_type=result_type,
            callee=function_name,
            callee_operands=operands,
            op_bundle_operands=(),
            op_bundle_sizes=(),
        )
        return [call] if has_result else []


def ir2mlir(
    signature: KernelSignature, body: ir.Region, ctx: ir.IRContext
) -> mlir.Operation:
    lower = IR2MLIR(signature, body, ctx)
    op = lower()
    return op


__all__ = ("ir2mlir",)
