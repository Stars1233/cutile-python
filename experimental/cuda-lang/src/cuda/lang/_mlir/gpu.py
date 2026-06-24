# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from . import ArrayAttr
from . import DenseI32ArrayAttr
from . import DeviceMappingAttrInterface
from . import DeviceMaskingAttrInterface
from . import DictionaryAttr
from . import FlatSymbolRefAttr
from . import FunctionType
from . import IndexType
from . import IntegerAttr
from . import IntegerType
from . import MemRefType
from . import StringAttr
from . import SymbolRefAttr
from . import TypeAttr
from . import UnitAttr
from ._builtins import APInt
from ._builtins import AffineMap
from ._builtins import Attribute
from ._builtins import Region
from ._builtins import Type
from ._builtins import Value
from ._builtins import add_operation
from dataclasses import dataclass
from typing import Optional
from typing import Sequence
import dataclasses
import enum


# ========= 'gpu' dialect of MLIR ==========


# ---- Interfaces ----


class AsyncOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class OffloadingLLVMTranslationAttrInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class TargetAttrInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class TargetAttrVerifyInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


# ---- Enums ----


class AddressSpace(enum.Enum):
    Global = 1
    Workgroup = 2
    Private = 3

    def _print_mlir_unqualified(self, p):
        p(("", "global", "workgroup", "private",)[self._value_])


class AllReduceOperation(enum.Enum):
    ADD = 0
    MUL = 1
    MINUI = 2
    MINSI = 3
    MINNUMF = 4
    MAXUI = 5
    MAXSI = 6
    MAXNUMF = 7
    AND = 8
    OR = 9
    XOR = 10
    MINIMUMF = 11
    MAXIMUMF = 12

    def _print_mlir_unqualified(self, p):
        p(("add", "mul", "minui", "minsi", "minnumf", "maxui", "maxsi", "maxnumf", "and", "or",
           "xor", "minimumf", "maximumf",)[self._value_])


class BroadcastType(enum.Enum):
    first_active_lane = 0
    specific_lane = 1

    def _print_mlir_unqualified(self, p):
        p(("first_active_lane", "specific_lane",)[self._value_])


class CompilationTarget(enum.Enum):
    Offload = 1
    Assembly = 2
    Binary = 3
    Fatbin = 4

    def _print_mlir_unqualified(self, p):
        p(("", "offload", "assembly", "bin", "fatbin",)[self._value_])


class Dimension(enum.Enum):
    x = 0
    y = 1
    z = 2

    def _print_mlir_unqualified(self, p):
        p(("x", "y", "z",)[self._value_])


class MMAElementwiseOp(enum.Enum):
    ADDF = 0
    MULF = 1
    SUBF = 2
    MAXF = 3
    MINF = 4
    DIVF = 5
    ADDI = 6
    MULI = 7
    SUBI = 8
    DIVS = 9
    DIVU = 10
    NEGATEF = 11
    NEGATES = 12
    EXTF = 13

    def _print_mlir_unqualified(self, p):
        p(("addf", "mulf", "subf", "maxf", "minf", "divf", "addi", "muli", "subi", "divs", "divu",
           "negatef", "negates", "extf",)[self._value_])


class MappingId(enum.Enum):
    DimX = 0
    DimY = 1
    DimZ = 2
    LinearDim0 = 3
    LinearDim1 = 4
    LinearDim2 = 5
    LinearDim3 = 6
    LinearDim4 = 7
    LinearDim5 = 8
    LinearDim6 = 9
    LinearDim7 = 10
    LinearDim8 = 11
    LinearDim9 = 12

    def _print_mlir_unqualified(self, p):
        p(("x", "y", "z", "linear_dim_0", "linear_dim_1", "linear_dim_2", "linear_dim_3",
           "linear_dim_4", "linear_dim_5", "linear_dim_6", "linear_dim_7", "linear_dim_8",
           "linear_dim_9",)[self._value_])


class MappingIdAttr(IntegerAttr):
    def __init__(self, value: MappingId):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class Processor(enum.Enum):
    BlockX = 0
    BlockY = 1
    BlockZ = 2
    ThreadX = 3
    ThreadY = 4
    ThreadZ = 5
    Sequential = 6

    def _print_mlir_unqualified(self, p):
        p(("block_x", "block_y", "block_z", "thread_x", "thread_y", "thread_z",
           "sequential",)[self._value_])


class ProcessorAttr(IntegerAttr):
    def __init__(self, value: Processor):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class Prune2To4SpMatFlag(enum.Enum):
    NONE = 0
    PRUNE_ONLY = 1
    PRUNE_AND_CHECK = 2

    def _print_mlir_unqualified(self, p):
        p(("NONE", "PRUNE_ONLY", "PRUNE_AND_CHECK",)[self._value_])


class ShuffleMode(enum.Enum):
    XOR = 0
    UP = 2
    DOWN = 1
    IDX = 3

    def _print_mlir_unqualified(self, p):
        p(("xor", "down", "up", "idx",)[self._value_])


class SpGEMMWorkEstimationOrComputeKind(enum.Enum):
    WORK_ESTIMATION = 0
    COMPUTE = 1

    def _print_mlir_unqualified(self, p):
        p(("WORK_ESTIMATION", "COMPUTE",)[self._value_])


class TransposeMode(enum.Enum):
    NON_TRANSPOSE = 0
    TRANSPOSE = 1
    CONJUGATE_TRANSPOSE = 2

    def _print_mlir_unqualified(self, p):
        p(("NON_TRANSPOSE", "TRANSPOSE", "CONJUGATE_TRANSPOSE",)[self._value_])


# ---- Attributes ----


@dataclass(kw_only=True)
class AddressSpaceAttr(Attribute, dialect='gpu', mnemonic='address_space'):
    value: "AddressSpace"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class AllReduceOperationAttr(Attribute, dialect='gpu', mnemonic='all_reduce_op'):
    value: "AllReduceOperation"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class BroadcastTypeAttr(Attribute, dialect='gpu', mnemonic='broadcast'):
    value: "BroadcastType"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class DimensionAttr(Attribute, dialect='gpu', mnemonic='dim'):
    value: "Dimension"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class GPUBlockMappingAttr(Attribute, DeviceMappingAttrInterface, dialect='gpu', mnemonic='block'):
    block: "MappingId"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.block._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class GPULaneMappingAttr(Attribute, DeviceMappingAttrInterface, dialect='gpu', mnemonic='lane'):
    lane: "MappingId"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.lane._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class GPUMappingMaskAttr(Attribute, DeviceMaskingAttrInterface, dialect='gpu', mnemonic='mask'):
    mask: "int"

    def _print_mlir_unqualified(self, p):
        p("<")
        p(str(self.mask))
        p(">")


@dataclass(kw_only=True)
class GPUMemorySpaceMappingAttr(Attribute, DeviceMappingAttrInterface, dialect='gpu',
                                mnemonic='memory_space'):
    address_space: "AddressSpace"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.address_space._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class GPUThreadMappingAttr(Attribute, DeviceMappingAttrInterface, dialect='gpu',
                           mnemonic='thread'):
    thread: "MappingId"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.thread._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class GPUWarpMappingAttr(Attribute, DeviceMappingAttrInterface, dialect='gpu', mnemonic='warp'):
    warp: "MappingId"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.warp._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class GPUWarpgroupMappingAttr(Attribute, DeviceMappingAttrInterface, dialect='gpu',
                              mnemonic='warpgroup'):
    warpgroup: "MappingId"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.warpgroup._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class KernelMetadataAttr(Attribute, dialect='gpu', mnemonic='kernel_metadata'):
    name: "StringAttr"
    function_type: "Type"
    arg_attrs: Optional["ArrayAttr"] = None
    metadata: Optional["DictionaryAttr"] = None

    def _print_mlir_unqualified(self, p):
        p("<")
        self.name._print_mlir_unqualified(p)
        p(", ")
        self.function_type._print_mlir_unqualified(p)
        if (self.arg_attrs is not None or self.metadata is not None):
            p(",")
            comma = ""
            if self.arg_attrs is not None:
                p("arg_attrs = ")
                self.arg_attrs._print_mlir_unqualified(p)
                comma = ", "
            if self.metadata is not None:
                p(comma)
                p("metadata = ")
                self.metadata._print_mlir_unqualified(p)
        else:
            pass
        p(">")


@dataclass(kw_only=True)
class KernelTableAttr(Attribute, dialect='gpu', mnemonic='kernel_table'):
    kernel_table: "Sequence[KernelMetadataAttr]" = ()

    def _print_mlir_unqualified(self, p):
        p("<")
        if self.kernel_table != ():
            p("[")
            if self.kernel_table != ():
                p.print_qualified_attr(self.kernel_table)
            p("]")
        else:
            pass
        p(">")


@dataclass(kw_only=True)
class MMAElementwiseOpAttr(Attribute, dialect='gpu', mnemonic='mma_element_wise'):
    value: "MMAElementwiseOp"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class ObjectAttr(Attribute, dialect='gpu', mnemonic='object'):
    target: "Attribute"
    format: "CompilationTarget" = dataclasses.field(default_factory=lambda: CompilationTarget(4))
    object: "StringAttr"
    properties: Optional["DictionaryAttr"] = None
    kernels: Optional["KernelTableAttr"] = None

    def _print_mlir_unqualified(self, p):
        p("<")
        self.target._print_mlir_unqualified(p)
        p(",")
        if self.properties is not None:
            p(" properties =")
            if self.properties is not None:
                p(" ")
                self.properties._print_mlir_unqualified(p)
            p(",")
        else:
            pass
        if self.kernels is not None:
            p(" kernels =")
            if self.kernels is not None:
                p(" ")
                self.kernels._print_mlir_unqualified(p)
            p(",")
        else:
            pass
        p(" ")
        p.print_custom_Object(self.format, self.object)
        p(">")


@dataclass(kw_only=True)
class ParallelLoopDimMappingAttr(Attribute, dialect='gpu', mnemonic='loop_dim_map'):
    processor: "Processor"
    map: "AffineMap"
    bound: "AffineMap"

    def _print_mlir_unqualified(self, p):
        p("<processor = ")
        self.processor._print_mlir_unqualified(p)
        p(", map = ")
        self.map._print_mlir_unqualified(p)
        p(", bound = ")
        self.bound._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class Prune2To4SpMatFlagAttr(Attribute, dialect='gpu', mnemonic='prune_2to4_spmat_flag'):
    value: "Prune2To4SpMatFlag"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class SelectObjectAttr(Attribute, dialect='gpu', mnemonic='select_object'):
    target: "Attribute" = None

    def _print_mlir_unqualified(self, p):
        if self.target is not None:
            p("<")
            if self.target is not None:
                self.target._print_mlir_unqualified(p)
            p(">")
        else:
            pass


@dataclass(kw_only=True)
class ShuffleModeAttr(Attribute, dialect='gpu', mnemonic='shuffle_mode'):
    value: "ShuffleMode"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class SpGEMMWorkEstimationOrComputeKindAttr(Attribute, dialect='gpu',
                                            mnemonic='spgemm_work_estimation_or_compute_kind'):
    value: "SpGEMMWorkEstimationOrComputeKind"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class TransposeModeAttr(Attribute, dialect='gpu', mnemonic='mat_transpose_mode'):
    value: "TransposeMode"

    def _print_mlir_unqualified(self, p):
        self.value._print_mlir_unqualified(p)


# ---- Operators ----


def add_AllReduceOp(
    *,
    value: Value,
    op: Optional[AllReduceOperation] = None,
    uniform: bool = False,
    body: Region,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = value.type
    all_props = []
    if op is not None:
        all_props.append(('op', AllReduceOperationAttr(value=op)))
    if uniform:
        all_props.append(('uniform', UnitAttr()))
    return add_operation(
        name="gpu.all_reduce",
        result_type=result_type,
        operands=[value],
        properties=all_props,
        attributes=extra_attributes,
        regions=[body],
    )


def add_AllocOp(
    *,
    memref_type: MemRefType,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    dynamicSizes: Sequence[Value],
    symbolOperands: Sequence[Value],
    hostShared: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Optional[Value]]:
    all_props = []
    if hostShared:
        all_props.append(('hostShared', UnitAttr()))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([len(asyncDependencies), len(dynamicSizes),
                                         len(symbolOperands)])))
    return add_operation(
        name="gpu.alloc",
        result_type=(memref_type, asyncToken_type),
        operands=[*asyncDependencies, *dynamicSizes, *symbolOperands],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BarrierOp(
    *,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="gpu.barrier",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BinaryOp(
    *,
    sym_name: str,
    offloadingHandler: Optional[Attribute] = None,
    objects: ArrayAttr,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('sym_name', StringAttr(value=sym_name)))
    if offloadingHandler is not None:
        all_props.append(('offloadingHandler', offloadingHandler))
    all_props.append(('objects', objects))
    return add_operation(
        name="gpu.binary",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BlockDimOp(
    *,
    dimension: Dimension,
    upper_bound: Optional[APInt] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = IndexType()
    all_props = []
    all_props.append(('dimension', DimensionAttr(value=dimension)))
    if upper_bound is not None:
        all_props.append(('upper_bound', IntegerAttr(type=IndexType(), value=upper_bound)))
    return add_operation(
        name="gpu.block_dim",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BlockIdOp(
    *,
    dimension: Dimension,
    upper_bound: Optional[APInt] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = IndexType()
    all_props = []
    all_props.append(('dimension', DimensionAttr(value=dimension)))
    if upper_bound is not None:
        all_props.append(('upper_bound', IntegerAttr(type=IndexType(), value=upper_bound)))
    return add_operation(
        name="gpu.block_id",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterBlockIdOp(
    *,
    dimension: Dimension,
    upper_bound: Optional[APInt] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = IndexType()
    all_props = []
    all_props.append(('dimension', DimensionAttr(value=dimension)))
    if upper_bound is not None:
        all_props.append(('upper_bound', IntegerAttr(type=IndexType(), value=upper_bound)))
    return add_operation(
        name="gpu.cluster_block_id",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterDimBlocksOp(
    *,
    dimension: Dimension,
    upper_bound: Optional[APInt] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = IndexType()
    all_props = []
    all_props.append(('dimension', DimensionAttr(value=dimension)))
    if upper_bound is not None:
        all_props.append(('upper_bound', IntegerAttr(type=IndexType(), value=upper_bound)))
    return add_operation(
        name="gpu.cluster_dim_blocks",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterDimOp(
    *,
    dimension: Dimension,
    upper_bound: Optional[APInt] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = IndexType()
    all_props = []
    all_props.append(('dimension', DimensionAttr(value=dimension)))
    if upper_bound is not None:
        all_props.append(('upper_bound', IntegerAttr(type=IndexType(), value=upper_bound)))
    return add_operation(
        name="gpu.cluster_dim",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ClusterIdOp(
    *,
    dimension: Dimension,
    upper_bound: Optional[APInt] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = IndexType()
    all_props = []
    all_props.append(('dimension', DimensionAttr(value=dimension)))
    if upper_bound is not None:
        all_props.append(('upper_bound', IntegerAttr(type=IndexType(), value=upper_bound)))
    return add_operation(
        name="gpu.cluster_id",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Create2To4SpMatOp(
    *,
    spMat_type: Type,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    rows: Value,
    cols: Value,
    pruneFlag: Prune2To4SpMatFlag = Prune2To4SpMatFlag(2),
    memref: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Optional[Value]]:
    all_props = []
    all_props.append(('pruneFlag', Prune2To4SpMatFlagAttr(value=pruneFlag)))
    return add_operation(
        name="gpu.create_2to4_spmat",
        result_type=(spMat_type, asyncToken_type),
        operands=[*asyncDependencies, rows, cols, memref],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CreateBsrOp(
    *,
    spmat_type: Type,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    brows: Value,
    bcols: Value,
    bnnz: Value,
    rBlockSize: Value,
    cBlockSize: Value,
    bRowPos: Value,
    bColIdxs: Value,
    values: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Optional[Value]]:
    all_props = []
    return add_operation(
        name="gpu.create_bsr",
        result_type=(spmat_type, asyncToken_type),
        operands=[*asyncDependencies, brows, bcols, bnnz, rBlockSize, cBlockSize, bRowPos,
                  bColIdxs, values],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CreateCooAoSOp(
    *,
    spmat_type: Type,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    rows: Value,
    cols: Value,
    nnz: Value,
    idxs: Value,
    values: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Optional[Value]]:
    all_props = []
    return add_operation(
        name="gpu.create_coo_aos",
        result_type=(spmat_type, asyncToken_type),
        operands=[*asyncDependencies, rows, cols, nnz, idxs, values],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CreateCooOp(
    *,
    spmat_type: Type,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    rows: Value,
    cols: Value,
    nnz: Value,
    rowIdxs: Value,
    colIdxs: Value,
    values: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Optional[Value]]:
    all_props = []
    return add_operation(
        name="gpu.create_coo",
        result_type=(spmat_type, asyncToken_type),
        operands=[*asyncDependencies, rows, cols, nnz, rowIdxs, colIdxs, values],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CreateCscOp(
    *,
    spmat_type: Type,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    rows: Value,
    cols: Value,
    nnz: Value,
    colPos: Value,
    rowIdxs: Value,
    values: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Optional[Value]]:
    all_props = []
    return add_operation(
        name="gpu.create_csc",
        result_type=(spmat_type, asyncToken_type),
        operands=[*asyncDependencies, rows, cols, nnz, colPos, rowIdxs, values],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CreateCsrOp(
    *,
    spmat_type: Type,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    rows: Value,
    cols: Value,
    nnz: Value,
    rowPos: Value,
    colIdxs: Value,
    values: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Optional[Value]]:
    all_props = []
    return add_operation(
        name="gpu.create_csr",
        result_type=(spmat_type, asyncToken_type),
        operands=[*asyncDependencies, rows, cols, nnz, rowPos, colIdxs, values],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CreateDnTensorOp(
    *,
    dnTensor_type: Type,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    memref: Value,
    dims: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Optional[Value]]:
    all_props = []
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([len(asyncDependencies), 1, len(dims)])))
    return add_operation(
        name="gpu.create_dn_tensor",
        result_type=(dnTensor_type, asyncToken_type),
        operands=[*asyncDependencies, memref, *dims],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DeallocOp(
    *,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    memref: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    return add_operation(
        name="gpu.dealloc",
        result_type=asyncToken_type,
        operands=[*asyncDependencies, memref],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DestroyDnTensorOp(
    *,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    dnTensor: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    return add_operation(
        name="gpu.destroy_dn_tensor",
        result_type=asyncToken_type,
        operands=[*asyncDependencies, dnTensor],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DestroySpMatOp(
    *,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    spmat: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    return add_operation(
        name="gpu.destroy_sp_mat",
        result_type=asyncToken_type,
        operands=[*asyncDependencies, spmat],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DynamicSharedMemoryOp(
    *,
    resultMemref_type: MemRefType,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="gpu.dynamic_shared_memory",
        result_type=resultMemref_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GPUFuncOp(
    *,
    function_type: FunctionType,
    arg_attrs: Optional[ArrayAttr] = None,
    res_attrs: Optional[ArrayAttr] = None,
    workgroup_attrib_attrs: Optional[ArrayAttr] = None,
    private_attrib_attrs: Optional[ArrayAttr] = None,
    known_block_size: Optional[Sequence[int]] = None,
    known_grid_size: Optional[Sequence[int]] = None,
    known_cluster_size: Optional[Sequence[int]] = None,
    body: Region,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('function_type', TypeAttr(value=function_type)))
    if arg_attrs is not None:
        all_props.append(('arg_attrs', arg_attrs))
    if res_attrs is not None:
        all_props.append(('res_attrs', res_attrs))
    if workgroup_attrib_attrs is not None:
        all_props.append(('workgroup_attrib_attrs', workgroup_attrib_attrs))
    if private_attrib_attrs is not None:
        all_props.append(('private_attrib_attrs', private_attrib_attrs))
    if known_block_size is not None:
        all_props.append(('known_block_size', DenseI32ArrayAttr(known_block_size)))
    if known_grid_size is not None:
        all_props.append(('known_grid_size', DenseI32ArrayAttr(known_grid_size)))
    if known_cluster_size is not None:
        all_props.append(('known_cluster_size', DenseI32ArrayAttr(known_cluster_size)))
    return add_operation(
        name="gpu.func",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
        regions=[body],
    )


def add_GPUModuleOp(
    *,
    sym_name: str,
    targets: Optional[ArrayAttr] = None,
    offloadingHandler: Optional[Attribute] = None,
    bodyRegion: Region,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('sym_name', StringAttr(value=sym_name)))
    if targets is not None:
        all_props.append(('targets', targets))
    if offloadingHandler is not None:
        all_props.append(('offloadingHandler', offloadingHandler))
    return add_operation(
        name="gpu.module",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
        regions=[bodyRegion],
    )


def add_GlobalIdOp(
    *,
    dimension: Dimension,
    upper_bound: Optional[APInt] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = IndexType()
    all_props = []
    all_props.append(('dimension', DimensionAttr(value=dimension)))
    if upper_bound is not None:
        all_props.append(('upper_bound', IntegerAttr(type=IndexType(), value=upper_bound)))
    return add_operation(
        name="gpu.global_id",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GridDimOp(
    *,
    dimension: Dimension,
    upper_bound: Optional[APInt] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = IndexType()
    all_props = []
    all_props.append(('dimension', DimensionAttr(value=dimension)))
    if upper_bound is not None:
        all_props.append(('upper_bound', IntegerAttr(type=IndexType(), value=upper_bound)))
    return add_operation(
        name="gpu.grid_dim",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_HostRegisterOp(
    *,
    value: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="gpu.host_register",
        result_type=None,
        operands=[value],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_HostUnregisterOp(
    *,
    value: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="gpu.host_unregister",
        result_type=None,
        operands=[value],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LaneIdOp(
    *,
    upper_bound: Optional[APInt] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = IndexType()
    all_props = []
    if upper_bound is not None:
        all_props.append(('upper_bound', IntegerAttr(type=IndexType(), value=upper_bound)))
    return add_operation(
        name="gpu.lane_id",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LaunchFuncOp(
    *,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    kernel: SymbolRefAttr,
    gridSizeX: Value,
    gridSizeY: Value,
    gridSizeZ: Value,
    blockSizeX: Value,
    blockSizeY: Value,
    blockSizeZ: Value,
    clusterSizeX: Optional[Value] = None,
    clusterSizeY: Optional[Value] = None,
    clusterSizeZ: Optional[Value] = None,
    dynamicSharedMemorySize: Optional[Value] = None,
    kernelOperands: Sequence[Value],
    asyncObject: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    all_props.append(('kernel', kernel))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([len(asyncDependencies), 1, 1, 1, 1, 1, 1,
                                         int(clusterSizeX is not None),
                                         int(clusterSizeY is not None),
                                         int(clusterSizeZ is not None),
                                         int(dynamicSharedMemorySize is not None),
                                         len(kernelOperands), int(asyncObject is not None)])))
    return add_operation(
        name="gpu.launch_func",
        result_type=asyncToken_type,
        operands=[*asyncDependencies, gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY,
                  blockSizeZ, *([] if clusterSizeX is None else [clusterSizeX]),
                  *([] if clusterSizeY is None else [clusterSizeY]),
                  *([] if clusterSizeZ is None else [clusterSizeZ]),
                  *([] if dynamicSharedMemorySize is None else [dynamicSharedMemorySize]),
                  *kernelOperands, *([] if asyncObject is None else [asyncObject])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LaunchOp(
    *,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    gridSizeX: Value,
    gridSizeY: Value,
    gridSizeZ: Value,
    blockSizeX: Value,
    blockSizeY: Value,
    blockSizeZ: Value,
    clusterSizeX: Optional[Value] = None,
    clusterSizeY: Optional[Value] = None,
    clusterSizeZ: Optional[Value] = None,
    dynamicSharedMemorySize: Optional[Value] = None,
    module: Optional[str] = None,
    function: Optional[str] = None,
    body: Region,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    if module is not None:
        all_props.append(('module', FlatSymbolRefAttr(module)))
    if function is not None:
        all_props.append(('function', FlatSymbolRefAttr(function)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([len(asyncDependencies), 1, 1, 1, 1, 1, 1,
                                         int(clusterSizeX is not None),
                                         int(clusterSizeY is not None),
                                         int(clusterSizeZ is not None),
                                         int(dynamicSharedMemorySize is not None)])))
    return add_operation(
        name="gpu.launch",
        result_type=asyncToken_type,
        operands=[*asyncDependencies, gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY,
                  blockSizeZ, *([] if clusterSizeX is None else [clusterSizeX]),
                  *([] if clusterSizeY is None else [clusterSizeY]),
                  *([] if clusterSizeZ is None else [clusterSizeZ]),
                  *([] if dynamicSharedMemorySize is None else [dynamicSharedMemorySize])],
        properties=all_props,
        attributes=extra_attributes,
        regions=[body],
    )


def add_MemcpyOp(
    *,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    dst: Value,
    src: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    return add_operation(
        name="gpu.memcpy",
        result_type=asyncToken_type,
        operands=[*asyncDependencies, dst, src],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MemsetOp(
    *,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    dst: Value,
    value: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    return add_operation(
        name="gpu.memset",
        result_type=asyncToken_type,
        operands=[*asyncDependencies, dst, value],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_NumSubgroupsOp(
    *,
    upper_bound: Optional[APInt] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = IndexType()
    all_props = []
    if upper_bound is not None:
        all_props.append(('upper_bound', IntegerAttr(type=IndexType(), value=upper_bound)))
    return add_operation(
        name="gpu.num_subgroups",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_PrintfOp(
    *,
    format: str,
    args: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('format', StringAttr(value=format)))
    return add_operation(
        name="gpu.printf",
        result_type=None,
        operands=list(args),
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ReturnOp(
    *,
    operands: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="gpu.return",
        result_type=None,
        operands=list(operands),
        properties=all_props,
        attributes=extra_attributes,
    )


def add_RotateOp(
    *,
    value: Value,
    offset: int,
    width: int,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Value]:
    rotateResult_type = value.type
    valid_type = IntegerType.signless(1)
    all_props = []
    all_props.append(('offset', IntegerAttr.make(IntegerType.signless(32), offset)))
    all_props.append(('width', IntegerAttr.make(IntegerType.signless(32), width)))
    return add_operation(
        name="gpu.rotate",
        result_type=(rotateResult_type, valid_type),
        operands=[value],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SDDMMBufferSizeOp(
    *,
    bufferSz_type: IndexType,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    modeA: TransposeMode = TransposeMode(0),
    modeB: TransposeMode = TransposeMode(0),
    dnmatA: Value,
    dnmatB: Value,
    spmatC: Value,
    computeType: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Optional[Value]]:
    all_props = []
    all_props.append(('modeA', TransposeModeAttr(value=modeA)))
    all_props.append(('modeB', TransposeModeAttr(value=modeB)))
    all_props.append(('computeType', TypeAttr(value=computeType)))
    return add_operation(
        name="gpu.sddmm_buffer_size",
        result_type=(bufferSz_type, asyncToken_type),
        operands=[*asyncDependencies, dnmatA, dnmatB, spmatC],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SDDMMOp(
    *,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    modeA: TransposeMode = TransposeMode(0),
    modeB: TransposeMode = TransposeMode(0),
    dnmatA: Value,
    dnmatB: Value,
    spmatC: Value,
    computeType: Type,
    buffer: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    all_props.append(('modeA', TransposeModeAttr(value=modeA)))
    all_props.append(('modeB', TransposeModeAttr(value=modeB)))
    all_props.append(('computeType', TypeAttr(value=computeType)))
    return add_operation(
        name="gpu.sddmm",
        result_type=asyncToken_type,
        operands=[*asyncDependencies, dnmatA, dnmatB, spmatC, buffer],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SetCsrPointersOp(
    *,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    spmat: Value,
    positions: Value,
    coordinates: Value,
    values: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    return add_operation(
        name="gpu.set_csr_pointers",
        result_type=asyncToken_type,
        operands=[*asyncDependencies, spmat, positions, coordinates, values],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SetDefaultDeviceOp(
    *,
    devIndex: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="gpu.set_default_device",
        result_type=None,
        operands=[devIndex],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ShuffleOp(
    *,
    value: Value,
    offset: Value,
    width: Value,
    mode: ShuffleMode,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Value]:
    shuffleResult_type = value.type
    valid_type = IntegerType.signless(1)
    all_props = []
    all_props.append(('mode', ShuffleModeAttr(value=mode)))
    return add_operation(
        name="gpu.shuffle",
        result_type=(shuffleResult_type, valid_type),
        operands=[value, offset, width],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SpGEMMCopyOp(
    *,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    desc: Value,
    modeA: TransposeMode = TransposeMode(0),
    modeB: TransposeMode = TransposeMode(0),
    spmatA: Value,
    spmatB: Value,
    spmatC: Value,
    computeType: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    all_props.append(('modeA', TransposeModeAttr(value=modeA)))
    all_props.append(('modeB', TransposeModeAttr(value=modeB)))
    all_props.append(('computeType', TypeAttr(value=computeType)))
    return add_operation(
        name="gpu.spgemm_copy",
        result_type=asyncToken_type,
        operands=[*asyncDependencies, desc, spmatA, spmatB, spmatC],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SpGEMMCreateDescrOp(
    *,
    desc_type: Type,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Optional[Value]]:
    all_props = []
    return add_operation(
        name="gpu.spgemm_create_descr",
        result_type=(desc_type, asyncToken_type),
        operands=list(asyncDependencies),
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SpGEMMDestroyDescrOp(
    *,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    desc: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    return add_operation(
        name="gpu.spgemm_destroy_descr",
        result_type=asyncToken_type,
        operands=[*asyncDependencies, desc],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SpGEMMWorkEstimationOrComputeOp(
    *,
    bufferSzNew_type: IndexType,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    desc: Value,
    modeA: TransposeMode = TransposeMode(0),
    modeB: TransposeMode = TransposeMode(0),
    spmatA: Value,
    spmatB: Value,
    spmatC: Value,
    computeType: Type,
    bufferSz: Value,
    buffer: Value,
    kind: SpGEMMWorkEstimationOrComputeKind,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Optional[Value]]:
    all_props = []
    all_props.append(('modeA', TransposeModeAttr(value=modeA)))
    all_props.append(('modeB', TransposeModeAttr(value=modeB)))
    all_props.append(('computeType', TypeAttr(value=computeType)))
    all_props.append(('kind', SpGEMMWorkEstimationOrComputeKindAttr(value=kind)))
    return add_operation(
        name="gpu.spgemm_work_estimation_or_compute",
        result_type=(bufferSzNew_type, asyncToken_type),
        operands=[*asyncDependencies, desc, spmatA, spmatB, spmatC, bufferSz, buffer],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SpMMBufferSizeOp(
    *,
    bufferSzs_types: Sequence[IndexType],
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    modeA: TransposeMode = TransposeMode(0),
    modeB: TransposeMode = TransposeMode(0),
    spmatA: Value,
    dnmatB: Value,
    dnmatC: Value,
    computeType: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[tuple[Value, ...], Optional[Value]]:
    all_props = []
    all_props.append(('modeA', TransposeModeAttr(value=modeA)))
    all_props.append(('modeB', TransposeModeAttr(value=modeB)))
    all_props.append(('computeType', TypeAttr(value=computeType)))
    return add_operation(
        name="gpu.spmm_buffer_size",
        result_type=(bufferSzs_types, asyncToken_type),
        operands=[*asyncDependencies, spmatA, dnmatB, dnmatC],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SpMMOp(
    *,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    modeA: TransposeMode = TransposeMode(0),
    modeB: TransposeMode = TransposeMode(0),
    spmatA: Value,
    dnmatB: Value,
    dnmatC: Value,
    computeType: Type,
    buffers: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    all_props.append(('modeA', TransposeModeAttr(value=modeA)))
    all_props.append(('modeB', TransposeModeAttr(value=modeB)))
    all_props.append(('computeType', TypeAttr(value=computeType)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([len(asyncDependencies), 1, 1, 1, len(buffers)])))
    return add_operation(
        name="gpu.spmm",
        result_type=asyncToken_type,
        operands=[*asyncDependencies, spmatA, dnmatB, dnmatC, *buffers],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SpMVBufferSizeOp(
    *,
    bufferSz_type: IndexType,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    modeA: TransposeMode = TransposeMode(0),
    spmatA: Value,
    dnX: Value,
    dnY: Value,
    computeType: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Optional[Value]]:
    all_props = []
    all_props.append(('modeA', TransposeModeAttr(value=modeA)))
    all_props.append(('computeType', TypeAttr(value=computeType)))
    return add_operation(
        name="gpu.spmv_buffer_size",
        result_type=(bufferSz_type, asyncToken_type),
        operands=[*asyncDependencies, spmatA, dnX, dnY],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SpMVOp(
    *,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    modeA: TransposeMode = TransposeMode(0),
    spmatA: Value,
    dnX: Value,
    dnY: Value,
    computeType: Type,
    buffer: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    all_props.append(('modeA', TransposeModeAttr(value=modeA)))
    all_props.append(('computeType', TypeAttr(value=computeType)))
    return add_operation(
        name="gpu.spmv",
        result_type=asyncToken_type,
        operands=[*asyncDependencies, spmatA, dnX, dnY, buffer],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SpMatGetSizeOp(
    *,
    rows_type: IndexType,
    cols_type: IndexType,
    nnz_type: IndexType,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    spmat: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Value, Value, Optional[Value]]:
    all_props = []
    return add_operation(
        name="gpu.spmat_get_size",
        result_type=(rows_type, cols_type, nnz_type, asyncToken_type),
        operands=[*asyncDependencies, spmat],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SubgroupBroadcastOp(
    *,
    src: Value,
    lane: Optional[Value] = None,
    broadcast_type: BroadcastType,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = src.type
    all_props = []
    all_props.append(('broadcast_type', BroadcastTypeAttr(value=broadcast_type)))
    return add_operation(
        name="gpu.subgroup_broadcast",
        result_type=result_type,
        operands=[src, *([] if lane is None else [lane])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SubgroupIdOp(
    *,
    upper_bound: Optional[APInt] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = IndexType()
    all_props = []
    if upper_bound is not None:
        all_props.append(('upper_bound', IntegerAttr(type=IndexType(), value=upper_bound)))
    return add_operation(
        name="gpu.subgroup_id",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SubgroupMmaComputeOp(
    *,
    opA: Value,
    opB: Value,
    opC: Value,
    a_transpose: Optional[bool] = None,
    b_transpose: Optional[bool] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = opC.type
    all_props = []
    if a_transpose is not None:
        all_props.append(('a_transpose', UnitAttr()))
    if b_transpose is not None:
        all_props.append(('b_transpose', UnitAttr()))
    return add_operation(
        name="gpu.subgroup_mma_compute",
        result_type=res_type,
        operands=[opA, opB, opC],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SubgroupMmaConstantMatrixOp(
    *,
    res_type: Type,
    value: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="gpu.subgroup_mma_constant_matrix",
        result_type=res_type,
        operands=[value],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SubgroupMmaElementwiseOp(
    *,
    res_type: Type,
    args: Sequence[Value],
    opType: MMAElementwiseOp,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('opType', MMAElementwiseOpAttr(value=opType)))
    return add_operation(
        name="gpu.subgroup_mma_elementwise",
        result_type=res_type,
        operands=list(args),
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SubgroupMmaExtractThreadLocalOp(
    *,
    matrix: Value,
    indices: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = matrix.type.get_element_type()
    all_props = []
    return add_operation(
        name="gpu.subgroup_mma_extract_thread_local",
        result_type=res_type,
        operands=[matrix, *indices],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SubgroupMmaInsertThreadLocalOp(
    *,
    res_type: Type,
    value: Value,
    matrix: Value,
    indices: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="gpu.subgroup_mma_insert_thread_local",
        result_type=res_type,
        operands=[value, matrix, *indices],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SubgroupMmaLoadMatrixOp(
    *,
    res_type: Type,
    srcMemref: Value,
    indices: Sequence[Value],
    leadDimension: APInt,
    transpose: Optional[bool] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('leadDimension', IntegerAttr(type=IndexType(), value=leadDimension)))
    if transpose is not None:
        all_props.append(('transpose', UnitAttr()))
    return add_operation(
        name="gpu.subgroup_mma_load_matrix",
        result_type=res_type,
        operands=[srcMemref, *indices],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SubgroupMmaStoreMatrixOp(
    *,
    src: Value,
    dstMemref: Value,
    indices: Sequence[Value],
    leadDimension: APInt,
    transpose: Optional[bool] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('leadDimension', IntegerAttr(type=IndexType(), value=leadDimension)))
    if transpose is not None:
        all_props.append(('transpose', UnitAttr()))
    return add_operation(
        name="gpu.subgroup_mma_store_matrix",
        result_type=None,
        operands=[src, dstMemref, *indices],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SubgroupReduceOp(
    *,
    value: Value,
    op: AllReduceOperation,
    uniform: bool = False,
    cluster_size: Optional[int] = None,
    cluster_stride: int = 1,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = value.type
    all_props = []
    all_props.append(('op', AllReduceOperationAttr(value=op)))
    if uniform:
        all_props.append(('uniform', UnitAttr()))
    if cluster_size is not None:
        all_props.append(('cluster_size',
                          IntegerAttr.make(IntegerType.signless(32), cluster_size)))
    all_props.append(('cluster_stride',
                      IntegerAttr.make(IntegerType.signless(32), cluster_stride)))
    return add_operation(
        name="gpu.subgroup_reduce",
        result_type=result_type,
        operands=[value],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SubgroupSizeOp(
    *,
    upper_bound: Optional[APInt] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = IndexType()
    all_props = []
    if upper_bound is not None:
        all_props.append(('upper_bound', IntegerAttr(type=IndexType(), value=upper_bound)))
    return add_operation(
        name="gpu.subgroup_size",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_TerminatorOp(
    *,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="gpu.terminator",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ThreadIdOp(
    *,
    dimension: Dimension,
    upper_bound: Optional[APInt] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = IndexType()
    all_props = []
    all_props.append(('dimension', DimensionAttr(value=dimension)))
    if upper_bound is not None:
        all_props.append(('upper_bound', IntegerAttr(type=IndexType(), value=upper_bound)))
    return add_operation(
        name="gpu.thread_id",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_WaitOp(
    *,
    asyncToken_type: Optional[Type],
    asyncDependencies: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    return add_operation(
        name="gpu.wait",
        result_type=asyncToken_type,
        operands=list(asyncDependencies),
        properties=all_props,
        attributes=extra_attributes,
    )


def add_WarpExecuteOnLane0Op(
    *,
    results_types: Sequence[Type],
    laneid: Value,
    warp_size: int,
    args: Sequence[Value],
    warpRegion: Region,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, ...]:
    all_props = []
    all_props.append(('warp_size', IntegerAttr.make(IntegerType.signless(64), warp_size)))
    return add_operation(
        name="gpu.warp_execute_on_lane_0",
        result_type=results_types,
        operands=[laneid, *args],
        properties=all_props,
        attributes=extra_attributes,
        regions=[warpRegion],
    )


def add_YieldOp(
    *,
    values: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="gpu.yield",
        result_type=None,
        operands=list(values),
        properties=all_props,
        attributes=extra_attributes,
    )
