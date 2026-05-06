# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from . import ArrayAttr
from . import BoolAttr
from . import DLTIQueryInterface
from . import DataLayoutTypeInterface
from . import DenseElementType
from . import DenseI32ArrayAttr
from . import DenseI64ArrayAttr
from . import DenseIntElementsAttr
from . import DestructurableTypeInterface
from . import DictionaryAttr
from . import DistinctAttr
from . import FlatSymbolRefAttr
from . import Float64Type
from . import FloatAttr
from . import FloatType
from . import FusedLoc
from . import IntegerAttr
from . import IntegerType
from . import StringAttr
from . import SymbolRefAttr
from . import TypeAttr
from . import UnitAttr
from . import VectorElementTypeInterface
from . import VectorType
from . import _util
from . import ptr
from ._builtins import APFloat
from ._builtins import APInt
from ._builtins import Attribute
from ._builtins import BlockLabel
from ._builtins import Region
from ._builtins import Type
from ._builtins import Value
from ._builtins import add_operation
from dataclasses import dataclass
from typing import Optional
from typing import Sequence
import dataclasses
import enum


# ========= 'llvm' dialect of MLIR ==========


# ---- Interfaces ----


class AccessGroupOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class AliasAnalysisOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class DereferenceableOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class DisjointFlagInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class ExactFlagInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class FPExceptionBehaviorOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class FastmathFlagsInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class IntegerOverflowFlagsInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class DIRecursiveTypeAttrInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class LLVMAddrSpaceAttrInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class PointerElementTypeInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class TargetAttrInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class NonNegFlagInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class OneToOneIntrinsicOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class RoundingModeOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


# ---- Enums ----


class AsmDialect(enum.Enum):
    AD_ATT = 0
    AD_Intel = 1

    def _print_mlir_unqualified(self, p):
        p(("att", "intel",)[self._value_])


class AsmDialectAttr(IntegerAttr):
    def __init__(self, value: AsmDialect):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class AtomicBinOp(enum.Enum):
    xchg = 0
    add = 1
    sub = 2
    _and = 3
    nand = 4
    _or = 5
    _xor = 6
    max = 7
    min = 8
    umax = 9
    umin = 10
    fadd = 11
    fsub = 12
    fmax = 13
    fmin = 14
    uinc_wrap = 15
    udec_wrap = 16
    usub_cond = 17
    usub_sat = 18
    fmaximum = 19
    fminimum = 20
    fmaximumnum = 21
    fminimumnum = 22

    def _print_mlir_unqualified(self, p):
        p(("xchg", "add", "sub", "_and", "nand", "_or", "_xor", "max", "min", "umax", "umin",
           "fadd", "fsub", "fmax", "fmin", "uinc_wrap", "udec_wrap", "usub_cond", "usub_sat",
           "fmaximum", "fminimum", "fmaximumnum", "fminimumnum",)[self._value_])


class AtomicBinOpAttr(IntegerAttr):
    def __init__(self, value: AtomicBinOp):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class AtomicOrdering(enum.Enum):
    not_atomic = 0
    unordered = 1
    monotonic = 2
    acquire = 4
    release = 5
    acq_rel = 6
    seq_cst = 7

    def _print_mlir_unqualified(self, p):
        p(("not_atomic", "unordered", "monotonic", "", "acquire", "release", "acq_rel",
           "seq_cst",)[self._value_])


class AtomicOrderingAttr(IntegerAttr):
    def __init__(self, value: AtomicOrdering):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class cconv_CConv(enum.Enum):
    C = 0
    Fast = 8
    Cold = 9
    GHC = 10
    HiPE = 11
    AnyReg = 13
    PreserveMost = 14
    PreserveAll = 15
    Swift = 16
    CXX_FAST_TLS = 17
    Tail = 18
    CFGuard_Check = 19
    SwiftTail = 20
    X86_StdCall = 64
    X86_FastCall = 65
    ARM_APCS = 66
    ARM_AAPCS = 67
    ARM_AAPCS_VFP = 68
    MSP430_INTR = 69
    X86_ThisCall = 70
    PTX_Kernel = 71
    PTX_Device = 72
    SPIR_FUNC = 75
    SPIR_KERNEL = 76
    Intel_OCL_BI = 77
    X86_64_SysV = 78
    Win64 = 79
    X86_VectorCall = 80
    DUMMY_HHVM = 81
    DUMMY_HHVM_C = 82
    X86_INTR = 83
    AVR_INTR = 84
    AVR_BUILTIN = 86
    AMDGPU_VS = 87
    AMDGPU_GS = 88
    AMDGPU_CS = 90
    AMDGPU_KERNEL = 91
    X86_RegCall = 92
    AMDGPU_HS = 93
    MSP430_BUILTIN = 94
    AMDGPU_LS = 95
    AMDGPU_ES = 96
    AArch64_VectorCall = 97
    AArch64_SVE_VectorCall = 98
    WASM_EmscriptenInvoke = 99
    AMDGPU_Gfx = 100
    M68k_INTR = 101

    def _print_mlir_unqualified(self, p):
        p(("ccc", "", "", "", "", "", "", "", "fastcc", "coldcc", "cc_10", "cc_11", "",
           "anyregcc", "preserve_mostcc", "preserve_allcc", "swiftcc", "cxx_fast_tlscc", "tailcc",
           "cfguard_checkcc", "swifttailcc", "", "", "", "", "", "", "", "", "", "", "", "", "",
           "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
           "", "", "", "", "", "", "", "", "x86_stdcallcc", "x86_fastcallcc", "arm_apcscc",
           "arm_aapcscc", "arm_aapcs_vfpcc", "msp430_intrcc", "x86_thiscallcc", "ptx_kernelcc",
           "ptx_devicecc", "", "", "spir_funccc", "spir_kernelcc", "intel_ocl_bicc",
           "x86_64_sysvcc", "win64cc", "x86_vectorcallcc", "hhvmcc", "hhvm_ccc", "x86_intrcc",
           "avr_intrcc", "", "avr_builtincc", "amdgpu_vscc", "amdgpu_gscc", "", "amdgpu_cscc",
           "amdgpu_kernelcc", "x86_regcallcc", "amdgpu_hscc", "msp430_builtincc", "amdgpu_lscc",
           "amdgpu_escc", "aarch64_vectorcallcc", "aarch64_sve_vectorcallcc",
           "wasm_emscripten_invokecc", "amdgpu_gfxcc", "m68k_intrcc",)[self._value_])


class cconv_CConvAttr(IntegerAttr):
    def __init__(self, value: cconv_CConv):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


CConv = cconv_CConv


class comdat_Comdat(enum.Enum):
    Any = 0
    ExactMatch = 1
    Largest = 2
    NoDeduplicate = 3
    SameSize = 4

    def _print_mlir_unqualified(self, p):
        p(("any", "exactmatch", "largest", "nodeduplicate", "samesize",)[self._value_])


class comdat_ComdatAttr(IntegerAttr):
    def __init__(self, value: comdat_Comdat):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class DIFlags(enum.IntFlag):
    Zero = 0x0
    Bit0 = 0x1
    Bit1 = 0x2
    Private = 0x1
    Protected = 0x2
    Public = 0x3
    FwdDecl = 0x4
    AppleBlock = 0x8
    ReservedBit4 = 0x10
    Virtual = 0x20
    Artificial = 0x40
    Explicit = 0x80
    Prototyped = 0x100
    ObjcClassComplete = 0x200
    ObjectPointer = 0x400
    Vector = 0x800
    StaticMember = 0x1000
    LValueReference = 0x2000
    RValueReference = 0x4000
    ExportSymbols = 0x8000
    SingleInheritance = 0x10000
    MultipleInheritance = 0x10000
    VirtualInheritance = 0x10000
    IntroducedVirtual = 0x40000
    BitField = 0x80000
    NoReturn = 0x100000
    TypePassByValue = 0x400000
    TypePassByReference = 0x800000
    EnumClass = 0x1000000
    Thunk = 0x2000000
    NonTrivial = 0x4000000
    BigEndian = 0x8000000
    LittleEndian = 0x10000000
    AllCallsDescribed = 0x20000000

    def _print_mlir_unqualified(self, p):
        value = int(self._value_)
        if value == 0:
            p('Zero')
            return
        p.print_bit_enum(value, ((0x1, 'Private'), (0x2, 'Protected'), (0x3, 'Public'),),
                         ((0x1, 'Bit0'), (0x2, 'Bit1'), (0x4, 'FwdDecl'), (0x8, 'AppleBlock'),
                          (0x10, 'ReservedBit4'), (0x20, 'Virtual'), (0x40, 'Artificial'),
                          (0x80, 'Explicit'), (0x100, 'Prototyped'), (0x200, 'ObjcClassComplete'),
                          (0x400, 'ObjectPointer'), (0x800, 'Vector'), (0x1000, 'StaticMember'),
                          (0x2000, 'LValueReference'), (0x4000, 'RValueReference'),
                          (0x8000, 'ExportSymbols'), (0x10000, 'SingleInheritance'),
                          (0x10000, 'MultipleInheritance'), (0x10000, 'VirtualInheritance'),
                          (0x40000, 'IntroducedVirtual'), (0x80000, 'BitField'),
                          (0x100000, 'NoReturn'), (0x400000, 'TypePassByValue'),
                          (0x800000, 'TypePassByReference'), (0x1000000, 'EnumClass'),
                          (0x2000000, 'Thunk'), (0x4000000, 'NonTrivial'),
                          (0x8000000, 'BigEndian'), (0x10000000, 'LittleEndian'),
                          (0x20000000, 'AllCallsDescribed'),))


class DIFlagsAttr(IntegerAttr):
    def __init__(self, value: DIFlags):
        super().__init__(type=IntegerType.signless(32), value=APInt(value._value_, 32))


class DISubprogramFlags(enum.IntFlag):
    Virtual = 0x1
    PureVirtual = 0x2
    LocalToUnit = 0x4
    Definition = 0x8
    Optimized = 0x10
    Pure = 0x20
    Elemental = 0x40
    Recursive = 0x80
    MainSubprogram = 0x100
    Deleted = 0x200
    ObjCDirect = 0x800

    def _print_mlir_unqualified(self, p):
        value = int(self._value_)
        p.print_bit_enum(value, (),
                         ((0x1, 'Virtual'), (0x2, 'PureVirtual'), (0x4, 'LocalToUnit'),
                          (0x8, 'Definition'), (0x10, 'Optimized'), (0x20, 'Pure'),
                          (0x40, 'Elemental'), (0x80, 'Recursive'), (0x100, 'MainSubprogram'),
                          (0x200, 'Deleted'), (0x800, 'ObjCDirect'),))


class DISubprogramFlagsAttr(IntegerAttr):
    def __init__(self, value: DISubprogramFlags):
        super().__init__(type=IntegerType.signless(32), value=APInt(value._value_, 32))


class DenormalModeKind(enum.Enum):
    IEEE = 0
    PreserveSign = 1
    PositiveZero = 2
    Dynamic = 3
    Invalid = -1

    def _print_mlir_unqualified(self, p):
        p(("ieee", "preservesign", "positivezero", "invalid",)[self._value_])


class DenormalModeKindAttr(IntegerAttr):
    def __init__(self, value: DenormalModeKind):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class FCmpPredicate(enum.Enum):
    _false = 0
    oeq = 1
    ogt = 2
    oge = 3
    olt = 4
    ole = 5
    one = 6
    ord = 7
    ueq = 8
    ugt = 9
    uge = 10
    ult = 11
    ule = 12
    une = 13
    uno = 14
    _true = 15

    def _print_mlir_unqualified(self, p):
        p(("_false", "oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq", "ugt", "uge", "ult",
           "ule", "une", "uno", "_true",)[self._value_])


class FCmpPredicateAttr(IntegerAttr):
    def __init__(self, value: FCmpPredicate):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class FPExceptionBehavior(enum.Enum):
    Ignore = 0
    MayTrap = 1
    Strict = 2

    def _print_mlir_unqualified(self, p):
        p(("ignore", "maytrap", "strict",)[self._value_])


class FPExceptionBehaviorAttr(IntegerAttr):
    def __init__(self, value: FPExceptionBehavior):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class FastmathFlags(enum.IntFlag):
    none = 0x0
    nnan = 0x1
    ninf = 0x2
    nsz = 0x4
    arcp = 0x8
    contract = 0x10
    afn = 0x20
    reassoc = 0x40
    fast = 0x7f

    def _print_mlir_unqualified(self, p):
        value = int(self._value_)
        if value == 0:
            p('none')
            return
        p.print_bit_enum(value, ((0x7f, 'fast'),),
                         ((0x1, 'nnan'), (0x2, 'ninf'), (0x4, 'nsz'), (0x8, 'arcp'),
                          (0x10, 'contract'), (0x20, 'afn'), (0x40, 'reassoc'),))


class framePointerKind_FramePointerKind(enum.Enum):
    None_ = 0
    NonLeaf = 1
    All = 2
    Reserved = 3
    NonLeafNoReserve = 4

    def _print_mlir_unqualified(self, p):
        p(("none", "non-leaf", "all", "reserved", "non-leaf-no-reserve",)[self._value_])


class framePointerKind_FramePointerKindAttr(IntegerAttr):
    def __init__(self, value: framePointerKind_FramePointerKind):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class GEPNoWrapFlags(enum.IntFlag):
    none = 0x0
    inboundsFlag = 0x1
    nusw = 0x2
    nuw = 0x4
    inbounds = 0x3

    def _print_mlir_unqualified(self, p):
        value = int(self._value_)
        if value == 0:
            p('none')
            return
        p.print_bit_enum(value, ((0x3, 'inbounds'),),
                         ((0x1, 'inbounds_flag'), (0x2, 'nusw'), (0x4, 'nuw'),))


class ICmpPredicate(enum.Enum):
    eq = 0
    ne = 1
    slt = 2
    sle = 3
    sgt = 4
    sge = 5
    ult = 6
    ule = 7
    ugt = 8
    uge = 9

    def _print_mlir_unqualified(self, p):
        p(("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge",)[self._value_])


class ICmpPredicateAttr(IntegerAttr):
    def __init__(self, value: ICmpPredicate):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class IntegerOverflowFlags(enum.IntFlag):
    none = 0x0
    nsw = 0x1
    nuw = 0x2

    def _print_mlir_unqualified(self, p):
        value = int(self._value_)
        if value == 0:
            p('none')
            return
        p.print_bit_enum(value, (), ((0x1, 'nsw'), (0x2, 'nuw'),))


class DIEmissionKind(enum.Enum):
    None_ = 0
    Full = 1
    LineTablesOnly = 2
    DebugDirectivesOnly = 3

    def _print_mlir_unqualified(self, p):
        p(("None", "Full", "LineTablesOnly", "DebugDirectivesOnly",)[self._value_])


class DIEmissionKindAttr(IntegerAttr):
    def __init__(self, value: DIEmissionKind):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class DINameTableKind(enum.Enum):
    Default = 0
    GNU = 1
    None_ = 2
    Apple = 3

    def _print_mlir_unqualified(self, p):
        p(("Default", "GNU", "None", "Apple",)[self._value_])


class DINameTableKindAttr(IntegerAttr):
    def __init__(self, value: DINameTableKind):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class ProfileSummaryFormatKind(enum.Enum):
    SampleProfile = 0
    InstrProf = 1
    CSInstrProf = 2

    def _print_mlir_unqualified(self, p):
        p(("SampleProfile", "InstrProf", "CSInstrProf",)[self._value_])


class ProfileSummaryFormatKindAttr(IntegerAttr):
    def __init__(self, value: ProfileSummaryFormatKind):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class linkage_Linkage(enum.Enum):
    External = 0
    AvailableExternally = 1
    Linkonce = 2
    LinkonceODR = 3
    Weak = 4
    WeakODR = 5
    Appending = 6
    Internal = 7
    Private = 8
    ExternWeak = 9
    Common = 10

    def _print_mlir_unqualified(self, p):
        p(("external", "available_externally", "linkonce", "linkonce_odr", "weak", "weak_odr",
           "appending", "internal", "private", "extern_weak", "common",)[self._value_])


class linkage_LinkageAttr(IntegerAttr):
    def __init__(self, value: linkage_Linkage):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


Linkage = linkage_Linkage


class ModFlagBehavior(enum.Enum):
    Error = 1
    Warning = 2
    Require = 3
    Override = 4
    Append = 5
    AppendUnique = 6
    Max = 7
    Min = 8

    def _print_mlir_unqualified(self, p):
        p(("", "error", "warning", "require", "override", "append", "append_unique", "max",
           "min",)[self._value_])


class ModFlagBehaviorAttr(IntegerAttr):
    def __init__(self, value: ModFlagBehavior):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class ModRefInfo(enum.Enum):
    NoModRef = 0
    Ref = 1
    Mod = 2
    ModRef = 3

    def _print_mlir_unqualified(self, p):
        p(("none", "read", "write", "readwrite",)[self._value_])


class ModRefInfoAttr(IntegerAttr):
    def __init__(self, value: ModRefInfo):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class RoundingMode(enum.Enum):
    TowardZero = 0
    NearestTiesToEven = 1
    TowardPositive = 2
    TowardNegative = 3
    NearestTiesToAway = 4
    Dynamic = 7
    Invalid = -1

    def _print_mlir_unqualified(self, p):
        p(("towardzero", "tonearest", "upward", "downward", "tonearestaway", "", "",
           "invalid",)[self._value_])


class RoundingModeAttr(IntegerAttr):
    def __init__(self, value: RoundingMode):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class tailcallkind_TailCallKind(enum.Enum):
    None_ = 0
    NoTail = 3
    MustTail = 2
    Tail = 1

    def _print_mlir_unqualified(self, p):
        p(("none", "tail", "musttail", "notail",)[self._value_])


class tailcallkind_TailCallKindAttr(IntegerAttr):
    def __init__(self, value: tailcallkind_TailCallKind):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


TailCallKind = tailcallkind_TailCallKind


class uwtable_UWTableKind(enum.Enum):
    None_ = 0
    Sync = 1
    Async = 2

    def _print_mlir_unqualified(self, p):
        p(("none", "sync", "async",)[self._value_])


class uwtable_UWTableKindAttr(IntegerAttr):
    def __init__(self, value: uwtable_UWTableKind):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class UnnamedAddr(enum.Enum):
    None_ = 0
    Local = 1
    Global = 2

    def _print_mlir_unqualified(self, p):
        p(("", "local_unnamed_addr", "unnamed_addr",)[self._value_])


class UnnamedAddrAttr(IntegerAttr):
    def __init__(self, value: UnnamedAddr):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class Visibility(enum.Enum):
    Default = 0
    Hidden = 1
    Protected = 2

    def _print_mlir_unqualified(self, p):
        p(("", "hidden", "protected",)[self._value_])


class VisibilityAttr(IntegerAttr):
    def __init__(self, value: Visibility):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


# ---- Attributes ----


class DINodeAttr(Attribute):
    pass


class DIScopeAttr(DINodeAttr):
    pass


class DILocalScopeAttr(DIScopeAttr):
    pass


class DITypeAttr(DINodeAttr):
    pass


class DIVariableAttr(DINodeAttr):
    pass


class TBAANodeAttr(Attribute):
    pass


@dataclass(kw_only=True)
class CConvAttr(Attribute, dialect='llvm', mnemonic='cconv'):
    CallingConv: "CConv"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.CallingConv._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class ComdatAttr(Attribute, dialect='llvm', mnemonic='comdat'):
    comdat: "comdat_Comdat"

    def _print_mlir_unqualified(self, p):
        self.comdat._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class FramePointerKindAttr(Attribute, dialect='llvm', mnemonic='framePointerKind'):
    framePointerKind: "framePointerKind_FramePointerKind"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.framePointerKind._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class AccessGroupAttr(Attribute, dialect='llvm', mnemonic='access_group'):
    id: "DistinctAttr"

    def _print_mlir_unqualified(self, p):
        p("<id = ")
        self.id._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class AddressSpaceAttr(Attribute, LLVMAddrSpaceAttrInterface, ptr.MemorySpaceAttrInterface,
                       dialect='llvm', mnemonic='address_space'):
    addressSpace: "int"

    def _print_mlir_unqualified(self, p):
        p("<")
        p(str(self.addressSpace))
        p(">")


@dataclass(kw_only=True)
class AliasScopeAttr(Attribute, dialect='llvm', mnemonic='alias_scope'):
    id: "Attribute"
    domain: "AliasScopeDomainAttr"
    description: Optional["StringAttr"] = None

    def _print_mlir_unqualified(self, p):
        p("<id = ")
        self.id._print_mlir_unqualified(p)
        p(", domain = ")
        self.domain._print_mlir_unqualified(p)
        if self.description is not None:
            p(", description = ")
            self.description._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class AliasScopeDomainAttr(Attribute, dialect='llvm', mnemonic='alias_scope_domain'):
    id: "Attribute"
    description: Optional["StringAttr"] = None

    def _print_mlir_unqualified(self, p):
        p("<id = ")
        self.id._print_mlir_unqualified(p)
        if self.description is not None:
            p(", description = ")
            self.description._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class BlockAddressAttr(Attribute, dialect='llvm', mnemonic='blockaddress'):
    function: "FlatSymbolRefAttr"
    tag: "BlockTagAttr"

    def _print_mlir_unqualified(self, p):
        p("<function = ")
        self.function._print_mlir_unqualified(p)
        p(", tag = ")
        self.tag._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class BlockTagAttr(Attribute, dialect='llvm', mnemonic='blocktag'):
    id: "int"

    def _print_mlir_unqualified(self, p):
        p("<id = ")
        p(str(self.id))
        p(">")


@dataclass(kw_only=True)
class ConstantRangeAttr(Attribute, dialect='llvm', mnemonic='constant_range'):
    lower: "APInt"
    upper: "APInt"


@dataclass(kw_only=True)
class DIAnnotationAttr(DINodeAttr, dialect='llvm', mnemonic='di_annotation'):
    name: "StringAttr"
    value: "StringAttr"

    def _print_mlir_unqualified(self, p):
        p("<name = ")
        self.name._print_mlir_unqualified(p)
        p(", value = ")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class DIBasicTypeAttr(DITypeAttr, dialect='llvm', mnemonic='di_basic_type'):
    tag: "int" = 0
    name: Optional["StringAttr"] = None
    sizeInBits: "int" = 0
    encoding: "int" = 0

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.tag != 0:
            p("tag = ")
            p(str(self.tag))
            comma = ", "
        if self.name is not None:
            p(comma)
            p("name = ")
            self.name._print_mlir_unqualified(p)
            comma = ", "
        if self.sizeInBits != 0:
            p(comma)
            p("sizeInBits = ")
            p(str(self.sizeInBits))
            comma = ", "
        if self.encoding != 0:
            p(comma)
            p("encoding = ")
            p(str(self.encoding))
        p(">")


@dataclass(kw_only=True)
class DICommonBlockAttr(DIScopeAttr, dialect='llvm', mnemonic='di_common_block'):
    scope: "DIScopeAttr"
    decl: Optional["DIGlobalVariableAttr"] = None
    name: "StringAttr"
    file: Optional["DIFileAttr"] = None
    line: "int" = 0

    def _print_mlir_unqualified(self, p):
        p("<scope = ")
        self.scope._print_mlir_unqualified(p)
        if self.decl is not None:
            p(", decl = ")
            self.decl._print_mlir_unqualified(p)
        p(", name = ")
        self.name._print_mlir_unqualified(p)
        if self.file is not None:
            p(", file = ")
            self.file._print_mlir_unqualified(p)
        if self.line != 0:
            p(", line = ")
            p(str(self.line))
        p(">")


@dataclass(kw_only=True)
class DICompileUnitAttr(DIScopeAttr, DIRecursiveTypeAttrInterface, dialect='llvm',
                        mnemonic='di_compile_unit'):
    recId: Optional["DistinctAttr"] = None
    isRecSelf: "bool" = False
    id: Optional["DistinctAttr"] = None
    sourceLanguage: "int" = 0
    file: Optional["DIFileAttr"] = None
    producer: Optional["StringAttr"] = None
    isOptimized: "bool" = False
    emissionKind: "DIEmissionKind" = dataclasses.field(default_factory=lambda: DIEmissionKind(0))
    isDebugInfoForProfiling: "bool" = False
    nameTableKind: "DINameTableKind" = dataclasses.field(default_factory=lambda: DINameTableKind(0))  # noqa: E501
    splitDebugFilename: Optional["StringAttr"] = None
    importedEntities: "Sequence[DINodeAttr]" = ()

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.recId is not None:
            p("recId = ")
            self.recId._print_mlir_unqualified(p)
            comma = ", "
        if self.isRecSelf:
            p(comma)
            p("isRecSelf = ")
            p("true" if self.isRecSelf else "false")
            comma = ", "
        if self.id is not None:
            p(comma)
            p("id = ")
            self.id._print_mlir_unqualified(p)
            comma = ", "
        if self.sourceLanguage != 0:
            p(comma)
            p("sourceLanguage = ")
            p(str(self.sourceLanguage))
            comma = ", "
        if self.file is not None:
            p(comma)
            p("file = ")
            self.file._print_mlir_unqualified(p)
            comma = ", "
        if self.producer is not None:
            p(comma)
            p("producer = ")
            self.producer._print_mlir_unqualified(p)
            comma = ", "
        if self.isOptimized:
            p(comma)
            p("isOptimized = ")
            p("true" if self.isOptimized else "false")
            comma = ", "
        if self.emissionKind != DIEmissionKind(0):
            p(comma)
            p("emissionKind = ")
            self.emissionKind._print_mlir_unqualified(p)
            comma = ", "
        if self.isDebugInfoForProfiling:
            p(comma)
            p("isDebugInfoForProfiling = ")
            p("true" if self.isDebugInfoForProfiling else "false")
            comma = ", "
        if self.nameTableKind != DINameTableKind(0):
            p(comma)
            p("nameTableKind = ")
            self.nameTableKind._print_mlir_unqualified(p)
            comma = ", "
        if self.splitDebugFilename is not None:
            p(comma)
            p("splitDebugFilename = ")
            self.splitDebugFilename._print_mlir_unqualified(p)
            comma = ", "
        if self.importedEntities != ():
            p(comma)
            p("importedEntities = ")
            p.print_array(self.importedEntities)
        p(">")


@dataclass(kw_only=True)
class DICompositeTypeAttr(DITypeAttr, DIRecursiveTypeAttrInterface, dialect='llvm',
                          mnemonic='di_composite_type'):
    recId: Optional["DistinctAttr"] = None
    isRecSelf: "bool" = False
    tag: "int" = 0
    name: Optional["StringAttr"] = None
    file: Optional["DIFileAttr"] = None
    line: "int" = 0
    scope: Optional["DIScopeAttr"] = None
    baseType: Optional["DITypeAttr"] = None
    flags: "DIFlags" = dataclasses.field(default_factory=lambda: DIFlags(0))
    sizeInBits: "int" = 0
    alignInBits: "int" = 0
    dataLocation: Optional["DIExpressionAttr"] = None
    rank: Optional["DIExpressionAttr"] = None
    allocated: Optional["DIExpressionAttr"] = None
    associated: Optional["DIExpressionAttr"] = None
    elements: "Sequence[DINodeAttr]" = ()

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.recId is not None:
            p("recId = ")
            self.recId._print_mlir_unqualified(p)
            comma = ", "
        if self.isRecSelf:
            p(comma)
            p("isRecSelf = ")
            p("true" if self.isRecSelf else "false")
            comma = ", "
        if self.tag != 0:
            p(comma)
            p("tag = ")
            p(str(self.tag))
            comma = ", "
        if self.name is not None:
            p(comma)
            p("name = ")
            self.name._print_mlir_unqualified(p)
            comma = ", "
        if self.file is not None:
            p(comma)
            p("file = ")
            self.file._print_mlir_unqualified(p)
            comma = ", "
        if self.line != 0:
            p(comma)
            p("line = ")
            p(str(self.line))
            comma = ", "
        if self.scope is not None:
            p(comma)
            p("scope = ")
            self.scope._print_mlir_unqualified(p)
            comma = ", "
        if self.baseType is not None:
            p(comma)
            p("baseType = ")
            self.baseType._print_mlir_unqualified(p)
            comma = ", "
        if self.flags != DIFlags(0):
            p(comma)
            p("flags = ")
            self.flags._print_mlir_unqualified(p)
            comma = ", "
        if self.sizeInBits != 0:
            p(comma)
            p("sizeInBits = ")
            p(str(self.sizeInBits))
            comma = ", "
        if self.alignInBits != 0:
            p(comma)
            p("alignInBits = ")
            p(str(self.alignInBits))
            comma = ", "
        if self.dataLocation is not None:
            p(comma)
            p("dataLocation = ")
            self.dataLocation._print_mlir_unqualified(p)
            comma = ", "
        if self.rank is not None:
            p(comma)
            p("rank = ")
            self.rank._print_mlir_unqualified(p)
            comma = ", "
        if self.allocated is not None:
            p(comma)
            p("allocated = ")
            self.allocated._print_mlir_unqualified(p)
            comma = ", "
        if self.associated is not None:
            p(comma)
            p("associated = ")
            self.associated._print_mlir_unqualified(p)
            comma = ", "
        if self.elements != ():
            p(comma)
            p("elements = ")
            p.print_array(self.elements)
        p(">")


@dataclass(kw_only=True)
class DIDerivedTypeAttr(DITypeAttr, dialect='llvm', mnemonic='di_derived_type'):
    tag: "int" = 0
    name: Optional["StringAttr"] = None
    file: Optional["DIFileAttr"] = None
    line: "int" = 0
    scope: Optional["DIScopeAttr"] = None
    baseType: Optional["DITypeAttr"] = None
    sizeInBits: "int" = 0
    alignInBits: "int" = 0
    offsetInBits: "int" = 0
    dwarfAddressSpace: "Optional[int]" = None
    flags: "DIFlags" = dataclasses.field(default_factory=lambda: DIFlags(0))
    extraData: Optional["DINodeAttr"] = None

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.tag != 0:
            p("tag = ")
            p(str(self.tag))
            comma = ", "
        if self.name is not None:
            p(comma)
            p("name = ")
            self.name._print_mlir_unqualified(p)
            comma = ", "
        if self.file is not None:
            p(comma)
            p("file = ")
            self.file._print_mlir_unqualified(p)
            comma = ", "
        if self.line != 0:
            p(comma)
            p("line = ")
            p(str(self.line))
            comma = ", "
        if self.scope is not None:
            p(comma)
            p("scope = ")
            self.scope._print_mlir_unqualified(p)
            comma = ", "
        if self.baseType is not None:
            p(comma)
            p("baseType = ")
            self.baseType._print_mlir_unqualified(p)
            comma = ", "
        if self.sizeInBits != 0:
            p(comma)
            p("sizeInBits = ")
            p(str(self.sizeInBits))
            comma = ", "
        if self.alignInBits != 0:
            p(comma)
            p("alignInBits = ")
            p(str(self.alignInBits))
            comma = ", "
        if self.offsetInBits != 0:
            p(comma)
            p("offsetInBits = ")
            p(str(self.offsetInBits))
            comma = ", "
        if self.dwarfAddressSpace is not None:
            p(comma)
            p("dwarfAddressSpace = ")
            p.if_present(self.dwarfAddressSpace, lambda: p(str(self.dwarfAddressSpace)))
            comma = ", "
        if self.flags != DIFlags(0):
            p(comma)
            p("flags = ")
            self.flags._print_mlir_unqualified(p)
            comma = ", "
        if self.extraData is not None:
            p(comma)
            p("extraData = ")
            self.extraData._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class DIExpressionAttr(Attribute, dialect='llvm', mnemonic='di_expression'):
    operations: "Sequence[DIExpressionElemAttr]" = ()

    def _print_mlir_unqualified(self, p):
        p("<")
        if self.operations != ():
            p("[")
            if self.operations != ():
                p.print_array(self.operations)
            p("]")
        else:
            pass
        p(">")


@dataclass(kw_only=True)
class DIExpressionElemAttr(Attribute, dialect='llvm', mnemonic='di_expression_elem'):
    opcode: "int"
    arguments: "Sequence[int]" = ()

    def _print_mlir_unqualified(self, p):
        p(str(self.opcode))
        if self.arguments != ():
            p("(")
            p.print_custom_ExpressionArg(self.opcode, self.arguments)
            p(")")
        else:
            pass


@dataclass(kw_only=True)
class DIFileAttr(DIScopeAttr, dialect='llvm', mnemonic='di_file'):
    name: "StringAttr"
    directory: "StringAttr"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.name._print_mlir_unqualified(p)
        p(" in ")
        self.directory._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class DIGenericSubrangeAttr(DINodeAttr, dialect='llvm', mnemonic='di_generic_subrange'):
    count: "Attribute" = None
    lowerBound: "Attribute"
    upperBound: "Attribute" = None
    stride: "Attribute"

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.count is not None:
            p("count = ")
            self.count._print_mlir_unqualified(p)
        p(comma)
        p("lowerBound = ")
        self.lowerBound._print_mlir_unqualified(p)
        if self.upperBound is not None:
            p(", upperBound = ")
            self.upperBound._print_mlir_unqualified(p)
        p(", stride = ")
        self.stride._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class DIGlobalVariableAttr(DINodeAttr, dialect='llvm', mnemonic='di_global_variable'):
    scope: Optional["DIScopeAttr"] = None
    name: Optional["StringAttr"] = None
    linkageName: Optional["StringAttr"] = None
    file: "DIFileAttr"
    line: "int"
    type: "DITypeAttr"
    isLocalToUnit: "bool" = False
    isDefined: "bool" = False
    alignInBits: "int" = 0

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.scope is not None:
            p("scope = ")
            self.scope._print_mlir_unqualified(p)
            comma = ", "
        if self.name is not None:
            p(comma)
            p("name = ")
            self.name._print_mlir_unqualified(p)
            comma = ", "
        if self.linkageName is not None:
            p(comma)
            p("linkageName = ")
            self.linkageName._print_mlir_unqualified(p)
        p(comma)
        p("file = ")
        self.file._print_mlir_unqualified(p)
        p(", line = ")
        p(str(self.line))
        p(", type = ")
        self.type._print_mlir_unqualified(p)
        if self.isLocalToUnit:
            p(", isLocalToUnit = ")
            p("true" if self.isLocalToUnit else "false")
        if self.isDefined:
            p(", isDefined = ")
            p("true" if self.isDefined else "false")
        if self.alignInBits != 0:
            p(", alignInBits = ")
            p(str(self.alignInBits))
        p(">")


@dataclass(kw_only=True)
class DIGlobalVariableExpressionAttr(Attribute, dialect='llvm',
                                     mnemonic='di_global_variable_expression'):
    var: "DIGlobalVariableAttr"
    expr: Optional["DIExpressionAttr"] = None

    def _print_mlir_unqualified(self, p):
        p("<var = ")
        self.var._print_mlir_unqualified(p)
        if self.expr is not None:
            p(", expr = ")
            self.expr._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class DIImportedEntityAttr(DINodeAttr, dialect='llvm', mnemonic='di_imported_entity'):
    tag: "int" = 0
    scope: "DIScopeAttr"
    entity: "DINodeAttr"
    file: Optional["DIFileAttr"] = None
    line: "int" = 0
    name: Optional["StringAttr"] = None
    elements: "Sequence[DINodeAttr]" = ()

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.tag != 0:
            p("tag = ")
            p(str(self.tag))
        p(comma)
        p("scope = ")
        self.scope._print_mlir_unqualified(p)
        p(", entity = ")
        self.entity._print_mlir_unqualified(p)
        if self.file is not None:
            p(", file = ")
            self.file._print_mlir_unqualified(p)
        if self.line != 0:
            p(", line = ")
            p(str(self.line))
        if self.name is not None:
            p(", name = ")
            self.name._print_mlir_unqualified(p)
        if self.elements != ():
            p(", elements = ")
            p.print_array(self.elements)
        p(">")


@dataclass(kw_only=True)
class DILabelAttr(DINodeAttr, dialect='llvm', mnemonic='di_label'):
    scope: "DIScopeAttr"
    name: Optional["StringAttr"] = None
    file: Optional["DIFileAttr"] = None
    line: "int" = 0

    def _print_mlir_unqualified(self, p):
        p("<scope = ")
        self.scope._print_mlir_unqualified(p)
        if self.name is not None:
            p(", name = ")
            self.name._print_mlir_unqualified(p)
        if self.file is not None:
            p(", file = ")
            self.file._print_mlir_unqualified(p)
        if self.line != 0:
            p(", line = ")
            p(str(self.line))
        p(">")


@dataclass(kw_only=True)
class DILexicalBlockAttr(DILocalScopeAttr, dialect='llvm', mnemonic='di_lexical_block'):
    scope: "DIScopeAttr"
    file: Optional["DIFileAttr"] = None
    line: "int" = 0
    column: "int" = 0

    def _print_mlir_unqualified(self, p):
        p("<scope = ")
        self.scope._print_mlir_unqualified(p)
        if self.file is not None:
            p(", file = ")
            self.file._print_mlir_unqualified(p)
        if self.line != 0:
            p(", line = ")
            p(str(self.line))
        if self.column != 0:
            p(", column = ")
            p(str(self.column))
        p(">")


@dataclass(kw_only=True)
class DILexicalBlockFileAttr(DILocalScopeAttr, dialect='llvm', mnemonic='di_lexical_block_file'):
    scope: "DIScopeAttr"
    file: Optional["DIFileAttr"] = None
    discriminator: "int"

    def _print_mlir_unqualified(self, p):
        p("<scope = ")
        self.scope._print_mlir_unqualified(p)
        if self.file is not None:
            p(", file = ")
            self.file._print_mlir_unqualified(p)
        p(", discriminator = ")
        p(str(self.discriminator))
        p(">")


@dataclass(kw_only=True)
class DILocalVariableAttr(DINodeAttr, dialect='llvm', mnemonic='di_local_variable'):
    scope: "DIScopeAttr"
    name: Optional["StringAttr"] = None
    file: Optional["DIFileAttr"] = None
    line: "int" = 0
    arg: "int" = 0
    alignInBits: "int" = 0
    type: Optional["DITypeAttr"] = None
    flags: "DIFlags" = dataclasses.field(default_factory=lambda: DIFlags(0))

    def _print_mlir_unqualified(self, p):
        p("<scope = ")
        self.scope._print_mlir_unqualified(p)
        if self.name is not None:
            p(", name = ")
            self.name._print_mlir_unqualified(p)
        if self.file is not None:
            p(", file = ")
            self.file._print_mlir_unqualified(p)
        if self.line != 0:
            p(", line = ")
            p(str(self.line))
        if self.arg != 0:
            p(", arg = ")
            p(str(self.arg))
        if self.alignInBits != 0:
            p(", alignInBits = ")
            p(str(self.alignInBits))
        if self.type is not None:
            p(", type = ")
            self.type._print_mlir_unqualified(p)
        if self.flags != DIFlags(0):
            p(", flags = ")
            self.flags._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class DIModuleAttr(DIScopeAttr, dialect='llvm', mnemonic='di_module'):
    file: Optional["DIFileAttr"] = None
    scope: Optional["DIScopeAttr"] = None
    name: Optional["StringAttr"] = None
    configMacros: Optional["StringAttr"] = None
    includePath: Optional["StringAttr"] = None
    apinotes: Optional["StringAttr"] = None
    line: "int" = 0
    isDecl: "bool" = False

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.file is not None:
            p("file = ")
            self.file._print_mlir_unqualified(p)
            comma = ", "
        if self.scope is not None:
            p(comma)
            p("scope = ")
            self.scope._print_mlir_unqualified(p)
            comma = ", "
        if self.name is not None:
            p(comma)
            p("name = ")
            self.name._print_mlir_unqualified(p)
            comma = ", "
        if self.configMacros is not None:
            p(comma)
            p("configMacros = ")
            self.configMacros._print_mlir_unqualified(p)
            comma = ", "
        if self.includePath is not None:
            p(comma)
            p("includePath = ")
            self.includePath._print_mlir_unqualified(p)
            comma = ", "
        if self.apinotes is not None:
            p(comma)
            p("apinotes = ")
            self.apinotes._print_mlir_unqualified(p)
            comma = ", "
        if self.line != 0:
            p(comma)
            p("line = ")
            p(str(self.line))
            comma = ", "
        if self.isDecl:
            p(comma)
            p("isDecl = ")
            p("true" if self.isDecl else "false")
        p(">")


@dataclass(kw_only=True)
class DINamespaceAttr(DIScopeAttr, dialect='llvm', mnemonic='di_namespace'):
    name: Optional["StringAttr"] = None
    scope: Optional["DIScopeAttr"] = None
    exportSymbols: "bool"

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.name is not None:
            p("name = ")
            self.name._print_mlir_unqualified(p)
            comma = ", "
        if self.scope is not None:
            p(comma)
            p("scope = ")
            self.scope._print_mlir_unqualified(p)
        p(comma)
        p("exportSymbols = ")
        p("true" if self.exportSymbols else "false")
        p(">")


@dataclass(kw_only=True)
class DINullTypeAttr(DITypeAttr, dialect='llvm', mnemonic='di_null_type'):

    def _print_mlir_unqualified(self, p):
        pass


@dataclass(kw_only=True)
class DIStringTypeAttr(DITypeAttr, dialect='llvm', mnemonic='di_string_type'):
    tag: "int" = 0
    name: Optional["StringAttr"] = None
    sizeInBits: "int" = 0
    alignInBits: "int" = 0
    stringLength: Optional["DIVariableAttr"] = None
    stringLengthExp: Optional["DIExpressionAttr"] = None
    stringLocationExp: Optional["DIExpressionAttr"] = None
    encoding: "int" = 0

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.tag != 0:
            p("tag = ")
            p(str(self.tag))
            comma = ", "
        if self.name is not None:
            p(comma)
            p("name = ")
            self.name._print_mlir_unqualified(p)
            comma = ", "
        if self.sizeInBits != 0:
            p(comma)
            p("sizeInBits = ")
            p(str(self.sizeInBits))
            comma = ", "
        if self.alignInBits != 0:
            p(comma)
            p("alignInBits = ")
            p(str(self.alignInBits))
            comma = ", "
        if self.stringLength is not None:
            p(comma)
            p("stringLength = ")
            self.stringLength._print_mlir_unqualified(p)
            comma = ", "
        if self.stringLengthExp is not None:
            p(comma)
            p("stringLengthExp = ")
            self.stringLengthExp._print_mlir_unqualified(p)
            comma = ", "
        if self.stringLocationExp is not None:
            p(comma)
            p("stringLocationExp = ")
            self.stringLocationExp._print_mlir_unqualified(p)
            comma = ", "
        if self.encoding != 0:
            p(comma)
            p("encoding = ")
            p(str(self.encoding))
        p(">")


@dataclass(kw_only=True)
class DISubprogramAttr(DILocalScopeAttr, DIRecursiveTypeAttrInterface, dialect='llvm',
                       mnemonic='di_subprogram'):
    recId: Optional["DistinctAttr"] = None
    isRecSelf: "bool" = False
    id: Optional["DistinctAttr"] = None
    compileUnit: Optional["DICompileUnitAttr"] = None
    scope: Optional["DIScopeAttr"] = None
    name: Optional["StringAttr"] = None
    linkageName: Optional["StringAttr"] = None
    file: Optional["DIFileAttr"] = None
    line: "int" = 0
    scopeLine: "int" = 0
    subprogramFlags: "DISubprogramFlags" = dataclasses.field(default_factory=lambda: DISubprogramFlags(0))  # noqa: E501
    type: Optional["DISubroutineTypeAttr"] = None
    retainedNodes: "Sequence[DINodeAttr]" = ()
    annotations: "Sequence[DINodeAttr]" = ()

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.recId is not None:
            p("recId = ")
            self.recId._print_mlir_unqualified(p)
            comma = ", "
        if self.isRecSelf:
            p(comma)
            p("isRecSelf = ")
            p("true" if self.isRecSelf else "false")
            comma = ", "
        if self.id is not None:
            p(comma)
            p("id = ")
            self.id._print_mlir_unqualified(p)
            comma = ", "
        if self.compileUnit is not None:
            p(comma)
            p("compileUnit = ")
            self.compileUnit._print_mlir_unqualified(p)
            comma = ", "
        if self.scope is not None:
            p(comma)
            p("scope = ")
            self.scope._print_mlir_unqualified(p)
            comma = ", "
        if self.name is not None:
            p(comma)
            p("name = ")
            self.name._print_mlir_unqualified(p)
            comma = ", "
        if self.linkageName is not None:
            p(comma)
            p("linkageName = ")
            self.linkageName._print_mlir_unqualified(p)
            comma = ", "
        if self.file is not None:
            p(comma)
            p("file = ")
            self.file._print_mlir_unqualified(p)
            comma = ", "
        if self.line != 0:
            p(comma)
            p("line = ")
            p(str(self.line))
            comma = ", "
        if self.scopeLine != 0:
            p(comma)
            p("scopeLine = ")
            p(str(self.scopeLine))
            comma = ", "
        if self.subprogramFlags != DISubprogramFlags(0):
            p(comma)
            p("subprogramFlags = ")
            self.subprogramFlags._print_mlir_unqualified(p)
            comma = ", "
        if self.type is not None:
            p(comma)
            p("type = ")
            self.type._print_mlir_unqualified(p)
            comma = ", "
        if self.retainedNodes != ():
            p(comma)
            p("retainedNodes = ")
            p.print_array(self.retainedNodes)
            comma = ", "
        if self.annotations != ():
            p(comma)
            p("annotations = ")
            p.print_array(self.annotations)
        p(">")


@dataclass(kw_only=True)
class DISubrangeAttr(DINodeAttr, dialect='llvm', mnemonic='di_subrange'):
    count: "Attribute" = None
    lowerBound: "Attribute" = None
    upperBound: "Attribute" = None
    stride: "Attribute" = None

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.count is not None:
            p("count = ")
            self.count._print_mlir_unqualified(p)
            comma = ", "
        if self.lowerBound is not None:
            p(comma)
            p("lowerBound = ")
            self.lowerBound._print_mlir_unqualified(p)
            comma = ", "
        if self.upperBound is not None:
            p(comma)
            p("upperBound = ")
            self.upperBound._print_mlir_unqualified(p)
            comma = ", "
        if self.stride is not None:
            p(comma)
            p("stride = ")
            self.stride._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class DISubroutineTypeAttr(DITypeAttr, dialect='llvm', mnemonic='di_subroutine_type'):
    callingConvention: "int" = 0
    types: "Sequence[DITypeAttr]" = ()

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.callingConvention != 0:
            p("callingConvention = ")
            p(str(self.callingConvention))
            comma = ", "
        if self.types != ():
            p(comma)
            p("types = ")
            p.print_array(self.types)
        p(">")


@dataclass(kw_only=True)
class DSOLocalEquivalentAttr(Attribute, dialect='llvm', mnemonic='dso_local_equivalent'):
    sym: "FlatSymbolRefAttr"

    def _print_mlir_unqualified(self, p):
        self.sym._print_mlir_unqualified(p)


@dataclass(kw_only=True)
class DenormalFPEnvAttr(Attribute, dialect='llvm', mnemonic='denormal_fpenv'):
    default_output_mode: "DenormalModeKind"
    default_input_mode: "DenormalModeKind"
    float_output_mode: "DenormalModeKind"
    float_input_mode: "DenormalModeKind"

    def _print_mlir_unqualified(self, p):
        p("<default_output_mode = ")
        self.default_output_mode._print_mlir_unqualified(p)
        p(", default_input_mode = ")
        self.default_input_mode._print_mlir_unqualified(p)
        p(", float_output_mode = ")
        self.float_output_mode._print_mlir_unqualified(p)
        p(", float_input_mode = ")
        self.float_input_mode._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class DependentLibrariesAttr(Attribute, dialect='llvm', mnemonic='dependent_libraries'):
    libs: "Sequence[StringAttr]" = ()

    def _print_mlir_unqualified(self, p):
        p("<")
        if self.libs != ():
            p.print_array(self.libs)
        p(">")


@dataclass(kw_only=True)
class DereferenceableAttr(Attribute, dialect='llvm', mnemonic='dereferenceable'):
    bytes: "int"
    mayBeNull: "bool" = False

    def _print_mlir_unqualified(self, p):
        p("<bytes = ")
        p(str(self.bytes))
        if self.mayBeNull:
            p(", mayBeNull = ")
            p("true" if self.mayBeNull else "false")
        p(">")


@dataclass(kw_only=True)
class MDConstantAttr(Attribute, dialect='llvm', mnemonic='md_const'):
    value: "Attribute"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class MDFuncAttr(Attribute, dialect='llvm', mnemonic='md_func'):
    name: "FlatSymbolRefAttr"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.name._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class MDNodeAttr(Attribute, dialect='llvm', mnemonic='md_node'):
    operands: "Sequence[Attribute]" = ()

    def _print_mlir_unqualified(self, p):
        p("<")
        if self.operands != ():
            if self.operands != ():
                p.print_array(self.operands)
            p(">")
        else:
            p(">")


@dataclass(kw_only=True)
class MDStringAttr(Attribute, dialect='llvm', mnemonic='md_string'):
    value: "StringAttr"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class MMRATagAttr(Attribute, dialect='llvm', mnemonic='mmra_tag'):
    prefix: "str"
    suffix: "str"

    def _print_mlir_unqualified(self, p):
        p("<")
        p.print_escaped_string(self.prefix)
        p(" : ")
        p.print_escaped_string(self.suffix)
        p(">")


@dataclass(kw_only=True)
class MemoryEffectsAttr(Attribute, dialect='llvm', mnemonic='memory_effects'):
    other: "ModRefInfo"
    argMem: "ModRefInfo"
    inaccessibleMem: "ModRefInfo"
    errnoMem: "ModRefInfo"
    targetMem0: "ModRefInfo"
    targetMem1: "ModRefInfo"

    def _print_mlir_unqualified(self, p):
        p("<other = ")
        self.other._print_mlir_unqualified(p)
        p(", argMem = ")
        self.argMem._print_mlir_unqualified(p)
        p(", inaccessibleMem = ")
        self.inaccessibleMem._print_mlir_unqualified(p)
        p(", errnoMem = ")
        self.errnoMem._print_mlir_unqualified(p)
        p(", targetMem0 = ")
        self.targetMem0._print_mlir_unqualified(p)
        p(", targetMem1 = ")
        self.targetMem1._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class PoisonAttr(Attribute, dialect='llvm', mnemonic='poison'):

    def _print_mlir_unqualified(self, p):
        pass


@dataclass(kw_only=True)
class TBAAMemberAttr(Attribute, dialect='llvm', mnemonic='tbaa_member'):
    typeDesc: "TBAANodeAttr"
    offset: "int"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.typeDesc._print_mlir_unqualified(p)
        p(", ")
        p(str(self.offset))
        p(">")


@dataclass(kw_only=True)
class TBAARootAttr(TBAANodeAttr, dialect='llvm', mnemonic='tbaa_root'):
    id: Optional["StringAttr"] = None

    def _print_mlir_unqualified(self, p):
        if self.id is not None:
            p("<")
            if self.id is not None:
                p("id = ")
                self.id._print_mlir_unqualified(p)
            p(">")
        else:
            pass


@dataclass(kw_only=True)
class TBAATagAttr(Attribute, dialect='llvm', mnemonic='tbaa_tag'):
    base_type: "TBAATypeDescriptorAttr"
    access_type: "TBAATypeDescriptorAttr"
    offset: "int"
    constant: "bool" = False

    def _print_mlir_unqualified(self, p):
        p("<base_type = ")
        self.base_type._print_mlir_unqualified(p)
        p(", access_type = ")
        self.access_type._print_mlir_unqualified(p)
        p(", offset = ")
        p(str(self.offset))
        if self.constant:
            p(", constant = ")
            p("true" if self.constant else "false")
        p(">")


@dataclass(kw_only=True)
class TBAATypeDescriptorAttr(TBAANodeAttr, dialect='llvm', mnemonic='tbaa_type_desc'):
    id: "str"
    members: "Sequence[TBAAMemberAttr]"

    def _print_mlir_unqualified(self, p):
        p("<id = ")
        p.print_escaped_string(self.id)
        p(", members = ")
        p.print_array(self.members)
        p(">")


@dataclass(kw_only=True)
class TargetAttr(Attribute, DLTIQueryInterface, TargetAttrInterface, dialect='llvm',
                 mnemonic='target'):
    triple: "StringAttr"
    chip: "StringAttr"
    features: Optional["TargetFeaturesAttr"] = None

    def _print_mlir_unqualified(self, p):
        p("<triple = ")
        self.triple._print_mlir_unqualified(p)
        p(", chip = ")
        self.chip._print_mlir_unqualified(p)
        if self.features is not None:
            p(", features = ")
            self.features._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class TargetFeaturesAttr(Attribute, DLTIQueryInterface, dialect='llvm',
                         mnemonic='target_features'):
    features: "Sequence[StringAttr]" = ()

    def _print_mlir_unqualified(self, p):
        p("<[")
        if self.features != ():
            if self.features != ():
                p.print_array(self.features)
            p("]")
        else:
            p("]")
        p(">")


@dataclass(kw_only=True)
class UndefAttr(Attribute, dialect='llvm', mnemonic='undef'):

    def _print_mlir_unqualified(self, p):
        pass


@dataclass(kw_only=True)
class VScaleRangeAttr(Attribute, dialect='llvm', mnemonic='vscale_range'):
    minRange: "IntegerAttr"
    maxRange: "IntegerAttr"

    def _print_mlir_unqualified(self, p):
        p("<minRange = ")
        self.minRange._print_mlir_unqualified(p)
        p(", maxRange = ")
        self.maxRange._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class VecTypeHintAttr(Attribute, dialect='llvm', mnemonic='vec_type_hint'):
    hint: "TypeAttr"
    is_signed: "bool" = False

    def _print_mlir_unqualified(self, p):
        p("<hint = ")
        self.hint._print_mlir_unqualified(p)
        if self.is_signed:
            p(", is_signed = ")
            p("true" if self.is_signed else "false")
        p(">")


@dataclass(kw_only=True)
class ZeroAttr(Attribute, dialect='llvm', mnemonic='zero'):

    def _print_mlir_unqualified(self, p):
        pass


@dataclass(kw_only=True)
class LinkageAttr(Attribute, dialect='llvm', mnemonic='linkage'):
    linkage: "linkage_Linkage"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.linkage._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class LoopAnnotationAttr(Attribute, dialect='llvm', mnemonic='loop_annotation'):
    disableNonforced: Optional["BoolAttr"] = None
    vectorize: Optional["LoopVectorizeAttr"] = None
    interleave: Optional["LoopInterleaveAttr"] = None
    unroll: Optional["LoopUnrollAttr"] = None
    unrollAndJam: Optional["LoopUnrollAndJamAttr"] = None
    licm: Optional["LoopLICMAttr"] = None
    distribute: Optional["LoopDistributeAttr"] = None
    pipeline: Optional["LoopPipelineAttr"] = None
    peeled: Optional["LoopPeeledAttr"] = None
    unswitch: Optional["LoopUnswitchAttr"] = None
    mustProgress: Optional["BoolAttr"] = None
    isVectorized: Optional["BoolAttr"] = None
    startLoc: Optional["FusedLoc"] = None
    endLoc: Optional["FusedLoc"] = None
    parallelAccesses: "Sequence[AccessGroupAttr]" = ()

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.disableNonforced is not None:
            p("disableNonforced = ")
            self.disableNonforced._print_mlir_unqualified(p)
            comma = ", "
        if self.vectorize is not None:
            p(comma)
            p("vectorize = ")
            self.vectorize._print_mlir_unqualified(p)
            comma = ", "
        if self.interleave is not None:
            p(comma)
            p("interleave = ")
            self.interleave._print_mlir_unqualified(p)
            comma = ", "
        if self.unroll is not None:
            p(comma)
            p("unroll = ")
            self.unroll._print_mlir_unqualified(p)
            comma = ", "
        if self.unrollAndJam is not None:
            p(comma)
            p("unrollAndJam = ")
            self.unrollAndJam._print_mlir_unqualified(p)
            comma = ", "
        if self.licm is not None:
            p(comma)
            p("licm = ")
            self.licm._print_mlir_unqualified(p)
            comma = ", "
        if self.distribute is not None:
            p(comma)
            p("distribute = ")
            self.distribute._print_mlir_unqualified(p)
            comma = ", "
        if self.pipeline is not None:
            p(comma)
            p("pipeline = ")
            self.pipeline._print_mlir_unqualified(p)
            comma = ", "
        if self.peeled is not None:
            p(comma)
            p("peeled = ")
            self.peeled._print_mlir_unqualified(p)
            comma = ", "
        if self.unswitch is not None:
            p(comma)
            p("unswitch = ")
            self.unswitch._print_mlir_unqualified(p)
            comma = ", "
        if self.mustProgress is not None:
            p(comma)
            p("mustProgress = ")
            self.mustProgress._print_mlir_unqualified(p)
            comma = ", "
        if self.isVectorized is not None:
            p(comma)
            p("isVectorized = ")
            self.isVectorized._print_mlir_unqualified(p)
            comma = ", "
        if self.startLoc is not None:
            p(comma)
            p("startLoc = ")
            self.startLoc._print_mlir_unqualified(p)
            comma = ", "
        if self.endLoc is not None:
            p(comma)
            p("endLoc = ")
            self.endLoc._print_mlir_unqualified(p)
            comma = ", "
        if self.parallelAccesses != ():
            p(comma)
            p("parallelAccesses = ")
            p.print_array(self.parallelAccesses)
        p(">")


@dataclass(kw_only=True)
class LoopDistributeAttr(Attribute, dialect='llvm', mnemonic='loop_distribute'):
    disable: Optional["BoolAttr"] = None
    followupCoincident: Optional["LoopAnnotationAttr"] = None
    followupSequential: Optional["LoopAnnotationAttr"] = None
    followupFallback: Optional["LoopAnnotationAttr"] = None
    followupAll: Optional["LoopAnnotationAttr"] = None

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.disable is not None:
            p("disable = ")
            self.disable._print_mlir_unqualified(p)
            comma = ", "
        if self.followupCoincident is not None:
            p(comma)
            p("followupCoincident = ")
            self.followupCoincident._print_mlir_unqualified(p)
            comma = ", "
        if self.followupSequential is not None:
            p(comma)
            p("followupSequential = ")
            self.followupSequential._print_mlir_unqualified(p)
            comma = ", "
        if self.followupFallback is not None:
            p(comma)
            p("followupFallback = ")
            self.followupFallback._print_mlir_unqualified(p)
            comma = ", "
        if self.followupAll is not None:
            p(comma)
            p("followupAll = ")
            self.followupAll._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class LoopInterleaveAttr(Attribute, dialect='llvm', mnemonic='loop_interleave'):
    count: "IntegerAttr"

    def _print_mlir_unqualified(self, p):
        p("<count = ")
        self.count._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class LoopLICMAttr(Attribute, dialect='llvm', mnemonic='loop_licm'):
    disable: Optional["BoolAttr"] = None
    versioningDisable: Optional["BoolAttr"] = None

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.disable is not None:
            p("disable = ")
            self.disable._print_mlir_unqualified(p)
            comma = ", "
        if self.versioningDisable is not None:
            p(comma)
            p("versioningDisable = ")
            self.versioningDisable._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class LoopPeeledAttr(Attribute, dialect='llvm', mnemonic='loop_peeled'):
    count: Optional["IntegerAttr"] = None

    def _print_mlir_unqualified(self, p):
        p("<")
        if self.count is not None:
            p("count = ")
            self.count._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class LoopPipelineAttr(Attribute, dialect='llvm', mnemonic='loop_pipeline'):
    disable: Optional["BoolAttr"] = None
    initiationinterval: Optional["IntegerAttr"] = None

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.disable is not None:
            p("disable = ")
            self.disable._print_mlir_unqualified(p)
            comma = ", "
        if self.initiationinterval is not None:
            p(comma)
            p("initiationinterval = ")
            self.initiationinterval._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class LoopUnrollAndJamAttr(Attribute, dialect='llvm', mnemonic='loop_unroll_and_jam'):
    disable: Optional["BoolAttr"] = None
    count: Optional["IntegerAttr"] = None
    followupOuter: Optional["LoopAnnotationAttr"] = None
    followupInner: Optional["LoopAnnotationAttr"] = None
    followupRemainderOuter: Optional["LoopAnnotationAttr"] = None
    followupRemainderInner: Optional["LoopAnnotationAttr"] = None
    followupAll: Optional["LoopAnnotationAttr"] = None

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.disable is not None:
            p("disable = ")
            self.disable._print_mlir_unqualified(p)
            comma = ", "
        if self.count is not None:
            p(comma)
            p("count = ")
            self.count._print_mlir_unqualified(p)
            comma = ", "
        if self.followupOuter is not None:
            p(comma)
            p("followupOuter = ")
            self.followupOuter._print_mlir_unqualified(p)
            comma = ", "
        if self.followupInner is not None:
            p(comma)
            p("followupInner = ")
            self.followupInner._print_mlir_unqualified(p)
            comma = ", "
        if self.followupRemainderOuter is not None:
            p(comma)
            p("followupRemainderOuter = ")
            self.followupRemainderOuter._print_mlir_unqualified(p)
            comma = ", "
        if self.followupRemainderInner is not None:
            p(comma)
            p("followupRemainderInner = ")
            self.followupRemainderInner._print_mlir_unqualified(p)
            comma = ", "
        if self.followupAll is not None:
            p(comma)
            p("followupAll = ")
            self.followupAll._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class LoopUnrollAttr(Attribute, dialect='llvm', mnemonic='loop_unroll'):
    disable: Optional["BoolAttr"] = None
    count: Optional["IntegerAttr"] = None
    runtimeDisable: Optional["BoolAttr"] = None
    full: Optional["BoolAttr"] = None
    followupUnrolled: Optional["LoopAnnotationAttr"] = None
    followupRemainder: Optional["LoopAnnotationAttr"] = None
    followupAll: Optional["LoopAnnotationAttr"] = None

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.disable is not None:
            p("disable = ")
            self.disable._print_mlir_unqualified(p)
            comma = ", "
        if self.count is not None:
            p(comma)
            p("count = ")
            self.count._print_mlir_unqualified(p)
            comma = ", "
        if self.runtimeDisable is not None:
            p(comma)
            p("runtimeDisable = ")
            self.runtimeDisable._print_mlir_unqualified(p)
            comma = ", "
        if self.full is not None:
            p(comma)
            p("full = ")
            self.full._print_mlir_unqualified(p)
            comma = ", "
        if self.followupUnrolled is not None:
            p(comma)
            p("followupUnrolled = ")
            self.followupUnrolled._print_mlir_unqualified(p)
            comma = ", "
        if self.followupRemainder is not None:
            p(comma)
            p("followupRemainder = ")
            self.followupRemainder._print_mlir_unqualified(p)
            comma = ", "
        if self.followupAll is not None:
            p(comma)
            p("followupAll = ")
            self.followupAll._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class LoopUnswitchAttr(Attribute, dialect='llvm', mnemonic='loop_unswitch'):
    partialDisable: Optional["BoolAttr"] = None

    def _print_mlir_unqualified(self, p):
        p("<")
        if self.partialDisable is not None:
            p("partialDisable = ")
            self.partialDisable._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class LoopVectorizeAttr(Attribute, dialect='llvm', mnemonic='loop_vectorize'):
    disable: Optional["BoolAttr"] = None
    predicateEnable: Optional["BoolAttr"] = None
    scalableEnable: Optional["BoolAttr"] = None
    width: Optional["IntegerAttr"] = None
    followupVectorized: Optional["LoopAnnotationAttr"] = None
    followupEpilogue: Optional["LoopAnnotationAttr"] = None
    followupAll: Optional["LoopAnnotationAttr"] = None

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.disable is not None:
            p("disable = ")
            self.disable._print_mlir_unqualified(p)
            comma = ", "
        if self.predicateEnable is not None:
            p(comma)
            p("predicateEnable = ")
            self.predicateEnable._print_mlir_unqualified(p)
            comma = ", "
        if self.scalableEnable is not None:
            p(comma)
            p("scalableEnable = ")
            self.scalableEnable._print_mlir_unqualified(p)
            comma = ", "
        if self.width is not None:
            p(comma)
            p("width = ")
            self.width._print_mlir_unqualified(p)
            comma = ", "
        if self.followupVectorized is not None:
            p(comma)
            p("followupVectorized = ")
            self.followupVectorized._print_mlir_unqualified(p)
            comma = ", "
        if self.followupEpilogue is not None:
            p(comma)
            p("followupEpilogue = ")
            self.followupEpilogue._print_mlir_unqualified(p)
            comma = ", "
        if self.followupAll is not None:
            p(comma)
            p("followupAll = ")
            self.followupAll._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class ModuleFlagAttr(Attribute, dialect='llvm', mnemonic='mlir.module_flag'):
    behavior: "ModFlagBehavior"
    key: "StringAttr"
    value: "Attribute"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.behavior._print_mlir_unqualified(p)
        p(", ")
        self.key._print_mlir_unqualified(p)
        p(", ")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class ModuleFlagCGProfileEntryAttr(Attribute, dialect='llvm', mnemonic='cgprofile_entry'):
    from_: Optional["FlatSymbolRefAttr"] = None
    to: Optional["FlatSymbolRefAttr"] = None
    count: "int"

    def _print_mlir_unqualified(self, p):
        p("<")
        comma = ""
        if self.from_ is not None:
            p("from = ")
            self.from_._print_mlir_unqualified(p)
            comma = ", "
        if self.to is not None:
            p(comma)
            p("to = ")
            self.to._print_mlir_unqualified(p)
        p(comma)
        p("count = ")
        p(str(self.count))
        p(">")


@dataclass(kw_only=True)
class ModuleFlagProfileSummaryAttr(Attribute, dialect='llvm', mnemonic='profile_summary'):
    format: "ProfileSummaryFormatKind"
    total_count: "int"
    max_count: "int"
    max_internal_count: "int"
    max_function_count: "int"
    num_counts: "int"
    num_functions: "int"
    is_partial_profile: "Optional[int]" = None
    partial_profile_ratio: Optional["FloatAttr"] = None
    detailed_summary: "Sequence[ModuleFlagProfileSummaryDetailedAttr]"

    def _print_mlir_unqualified(self, p):
        p("<format = ")
        self.format._print_mlir_unqualified(p)
        p(", total_count = ")
        p(str(self.total_count))
        p(", max_count = ")
        p(str(self.max_count))
        p(", max_internal_count = ")
        p(str(self.max_internal_count))
        p(", max_function_count = ")
        p(str(self.max_function_count))
        p(", num_counts = ")
        p(str(self.num_counts))
        p(", num_functions = ")
        p(str(self.num_functions))
        if self.is_partial_profile is not None:
            p(", is_partial_profile = ")
            p.if_present(self.is_partial_profile, lambda: p(str(self.is_partial_profile)))
        if self.partial_profile_ratio is not None:
            p(", partial_profile_ratio = ")
            self.partial_profile_ratio._print_mlir_unqualified(p)
        p(", detailed_summary = ")
        p.print_array(self.detailed_summary)
        p(">")


@dataclass(kw_only=True)
class ModuleFlagProfileSummaryDetailedAttr(Attribute, dialect='llvm',
                                           mnemonic='profile_summary_detailed'):
    cut_off: "int"
    min_count: "int"
    num_counts: "int"

    def _print_mlir_unqualified(self, p):
        p("<cut_off = ")
        p(str(self.cut_off))
        p(", min_count = ")
        p(str(self.min_count))
        p(", num_counts = ")
        p(str(self.num_counts))
        p(">")


@dataclass(kw_only=True)
class TailCallKindAttr(Attribute, dialect='llvm', mnemonic='tailcallkind'):
    tailCallKind: "TailCallKind"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.tailCallKind._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class UWTableKindAttr(Attribute, dialect='llvm', mnemonic='uwtableKind'):
    uwtableKind: "uwtable_UWTableKind"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.uwtableKind._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class WorkgroupAttributionAttr(Attribute, dialect='llvm', mnemonic='mlir.workgroup_attribution'):
    num_elements: "IntegerAttr"
    element_type: "TypeAttr"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.num_elements._print_mlir_unqualified(p)
        p(", ")
        self.element_type._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class FastmathFlagsAttr(Attribute, dialect='llvm', mnemonic='fastmath'):
    value: "FastmathFlags"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class IntegerOverflowFlagsAttr(Attribute, dialect='llvm', mnemonic='overflow'):
    value: "IntegerOverflowFlags"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


# ---- Types ----


class LLVMVoidType(Type):
    def _print_mlir_unqualified(self, p):
        p("void")


class LLVMTokenType(Type):
    pass


class LLVMLabelType(Type):
    pass


class LLVMMetadataType(Type):
    pass


@dataclass(kw_only=True)
class LLVMStructType(Type, dialect='llvm', mnemonic='struct'):
    name: str | None = None
    types: Sequence[Type] | None = None  # None if opaque
    packed: bool = False

    @staticmethod
    def make_literal(types: Sequence[Type]):
        return LLVMStructType(types=types)

    def _print_mlir_unqualified(self, p):
        p("<")
        if self.name is not None:
            p.print_escaped_string(self.name)
            p(", ")
        if self.types is None:
            p("opaque")
        else:
            if self.packed:
                p("packed ")
            p("(")
            comma = ""
            for t in self.types:
                p(comma)
                t.print_mlir(p)
                comma = ", "
            p(")")
        p(">")


@dataclass(kw_only=True)
class LLVMArrayType(Type, DataLayoutTypeInterface, DestructurableTypeInterface, dialect='llvm',
                    mnemonic='array'):
    elementType: "Type"
    numElements: "int"

    def _print_mlir_unqualified(self, p):
        p("<")
        p(str(self.numElements))
        p(" x ")
        p.print_custom_PrettyLLVMType(self.elementType)
        p(">")


@dataclass(kw_only=True)
class LLVMFunctionType(Type, dialect='llvm', mnemonic='func'):
    returnType: "Type"
    params: "Sequence[Type]"
    varArg: "bool"

    def _print_mlir_unqualified(self, p):
        p("<")
        p.print_custom_PrettyLLVMType(self.returnType)
        p(" (")
        p.print_custom_FunctionTypes(self.params, self.varArg)
        p(">")


@dataclass(kw_only=True)
class LLVMPPCFP128Type(Type, DenseElementType, VectorElementTypeInterface, FloatType,
                       dialect='llvm', mnemonic='ppc_fp128'):

    def _print_mlir_unqualified(self, p):
        pass


@dataclass(kw_only=True)
class LLVMPointerType(Type, DataLayoutTypeInterface, VectorElementTypeInterface, dialect='llvm',
                      mnemonic='ptr'):
    addressSpace: "int" = 0

    def _print_mlir_unqualified(self, p):
        if self.addressSpace != 0:
            p("<")
            if self.addressSpace != 0:
                p(str(self.addressSpace))
            p(">")
        else:
            pass


@dataclass(kw_only=True)
class LLVMTargetExtType(Type, dialect='llvm', mnemonic='target'):
    extTypeName: "str"
    typeParams: "Sequence[Type]" = ()
    intParams: "Sequence[int]" = ()

    def _print_mlir_unqualified(self, p):
        p("<")
        p.print_escaped_string(self.extTypeName)
        if (self.typeParams != () or self.intParams != ()):
            p(", ")
            p.print_custom_ExtTypeParams(self.typeParams, self.intParams)
        else:
            pass
        p(">")


@dataclass(kw_only=True)
class LLVMX86AMXType(Type, dialect='llvm', mnemonic='x86_amx'):

    def _print_mlir_unqualified(self, p):
        pass


# ---- Operators ----


def add_AShrOp(
    *,
    lhs: Value,
    rhs: Value,
    isExact: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    if isExact:
        all_props.append(('isExact', UnitAttr()))
    return add_operation(
        name="llvm.ashr",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AddOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    return add_operation(
        name="llvm.add",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AddrSpaceCastOp(
    *,
    res_type: Type,
    arg: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.addrspacecast",
        result_type=res_type,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AddressOfOp(
    *,
    res_type: LLVMPointerType,
    global_name: str,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('global_name', FlatSymbolRefAttr(global_name)))
    return add_operation(
        name="llvm.mlir.addressof",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AliasOp(
    *,
    alias_type: Type,
    sym_name: str,
    linkage: Linkage,
    dso_local: bool = False,
    thread_local_: bool = False,
    unnamed_addr: Optional[UnnamedAddr] = None,
    visibility_: Visibility = Visibility(0),
    initializer: Region,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('alias_type', TypeAttr(value=alias_type)))
    all_props.append(('sym_name', StringAttr(value=sym_name)))
    all_props.append(('linkage', LinkageAttr(linkage=linkage)))
    if dso_local:
        all_props.append(('dso_local', UnitAttr()))
    if thread_local_:
        all_props.append(('thread_local_', UnitAttr()))
    if unnamed_addr is not None:
        all_props.append(('unnamed_addr', UnnamedAddrAttr(unnamed_addr)))
    all_props.append(('visibility_', VisibilityAttr(visibility_)))
    return add_operation(
        name="llvm.mlir.alias",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
        regions=[initializer],
    )


def add_AllocaOp(
    *,
    res_type: LLVMPointerType,
    arraySize: Value,
    alignment: Optional[int] = None,
    elem_type: Type,
    inalloca: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if alignment is not None:
        all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(64), alignment)))
    all_props.append(('elem_type', TypeAttr(value=elem_type)))
    if inalloca:
        all_props.append(('inalloca', UnitAttr()))
    return add_operation(
        name="llvm.alloca",
        result_type=res_type,
        operands=[arraySize],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AndOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    return add_operation(
        name="llvm.and",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AtomicCmpXchgOp(
    *,
    ptr: Value,
    cmp: Value,
    val: Value,
    success_ordering: AtomicOrdering,
    failure_ordering: AtomicOrdering,
    syncscope: Optional[str] = None,
    alignment: Optional[int] = None,
    weak: bool = False,
    volatile_: bool = False,
    access_groups: Optional[ArrayAttr] = None,
    alias_scopes: Optional[ArrayAttr] = None,
    noalias_scopes: Optional[ArrayAttr] = None,
    tbaa: Optional[ArrayAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = _util.get_val_and_bool_struct_type(val.type)
    all_props = []
    all_props.append(('success_ordering', AtomicOrderingAttr(success_ordering)))
    all_props.append(('failure_ordering', AtomicOrderingAttr(failure_ordering)))
    if syncscope is not None:
        all_props.append(('syncscope', StringAttr(value=syncscope)))
    if alignment is not None:
        all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(64), alignment)))
    if weak:
        all_props.append(('weak', UnitAttr()))
    if volatile_:
        all_props.append(('volatile_', UnitAttr()))
    if access_groups is not None:
        all_props.append(('access_groups', access_groups))
    if alias_scopes is not None:
        all_props.append(('alias_scopes', alias_scopes))
    if noalias_scopes is not None:
        all_props.append(('noalias_scopes', noalias_scopes))
    if tbaa is not None:
        all_props.append(('tbaa', tbaa))
    return add_operation(
        name="llvm.cmpxchg",
        result_type=res_type,
        operands=[ptr, cmp, val],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AtomicRMWOp(
    *,
    bin_op: AtomicBinOp,
    ptr: Value,
    val: Value,
    ordering: AtomicOrdering,
    syncscope: Optional[str] = None,
    alignment: Optional[int] = None,
    volatile_: bool = False,
    access_groups: Optional[ArrayAttr] = None,
    alias_scopes: Optional[ArrayAttr] = None,
    noalias_scopes: Optional[ArrayAttr] = None,
    tbaa: Optional[ArrayAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = val.type
    all_props = []
    all_props.append(('bin_op', AtomicBinOpAttr(bin_op)))
    all_props.append(('ordering', AtomicOrderingAttr(ordering)))
    if syncscope is not None:
        all_props.append(('syncscope', StringAttr(value=syncscope)))
    if alignment is not None:
        all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(64), alignment)))
    if volatile_:
        all_props.append(('volatile_', UnitAttr()))
    if access_groups is not None:
        all_props.append(('access_groups', access_groups))
    if alias_scopes is not None:
        all_props.append(('alias_scopes', alias_scopes))
    if noalias_scopes is not None:
        all_props.append(('noalias_scopes', noalias_scopes))
    if tbaa is not None:
        all_props.append(('tbaa', tbaa))
    return add_operation(
        name="llvm.atomicrmw",
        result_type=res_type,
        operands=[ptr, val],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BitcastOp(
    *,
    res_type: Type,
    arg: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.bitcast",
        result_type=res_type,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BlockAddressOp(
    *,
    res_type: LLVMPointerType,
    block_addr: BlockAddressAttr,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('block_addr', block_addr))
    return add_operation(
        name="llvm.blockaddress",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BlockTagOp(
    *,
    tag: BlockTagAttr,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('tag', tag))
    return add_operation(
        name="llvm.blocktag",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BrOp(
    *,
    destOperands: Sequence[Value],
    loop_annotation: Optional[LoopAnnotationAttr] = None,
    dest: BlockLabel,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if loop_annotation is not None:
        all_props.append(('loop_annotation', loop_annotation))
    return add_operation(
        name="llvm.br",
        result_type=None,
        operands=list(destOperands),
        properties=all_props,
        attributes=extra_attributes,
        successors=[dest],
    )


def add_CallIntrinsicOp(
    *,
    results_type: Optional[Type],
    intrin: str,
    args: Sequence[Value],
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    op_bundle_operands: Sequence[Value],
    op_bundle_sizes: Sequence[int],
    op_bundle_tags: Optional[ArrayAttr] = None,
    arg_attrs: Optional[ArrayAttr] = None,
    res_attrs: Optional[ArrayAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    all_props.append(('intrin', StringAttr(value=intrin)))
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    all_props.append(('op_bundle_sizes', DenseI32ArrayAttr(op_bundle_sizes)))
    if op_bundle_tags is not None:
        all_props.append(('op_bundle_tags', op_bundle_tags))
    if arg_attrs is not None:
        all_props.append(('arg_attrs', arg_attrs))
    if res_attrs is not None:
        all_props.append(('res_attrs', res_attrs))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([len(args), len(op_bundle_operands)])))
    return add_operation(
        name="llvm.call_intrinsic",
        result_type=results_type,
        operands=[*args, *op_bundle_operands],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CallOp(
    *,
    result_type: Optional[Type],
    var_callee_type: Optional[LLVMFunctionType] = None,
    callee: Optional[str] = None,
    callee_operands: Sequence[Value],
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    CConv: cconv_CConv = cconv_CConv(0),
    TailCallKind: tailcallkind_TailCallKind = tailcallkind_TailCallKind(0),
    memory_effects: Optional[MemoryEffectsAttr] = None,
    convergent: bool = False,
    no_unwind: bool = False,
    will_return: bool = False,
    noreturn: bool = False,
    returns_twice: bool = False,
    hot: bool = False,
    cold: bool = False,
    noduplicate: bool = False,
    no_caller_saved_registers: bool = False,
    nocallback: bool = False,
    modular_format: Optional[str] = None,
    nobuiltins: Optional[ArrayAttr] = None,
    allocsize: Optional[Sequence[int]] = None,
    optsize: bool = False,
    minsize: bool = False,
    builtin: bool = False,
    nobuiltin: bool = False,
    save_reg_params: bool = False,
    zero_call_used_regs: Optional[str] = None,
    trap_func_name: Optional[str] = None,
    default_func_attrs: Optional[DictionaryAttr] = None,
    op_bundle_operands: Sequence[Value],
    op_bundle_sizes: Sequence[int],
    op_bundle_tags: Optional[ArrayAttr] = None,
    arg_attrs: Optional[ArrayAttr] = None,
    res_attrs: Optional[ArrayAttr] = None,
    no_inline: bool = False,
    always_inline: bool = False,
    inline_hint: bool = False,
    access_groups: Optional[ArrayAttr] = None,
    alias_scopes: Optional[ArrayAttr] = None,
    noalias_scopes: Optional[ArrayAttr] = None,
    tbaa: Optional[ArrayAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    if var_callee_type is not None:
        all_props.append(('var_callee_type', TypeAttr(value=var_callee_type)))
    if callee is not None:
        all_props.append(('callee', FlatSymbolRefAttr(callee)))
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    all_props.append(('CConv', CConvAttr(CallingConv=CConv)))
    all_props.append(('TailCallKind', TailCallKindAttr(tailCallKind=TailCallKind)))
    if memory_effects is not None:
        all_props.append(('memory_effects', memory_effects))
    if convergent:
        all_props.append(('convergent', UnitAttr()))
    if no_unwind:
        all_props.append(('no_unwind', UnitAttr()))
    if will_return:
        all_props.append(('will_return', UnitAttr()))
    if noreturn:
        all_props.append(('noreturn', UnitAttr()))
    if returns_twice:
        all_props.append(('returns_twice', UnitAttr()))
    if hot:
        all_props.append(('hot', UnitAttr()))
    if cold:
        all_props.append(('cold', UnitAttr()))
    if noduplicate:
        all_props.append(('noduplicate', UnitAttr()))
    if no_caller_saved_registers:
        all_props.append(('no_caller_saved_registers', UnitAttr()))
    if nocallback:
        all_props.append(('nocallback', UnitAttr()))
    if modular_format is not None:
        all_props.append(('modular_format', StringAttr(value=modular_format)))
    if nobuiltins is not None:
        all_props.append(('nobuiltins', nobuiltins))
    if allocsize is not None:
        all_props.append(('allocsize', DenseI32ArrayAttr(allocsize)))
    if optsize:
        all_props.append(('optsize', UnitAttr()))
    if minsize:
        all_props.append(('minsize', UnitAttr()))
    if builtin:
        all_props.append(('builtin', UnitAttr()))
    if nobuiltin:
        all_props.append(('nobuiltin', UnitAttr()))
    if save_reg_params:
        all_props.append(('save_reg_params', UnitAttr()))
    if zero_call_used_regs is not None:
        all_props.append(('zero_call_used_regs', StringAttr(value=zero_call_used_regs)))
    if trap_func_name is not None:
        all_props.append(('trap_func_name', StringAttr(value=trap_func_name)))
    if default_func_attrs is not None:
        all_props.append(('default_func_attrs', default_func_attrs))
    all_props.append(('op_bundle_sizes', DenseI32ArrayAttr(op_bundle_sizes)))
    if op_bundle_tags is not None:
        all_props.append(('op_bundle_tags', op_bundle_tags))
    if arg_attrs is not None:
        all_props.append(('arg_attrs', arg_attrs))
    if res_attrs is not None:
        all_props.append(('res_attrs', res_attrs))
    if no_inline:
        all_props.append(('no_inline', UnitAttr()))
    if always_inline:
        all_props.append(('always_inline', UnitAttr()))
    if inline_hint:
        all_props.append(('inline_hint', UnitAttr()))
    if access_groups is not None:
        all_props.append(('access_groups', access_groups))
    if alias_scopes is not None:
        all_props.append(('alias_scopes', alias_scopes))
    if noalias_scopes is not None:
        all_props.append(('noalias_scopes', noalias_scopes))
    if tbaa is not None:
        all_props.append(('tbaa', tbaa))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([len(callee_operands), len(op_bundle_operands)])))
    return add_operation(
        name="llvm.call",
        result_type=result_type,
        operands=[*callee_operands, *op_bundle_operands],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ComdatOp(
    *,
    sym_name: str,
    body: Region,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('sym_name', StringAttr(value=sym_name)))
    return add_operation(
        name="llvm.comdat",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
        regions=[body],
    )


def add_ComdatSelectorOp(
    *,
    sym_name: str,
    comdat: comdat_Comdat,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('sym_name', StringAttr(value=sym_name)))
    all_props.append(('comdat', comdat_ComdatAttr(comdat)))
    return add_operation(
        name="llvm.comdat_selector",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CondBrOp(
    *,
    condition: Value,
    trueDestOperands: Sequence[Value],
    falseDestOperands: Sequence[Value],
    branch_weights: Optional[Sequence[int]] = None,
    loop_annotation: Optional[LoopAnnotationAttr] = None,
    trueDest: BlockLabel,
    falseDest: BlockLabel,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if branch_weights is not None:
        all_props.append(('branch_weights', DenseI32ArrayAttr(branch_weights)))
    if loop_annotation is not None:
        all_props.append(('loop_annotation', loop_annotation))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([1, len(trueDestOperands), len(falseDestOperands)])))
    return add_operation(
        name="llvm.cond_br",
        result_type=None,
        operands=[condition, *trueDestOperands, *falseDestOperands],
        properties=all_props,
        attributes=extra_attributes,
        successors=[trueDest, falseDest],
    )


def add_ConstantOp(
    *,
    res_type: Type,
    value: Attribute,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('value', value))
    return add_operation(
        name="llvm.mlir.constant",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DSOLocalEquivalentOp(
    *,
    res_type: LLVMPointerType,
    function_name: str,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('function_name', FlatSymbolRefAttr(function_name)))
    return add_operation(
        name="llvm.dso_local_equivalent",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ExtractElementOp(
    *,
    vector: Value,
    position: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = vector.type.get_element_type()
    all_props = []
    return add_operation(
        name="llvm.extractelement",
        result_type=res_type,
        operands=[vector, position],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ExtractValueOp(
    *,
    res_type: Type,
    container: Value,
    position: Sequence[int],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('position', DenseI64ArrayAttr(position)))
    return add_operation(
        name="llvm.extractvalue",
        result_type=res_type,
        operands=[container],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FAddOp(
    *,
    lhs: Value,
    rhs: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.fadd",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FCmpOp(
    *,
    predicate: FCmpPredicate,
    lhs: Value,
    rhs: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = _util.get_i1_same_shape(lhs.type)
    all_props = []
    all_props.append(('predicate', FCmpPredicateAttr(predicate)))
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.fcmp",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FDivOp(
    *,
    lhs: Value,
    rhs: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.fdiv",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FMulOp(
    *,
    lhs: Value,
    rhs: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.fmul",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FNegOp(
    *,
    operand: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = operand.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.fneg",
        result_type=res_type,
        operands=[operand],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FPExtOp(
    *,
    res_type: Type,
    arg: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.fpext",
        result_type=res_type,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FPToSIOp(
    *,
    res_type: Type,
    arg: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.fptosi",
        result_type=res_type,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FPToUIOp(
    *,
    res_type: Type,
    arg: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.fptoui",
        result_type=res_type,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FPTruncOp(
    *,
    res_type: Type,
    arg: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.fptrunc",
        result_type=res_type,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FRemOp(
    *,
    lhs: Value,
    rhs: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.frem",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FSubOp(
    *,
    lhs: Value,
    rhs: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.fsub",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FenceOp(
    *,
    ordering: AtomicOrdering,
    syncscope: Optional[str] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('ordering', AtomicOrderingAttr(ordering)))
    if syncscope is not None:
        all_props.append(('syncscope', StringAttr(value=syncscope)))
    return add_operation(
        name="llvm.fence",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FreezeOp(
    *,
    val: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = val.type
    all_props = []
    return add_operation(
        name="llvm.freeze",
        result_type=res_type,
        operands=[val],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GEPOp(
    *,
    res_type: Type,
    base: Value,
    dynamicIndices: Sequence[Value],
    rawConstantIndices: Sequence[int],
    elem_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('rawConstantIndices', DenseI32ArrayAttr(rawConstantIndices)))
    all_props.append(('elem_type', TypeAttr(value=elem_type)))
    return add_operation(
        name="llvm.getelementptr",
        result_type=res_type,
        operands=[base, *dynamicIndices],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GlobalCtorsOp(
    *,
    ctors: ArrayAttr,
    priorities: ArrayAttr,
    data: ArrayAttr,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('ctors', ctors))
    all_props.append(('priorities', priorities))
    all_props.append(('data', data))
    return add_operation(
        name="llvm.mlir.global_ctors",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GlobalDtorsOp(
    *,
    dtors: ArrayAttr,
    priorities: ArrayAttr,
    data: ArrayAttr,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('dtors', dtors))
    all_props.append(('priorities', priorities))
    all_props.append(('data', data))
    return add_operation(
        name="llvm.mlir.global_dtors",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GlobalOp(
    *,
    global_type: Type,
    constant: bool = False,
    sym_name: str,
    linkage: Linkage,
    dso_local: bool = False,
    thread_local_: bool = False,
    externally_initialized: bool = False,
    value: Optional[Attribute] = None,
    alignment: Optional[int] = None,
    addr_space: int = 0,
    unnamed_addr: Optional[UnnamedAddr] = None,
    section: Optional[str] = None,
    comdat: Optional[SymbolRefAttr] = None,
    dbg_exprs: Optional[ArrayAttr] = None,
    visibility_: Visibility = Visibility(0),
    target_specific_attrs: Optional[ArrayAttr] = None,
    initializer: Region,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('global_type', TypeAttr(value=global_type)))
    if constant:
        all_props.append(('constant', UnitAttr()))
    all_props.append(('sym_name', StringAttr(value=sym_name)))
    all_props.append(('linkage', LinkageAttr(linkage=linkage)))
    if dso_local:
        all_props.append(('dso_local', UnitAttr()))
    if thread_local_:
        all_props.append(('thread_local_', UnitAttr()))
    if externally_initialized:
        all_props.append(('externally_initialized', UnitAttr()))
    if value is not None:
        all_props.append(('value', value))
    if alignment is not None:
        all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(64), alignment)))
    all_props.append(('addr_space', IntegerAttr.make(IntegerType.signless(32), addr_space)))
    if unnamed_addr is not None:
        all_props.append(('unnamed_addr', UnnamedAddrAttr(unnamed_addr)))
    if section is not None:
        all_props.append(('section', StringAttr(value=section)))
    if comdat is not None:
        all_props.append(('comdat', comdat))
    if dbg_exprs is not None:
        all_props.append(('dbg_exprs', dbg_exprs))
    all_props.append(('visibility_', VisibilityAttr(visibility_)))
    if target_specific_attrs is not None:
        all_props.append(('target_specific_attrs', target_specific_attrs))
    return add_operation(
        name="llvm.mlir.global",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
        regions=[initializer],
    )


def add_ICmpOp(
    *,
    predicate: ICmpPredicate,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = _util.get_i1_same_shape(lhs.type)
    all_props = []
    all_props.append(('predicate', ICmpPredicateAttr(predicate)))
    return add_operation(
        name="llvm.icmp",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_IFuncOp(
    *,
    sym_name: str,
    i_func_type: Type,
    resolver: str,
    resolver_type: Type,
    linkage: Linkage,
    dso_local: bool = False,
    address_space: int = 0,
    unnamed_addr: UnnamedAddr = UnnamedAddr(0),
    visibility_: Visibility = Visibility(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('sym_name', StringAttr(value=sym_name)))
    all_props.append(('i_func_type', TypeAttr(value=i_func_type)))
    all_props.append(('resolver', FlatSymbolRefAttr(resolver)))
    all_props.append(('resolver_type', TypeAttr(value=resolver_type)))
    all_props.append(('linkage', LinkageAttr(linkage=linkage)))
    if dso_local:
        all_props.append(('dso_local', UnitAttr()))
    all_props.append(('address_space', IntegerAttr.make(IntegerType.signless(32), address_space)))
    all_props.append(('unnamed_addr', UnnamedAddrAttr(unnamed_addr)))
    all_props.append(('visibility_', VisibilityAttr(visibility_)))
    return add_operation(
        name="llvm.mlir.ifunc",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_IndirectBrOp(
    *,
    addr: Value,
    succOperands: Sequence[Value],
    indbr_operand_segments: Sequence[int],
    successors: Sequence[BlockLabel],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('indbr_operand_segments', DenseI32ArrayAttr(indbr_operand_segments)))
    return add_operation(
        name="llvm.indirectbr",
        result_type=None,
        operands=[addr, *succOperands],
        properties=all_props,
        attributes=extra_attributes,
        successors=list(successors),
    )


def add_InlineAsmOp(
    *,
    res_type: Optional[Type],
    operands: Sequence[Value],
    asm_string: str,
    constraints: str,
    has_side_effects: bool = False,
    is_align_stack: bool = False,
    tail_call_kind: tailcallkind_TailCallKind = tailcallkind_TailCallKind(0),
    asm_dialect: Optional[AsmDialect] = None,
    operand_attrs: Optional[ArrayAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    all_props.append(('asm_string', StringAttr(value=asm_string)))
    all_props.append(('constraints', StringAttr(value=constraints)))
    if has_side_effects:
        all_props.append(('has_side_effects', UnitAttr()))
    if is_align_stack:
        all_props.append(('is_align_stack', UnitAttr()))
    all_props.append(('tail_call_kind', TailCallKindAttr(tailCallKind=tail_call_kind)))
    if asm_dialect is not None:
        all_props.append(('asm_dialect', AsmDialectAttr(asm_dialect)))
    if operand_attrs is not None:
        all_props.append(('operand_attrs', operand_attrs))
    return add_operation(
        name="llvm.inline_asm",
        result_type=res_type,
        operands=list(operands),
        properties=all_props,
        attributes=extra_attributes,
    )


def add_InsertElementOp(
    *,
    vector: Value,
    value: Value,
    position: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = vector.type
    all_props = []
    return add_operation(
        name="llvm.insertelement",
        result_type=res_type,
        operands=[vector, value, position],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_InsertValueOp(
    *,
    container: Value,
    value: Value,
    position: Sequence[int],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = container.type
    all_props = []
    all_props.append(('position', DenseI64ArrayAttr(position)))
    return add_operation(
        name="llvm.insertvalue",
        result_type=res_type,
        operands=[container, value],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_IntToPtrOp(
    *,
    res_type: Type,
    arg: Value,
    dereferenceable: Optional[DereferenceableAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if dereferenceable is not None:
        all_props.append(('dereferenceable', dereferenceable))
    return add_operation(
        name="llvm.inttoptr",
        result_type=res_type,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_InvokeOp(
    *,
    result_type: Optional[Type],
    var_callee_type: Optional[LLVMFunctionType] = None,
    callee: Optional[str] = None,
    callee_operands: Sequence[Value],
    arg_attrs: Optional[ArrayAttr] = None,
    res_attrs: Optional[ArrayAttr] = None,
    normalDestOperands: Sequence[Value],
    unwindDestOperands: Sequence[Value],
    branch_weights: Optional[Sequence[int]] = None,
    CConv: cconv_CConv = cconv_CConv(0),
    op_bundle_operands: Sequence[Value],
    op_bundle_sizes: Sequence[int],
    op_bundle_tags: Optional[ArrayAttr] = None,
    normalDest: BlockLabel,
    unwindDest: BlockLabel,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Optional[Value]:
    all_props = []
    if var_callee_type is not None:
        all_props.append(('var_callee_type', TypeAttr(value=var_callee_type)))
    if callee is not None:
        all_props.append(('callee', FlatSymbolRefAttr(callee)))
    if arg_attrs is not None:
        all_props.append(('arg_attrs', arg_attrs))
    if res_attrs is not None:
        all_props.append(('res_attrs', res_attrs))
    if branch_weights is not None:
        all_props.append(('branch_weights', DenseI32ArrayAttr(branch_weights)))
    all_props.append(('CConv', CConvAttr(CallingConv=CConv)))
    all_props.append(('op_bundle_sizes', DenseI32ArrayAttr(op_bundle_sizes)))
    if op_bundle_tags is not None:
        all_props.append(('op_bundle_tags', op_bundle_tags))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([len(callee_operands), len(normalDestOperands),
                                         len(unwindDestOperands), len(op_bundle_operands)])))
    return add_operation(
        name="llvm.invoke",
        result_type=result_type,
        operands=[*callee_operands, *normalDestOperands, *unwindDestOperands,
                  *op_bundle_operands],
        properties=all_props,
        attributes=extra_attributes,
        successors=[normalDest, unwindDest],
    )


def add_LLVMFuncOp(
    *,
    sym_name: str,
    sym_visibility: Optional[str] = None,
    function_type: LLVMFunctionType,
    linkage: Linkage = linkage_Linkage(0),
    dso_local: bool = False,
    CConv: cconv_CConv = cconv_CConv(0),
    comdat: Optional[SymbolRefAttr] = None,
    convergent: bool = False,
    personality: Optional[str] = None,
    garbageCollector: Optional[str] = None,
    passthrough: Optional[ArrayAttr] = None,
    arg_attrs: Optional[ArrayAttr] = None,
    res_attrs: Optional[ArrayAttr] = None,
    function_entry_count: Optional[int] = None,
    memory_effects: Optional[MemoryEffectsAttr] = None,
    visibility_: Visibility = Visibility(0),
    arm_streaming: bool = False,
    arm_locally_streaming: bool = False,
    arm_streaming_compatible: bool = False,
    arm_new_za: bool = False,
    arm_in_za: bool = False,
    arm_out_za: bool = False,
    arm_inout_za: bool = False,
    arm_preserves_za: bool = False,
    section: Optional[str] = None,
    unnamed_addr: Optional[UnnamedAddr] = None,
    alignment: Optional[int] = None,
    vscale_range: Optional[VScaleRangeAttr] = None,
    frame_pointer: Optional[FramePointerKindAttr] = None,
    target_cpu: Optional[str] = None,
    tune_cpu: Optional[str] = None,
    reciprocal_estimates: Optional[str] = None,
    prefer_vector_width: Optional[str] = None,
    target_features: Optional[TargetFeaturesAttr] = None,
    no_signed_zeros_fp_math: Optional[bool] = None,
    denormal_fpenv: Optional[DenormalFPEnvAttr] = None,
    fp_contract: Optional[str] = None,
    instrument_function_entry: Optional[str] = None,
    instrument_function_exit: Optional[str] = None,
    no_inline: bool = False,
    always_inline: bool = False,
    inline_hint: bool = False,
    no_unwind: bool = False,
    will_return: bool = False,
    noreturn: bool = False,
    optimize_none: bool = False,
    returns_twice: bool = False,
    hot: bool = False,
    cold: bool = False,
    noduplicate: bool = False,
    no_caller_saved_registers: bool = False,
    nocallback: bool = False,
    modular_format: Optional[str] = None,
    nobuiltins: Optional[ArrayAttr] = None,
    allocsize: Optional[Sequence[int]] = None,
    optsize: Optional[bool] = None,
    minsize: Optional[bool] = None,
    save_reg_params: Optional[bool] = None,
    zero_call_used_regs: Optional[str] = None,
    default_func_attrs: Optional[DictionaryAttr] = None,
    vec_type_hint: Optional[VecTypeHintAttr] = None,
    work_group_size_hint: Optional[Sequence[int]] = None,
    reqd_work_group_size: Optional[Sequence[int]] = None,
    intel_reqd_sub_group_size: Optional[int] = None,
    uwtable_kind: Optional[UWTableKindAttr] = None,
    use_sample_profile: Optional[bool] = None,
    body: Region,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('sym_name', StringAttr(value=sym_name)))
    if sym_visibility is not None:
        all_props.append(('sym_visibility', StringAttr(value=sym_visibility)))
    all_props.append(('function_type', TypeAttr(value=function_type)))
    all_props.append(('linkage', LinkageAttr(linkage=linkage)))
    if dso_local:
        all_props.append(('dso_local', UnitAttr()))
    all_props.append(('CConv', CConvAttr(CallingConv=CConv)))
    if comdat is not None:
        all_props.append(('comdat', comdat))
    if convergent:
        all_props.append(('convergent', UnitAttr()))
    if personality is not None:
        all_props.append(('personality', FlatSymbolRefAttr(personality)))
    if garbageCollector is not None:
        all_props.append(('garbageCollector', StringAttr(value=garbageCollector)))
    if passthrough is not None:
        all_props.append(('passthrough', passthrough))
    if arg_attrs is not None:
        all_props.append(('arg_attrs', arg_attrs))
    if res_attrs is not None:
        all_props.append(('res_attrs', res_attrs))
    if function_entry_count is not None:
        all_props.append(('function_entry_count',
                          IntegerAttr.make(IntegerType.signless(64), function_entry_count)))
    if memory_effects is not None:
        all_props.append(('memory_effects', memory_effects))
    all_props.append(('visibility_', VisibilityAttr(visibility_)))
    if arm_streaming:
        all_props.append(('arm_streaming', UnitAttr()))
    if arm_locally_streaming:
        all_props.append(('arm_locally_streaming', UnitAttr()))
    if arm_streaming_compatible:
        all_props.append(('arm_streaming_compatible', UnitAttr()))
    if arm_new_za:
        all_props.append(('arm_new_za', UnitAttr()))
    if arm_in_za:
        all_props.append(('arm_in_za', UnitAttr()))
    if arm_out_za:
        all_props.append(('arm_out_za', UnitAttr()))
    if arm_inout_za:
        all_props.append(('arm_inout_za', UnitAttr()))
    if arm_preserves_za:
        all_props.append(('arm_preserves_za', UnitAttr()))
    if section is not None:
        all_props.append(('section', StringAttr(value=section)))
    if unnamed_addr is not None:
        all_props.append(('unnamed_addr', UnnamedAddrAttr(unnamed_addr)))
    if alignment is not None:
        all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(64), alignment)))
    if vscale_range is not None:
        all_props.append(('vscale_range', vscale_range))
    if frame_pointer is not None:
        all_props.append(('frame_pointer', frame_pointer))
    if target_cpu is not None:
        all_props.append(('target_cpu', StringAttr(value=target_cpu)))
    if tune_cpu is not None:
        all_props.append(('tune_cpu', StringAttr(value=tune_cpu)))
    if reciprocal_estimates is not None:
        all_props.append(('reciprocal_estimates', StringAttr(value=reciprocal_estimates)))
    if prefer_vector_width is not None:
        all_props.append(('prefer_vector_width', StringAttr(value=prefer_vector_width)))
    if target_features is not None:
        all_props.append(('target_features', target_features))
    if no_signed_zeros_fp_math is not None:
        all_props.append(('no_signed_zeros_fp_math', BoolAttr(value=no_signed_zeros_fp_math)))
    if denormal_fpenv is not None:
        all_props.append(('denormal_fpenv', denormal_fpenv))
    if fp_contract is not None:
        all_props.append(('fp_contract', StringAttr(value=fp_contract)))
    if instrument_function_entry is not None:
        all_props.append(('instrument_function_entry',
                          StringAttr(value=instrument_function_entry)))
    if instrument_function_exit is not None:
        all_props.append(('instrument_function_exit', StringAttr(value=instrument_function_exit)))
    if no_inline:
        all_props.append(('no_inline', UnitAttr()))
    if always_inline:
        all_props.append(('always_inline', UnitAttr()))
    if inline_hint:
        all_props.append(('inline_hint', UnitAttr()))
    if no_unwind:
        all_props.append(('no_unwind', UnitAttr()))
    if will_return:
        all_props.append(('will_return', UnitAttr()))
    if noreturn:
        all_props.append(('noreturn', UnitAttr()))
    if optimize_none:
        all_props.append(('optimize_none', UnitAttr()))
    if returns_twice:
        all_props.append(('returns_twice', UnitAttr()))
    if hot:
        all_props.append(('hot', UnitAttr()))
    if cold:
        all_props.append(('cold', UnitAttr()))
    if noduplicate:
        all_props.append(('noduplicate', UnitAttr()))
    if no_caller_saved_registers:
        all_props.append(('no_caller_saved_registers', UnitAttr()))
    if nocallback:
        all_props.append(('nocallback', UnitAttr()))
    if modular_format is not None:
        all_props.append(('modular_format', StringAttr(value=modular_format)))
    if nobuiltins is not None:
        all_props.append(('nobuiltins', nobuiltins))
    if allocsize is not None:
        all_props.append(('allocsize', DenseI32ArrayAttr(allocsize)))
    if optsize is not None:
        all_props.append(('optsize', UnitAttr()))
    if minsize is not None:
        all_props.append(('minsize', UnitAttr()))
    if save_reg_params is not None:
        all_props.append(('save_reg_params', UnitAttr()))
    if zero_call_used_regs is not None:
        all_props.append(('zero_call_used_regs', StringAttr(value=zero_call_used_regs)))
    if default_func_attrs is not None:
        all_props.append(('default_func_attrs', default_func_attrs))
    if vec_type_hint is not None:
        all_props.append(('vec_type_hint', vec_type_hint))
    if work_group_size_hint is not None:
        all_props.append(('work_group_size_hint', DenseI32ArrayAttr(work_group_size_hint)))
    if reqd_work_group_size is not None:
        all_props.append(('reqd_work_group_size', DenseI32ArrayAttr(reqd_work_group_size)))
    if intel_reqd_sub_group_size is not None:
        all_props.append(('intel_reqd_sub_group_size',
                          IntegerAttr.make(IntegerType.signless(32), intel_reqd_sub_group_size)))
    if uwtable_kind is not None:
        all_props.append(('uwtable_kind', uwtable_kind))
    if use_sample_profile is not None:
        all_props.append(('use_sample_profile', BoolAttr(value=use_sample_profile)))
    return add_operation(
        name="llvm.func",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
        regions=[body],
    )


def add_LShrOp(
    *,
    lhs: Value,
    rhs: Value,
    isExact: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    if isExact:
        all_props.append(('isExact', UnitAttr()))
    return add_operation(
        name="llvm.lshr",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LandingpadOp(
    *,
    res_type: Type,
    cleanup: bool = False,
    operands: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if cleanup:
        all_props.append(('cleanup', UnitAttr()))
    return add_operation(
        name="llvm.landingpad",
        result_type=res_type,
        operands=list(operands),
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LinkerOptionsOp(
    *,
    options: ArrayAttr,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('options', options))
    return add_operation(
        name="llvm.linker_options",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LoadOp(
    *,
    res_type: Type,
    addr: Value,
    alignment: Optional[int] = None,
    volatile_: bool = False,
    nontemporal: bool = False,
    invariant: bool = False,
    invariantGroup: bool = False,
    ordering: AtomicOrdering = AtomicOrdering(0),
    syncscope: Optional[str] = None,
    dereferenceable: Optional[DereferenceableAttr] = None,
    access_groups: Optional[ArrayAttr] = None,
    alias_scopes: Optional[ArrayAttr] = None,
    noalias_scopes: Optional[ArrayAttr] = None,
    tbaa: Optional[ArrayAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if alignment is not None:
        all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(64), alignment)))
    if volatile_:
        all_props.append(('volatile_', UnitAttr()))
    if nontemporal:
        all_props.append(('nontemporal', UnitAttr()))
    if invariant:
        all_props.append(('invariant', UnitAttr()))
    if invariantGroup:
        all_props.append(('invariantGroup', UnitAttr()))
    all_props.append(('ordering', AtomicOrderingAttr(ordering)))
    if syncscope is not None:
        all_props.append(('syncscope', StringAttr(value=syncscope)))
    if dereferenceable is not None:
        all_props.append(('dereferenceable', dereferenceable))
    if access_groups is not None:
        all_props.append(('access_groups', access_groups))
    if alias_scopes is not None:
        all_props.append(('alias_scopes', alias_scopes))
    if noalias_scopes is not None:
        all_props.append(('noalias_scopes', noalias_scopes))
    if tbaa is not None:
        all_props.append(('tbaa', tbaa))
    return add_operation(
        name="llvm.load",
        result_type=res_type,
        operands=[addr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ModuleFlagsOp(
    *,
    flags: ArrayAttr,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('flags', flags))
    return add_operation(
        name="llvm.module_flags",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MulOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    return add_operation(
        name="llvm.mul",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_NamedMetadataOp(
    *,
    metadata_name: str,
    nodes: ArrayAttr,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('metadata_name', StringAttr(value=metadata_name)))
    all_props.append(('nodes', nodes))
    return add_operation(
        name="llvm.named_metadata",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_NoneTokenOp(
    *,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = LLVMTokenType()
    all_props = []
    return add_operation(
        name="llvm.mlir.none",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_OrOp(
    *,
    lhs: Value,
    rhs: Value,
    isDisjoint: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    if isDisjoint:
        all_props.append(('isDisjoint', UnitAttr()))
    return add_operation(
        name="llvm.or",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_PoisonOp(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.mlir.poison",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_PtrToAddrOp(
    *,
    res_type: Type,
    arg: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.ptrtoaddr",
        result_type=res_type,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_PtrToIntOp(
    *,
    res_type: Type,
    arg: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.ptrtoint",
        result_type=res_type,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ResumeOp(
    *,
    value: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="llvm.resume",
        result_type=None,
        operands=[value],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ReturnOp(
    *,
    arg: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="llvm.return",
        result_type=None,
        operands=[] if arg is None else [arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SDivOp(
    *,
    lhs: Value,
    rhs: Value,
    isExact: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    if isExact:
        all_props.append(('isExact', UnitAttr()))
    return add_operation(
        name="llvm.sdiv",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SExtOp(
    *,
    res_type: Type,
    arg: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.sext",
        result_type=res_type,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SIToFPOp(
    *,
    res_type: Type,
    arg: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.sitofp",
        result_type=res_type,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SRemOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    return add_operation(
        name="llvm.srem",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SelectOp(
    *,
    condition: Value,
    trueValue: Value,
    falseValue: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = falseValue.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.select",
        result_type=res_type,
        operands=[condition, trueValue, falseValue],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ShlOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    return add_operation(
        name="llvm.shl",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ShuffleVectorOp(
    *,
    res_type: VectorType,
    v1: Value,
    v2: Value,
    mask: Sequence[int],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('mask', DenseI32ArrayAttr(mask)))
    return add_operation(
        name="llvm.shufflevector",
        result_type=res_type,
        operands=[v1, v2],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_StoreOp(
    *,
    value: Value,
    addr: Value,
    alignment: Optional[int] = None,
    volatile_: bool = False,
    nontemporal: bool = False,
    invariantGroup: bool = False,
    ordering: AtomicOrdering = AtomicOrdering(0),
    syncscope: Optional[str] = None,
    access_groups: Optional[ArrayAttr] = None,
    alias_scopes: Optional[ArrayAttr] = None,
    noalias_scopes: Optional[ArrayAttr] = None,
    tbaa: Optional[ArrayAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if alignment is not None:
        all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(64), alignment)))
    if volatile_:
        all_props.append(('volatile_', UnitAttr()))
    if nontemporal:
        all_props.append(('nontemporal', UnitAttr()))
    if invariantGroup:
        all_props.append(('invariantGroup', UnitAttr()))
    all_props.append(('ordering', AtomicOrderingAttr(ordering)))
    if syncscope is not None:
        all_props.append(('syncscope', StringAttr(value=syncscope)))
    if access_groups is not None:
        all_props.append(('access_groups', access_groups))
    if alias_scopes is not None:
        all_props.append(('alias_scopes', alias_scopes))
    if noalias_scopes is not None:
        all_props.append(('noalias_scopes', noalias_scopes))
    if tbaa is not None:
        all_props.append(('tbaa', tbaa))
    return add_operation(
        name="llvm.store",
        result_type=None,
        operands=[value, addr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SubOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    return add_operation(
        name="llvm.sub",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SwitchOp(
    *,
    value: Value,
    defaultOperands: Sequence[Value],
    caseOperands: Sequence[Value],
    case_values: Optional[DenseIntElementsAttr] = None,
    case_operand_segments: Sequence[int],
    branch_weights: Optional[Sequence[int]] = None,
    defaultDestination: BlockLabel,
    caseDestinations: Sequence[BlockLabel],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if case_values is not None:
        all_props.append(('case_values', case_values))
    all_props.append(('case_operand_segments', DenseI32ArrayAttr(case_operand_segments)))
    if branch_weights is not None:
        all_props.append(('branch_weights', DenseI32ArrayAttr(branch_weights)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([1, len(defaultOperands), len(caseOperands)])))
    return add_operation(
        name="llvm.switch",
        result_type=None,
        operands=[value, *defaultOperands, *caseOperands],
        properties=all_props,
        attributes=extra_attributes,
        successors=[defaultDestination, *caseDestinations],
    )


def add_TruncOp(
    *,
    res_type: Type,
    arg: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.trunc",
        result_type=res_type,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_UDivOp(
    *,
    lhs: Value,
    rhs: Value,
    isExact: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    if isExact:
        all_props.append(('isExact', UnitAttr()))
    return add_operation(
        name="llvm.udiv",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_UIToFPOp(
    *,
    res_type: Type,
    arg: Value,
    nonNeg: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if nonNeg:
        all_props.append(('nonNeg', UnitAttr()))
    return add_operation(
        name="llvm.uitofp",
        result_type=res_type,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_URemOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    return add_operation(
        name="llvm.urem",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_UndefOp(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.mlir.undef",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_UnreachableOp(
    *,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="llvm.unreachable",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VaArgOp(
    *,
    res_type: Type,
    arg: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.va_arg",
        result_type=res_type,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_XOrOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = lhs.type
    all_props = []
    return add_operation(
        name="llvm.xor",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ZExtOp(
    *,
    res_type: Type,
    arg: Value,
    nonNeg: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if nonNeg:
        all_props.append(('nonNeg', UnitAttr()))
    return add_operation(
        name="llvm.zext",
        result_type=res_type,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ZeroOp(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.mlir.zero",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ACosOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.acos",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ASinOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.asin",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ATan2Op(
    *,
    a: Value,
    b: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.atan2",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ATanOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.atan",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AbsOp(
    *,
    res_type: Type,
    in_: Value,
    is_int_min_poison: bool,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('is_int_min_poison',
                      IntegerAttr.make(IntegerType.signless(1), int(is_int_min_poison))))
    return add_operation(
        name="llvm.intr.abs",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Annotation(
    *,
    integer: Value,
    annotation: Value,
    fileName: Value,
    line: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = integer.type
    all_props = []
    return add_operation(
        name="llvm.intr.annotation",
        result_type=res_type,
        operands=[integer, annotation, fileName, line],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AssumeOp(
    *,
    cond: Value,
    op_bundle_operands: Sequence[Value],
    op_bundle_sizes: Sequence[int],
    op_bundle_tags: Optional[ArrayAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('op_bundle_sizes', DenseI32ArrayAttr(op_bundle_sizes)))
    if op_bundle_tags is not None:
        all_props.append(('op_bundle_tags', op_bundle_tags))
    return add_operation(
        name="llvm.intr.assume",
        result_type=None,
        operands=[cond, *op_bundle_operands],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BitReverseOp(
    *,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    return add_operation(
        name="llvm.intr.bitreverse",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ByteSwapOp(
    *,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    return add_operation(
        name="llvm.intr.bswap",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConstrainedFAddIntr(
    *,
    arg_0: Value,
    arg_1: Value,
    roundingmode: RoundingMode,
    fpExceptionBehavior: FPExceptionBehavior,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = arg_0.type
    all_props = []
    all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    all_props.append(('fpExceptionBehavior', FPExceptionBehaviorAttr(fpExceptionBehavior)))
    return add_operation(
        name="llvm.intr.experimental.constrained.fadd",
        result_type=res_type,
        operands=[arg_0, arg_1],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConstrainedFDivIntr(
    *,
    arg_0: Value,
    arg_1: Value,
    roundingmode: RoundingMode,
    fpExceptionBehavior: FPExceptionBehavior,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = arg_0.type
    all_props = []
    all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    all_props.append(('fpExceptionBehavior', FPExceptionBehaviorAttr(fpExceptionBehavior)))
    return add_operation(
        name="llvm.intr.experimental.constrained.fdiv",
        result_type=res_type,
        operands=[arg_0, arg_1],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConstrainedFMAIntr(
    *,
    arg_0: Value,
    arg_1: Value,
    arg_2: Value,
    roundingmode: RoundingMode,
    fpExceptionBehavior: FPExceptionBehavior,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = arg_0.type
    all_props = []
    all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    all_props.append(('fpExceptionBehavior', FPExceptionBehaviorAttr(fpExceptionBehavior)))
    return add_operation(
        name="llvm.intr.experimental.constrained.fma",
        result_type=res_type,
        operands=[arg_0, arg_1, arg_2],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConstrainedFMulAddIntr(
    *,
    arg_0: Value,
    arg_1: Value,
    arg_2: Value,
    roundingmode: RoundingMode,
    fpExceptionBehavior: FPExceptionBehavior,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = arg_0.type
    all_props = []
    all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    all_props.append(('fpExceptionBehavior', FPExceptionBehaviorAttr(fpExceptionBehavior)))
    return add_operation(
        name="llvm.intr.experimental.constrained.fmuladd",
        result_type=res_type,
        operands=[arg_0, arg_1, arg_2],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConstrainedFMulIntr(
    *,
    arg_0: Value,
    arg_1: Value,
    roundingmode: RoundingMode,
    fpExceptionBehavior: FPExceptionBehavior,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = arg_0.type
    all_props = []
    all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    all_props.append(('fpExceptionBehavior', FPExceptionBehaviorAttr(fpExceptionBehavior)))
    return add_operation(
        name="llvm.intr.experimental.constrained.fmul",
        result_type=res_type,
        operands=[arg_0, arg_1],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConstrainedFPExtIntr(
    *,
    res_type: Type,
    arg_0: Value,
    fpExceptionBehavior: FPExceptionBehavior,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('fpExceptionBehavior', FPExceptionBehaviorAttr(fpExceptionBehavior)))
    return add_operation(
        name="llvm.intr.experimental.constrained.fpext",
        result_type=res_type,
        operands=[arg_0],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConstrainedFPTruncIntr(
    *,
    res_type: Type,
    arg_0: Value,
    roundingmode: RoundingMode,
    fpExceptionBehavior: FPExceptionBehavior,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    all_props.append(('fpExceptionBehavior', FPExceptionBehaviorAttr(fpExceptionBehavior)))
    return add_operation(
        name="llvm.intr.experimental.constrained.fptrunc",
        result_type=res_type,
        operands=[arg_0],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConstrainedFRemIntr(
    *,
    arg_0: Value,
    arg_1: Value,
    roundingmode: RoundingMode,
    fpExceptionBehavior: FPExceptionBehavior,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = arg_0.type
    all_props = []
    all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    all_props.append(('fpExceptionBehavior', FPExceptionBehaviorAttr(fpExceptionBehavior)))
    return add_operation(
        name="llvm.intr.experimental.constrained.frem",
        result_type=res_type,
        operands=[arg_0, arg_1],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConstrainedFSubIntr(
    *,
    arg_0: Value,
    arg_1: Value,
    roundingmode: RoundingMode,
    fpExceptionBehavior: FPExceptionBehavior,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = arg_0.type
    all_props = []
    all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    all_props.append(('fpExceptionBehavior', FPExceptionBehaviorAttr(fpExceptionBehavior)))
    return add_operation(
        name="llvm.intr.experimental.constrained.fsub",
        result_type=res_type,
        operands=[arg_0, arg_1],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConstrainedSIToFP(
    *,
    res_type: Type,
    arg_0: Value,
    roundingmode: RoundingMode,
    fpExceptionBehavior: FPExceptionBehavior,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    all_props.append(('fpExceptionBehavior', FPExceptionBehaviorAttr(fpExceptionBehavior)))
    return add_operation(
        name="llvm.intr.experimental.constrained.sitofp",
        result_type=res_type,
        operands=[arg_0],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConstrainedUIToFP(
    *,
    res_type: Type,
    arg_0: Value,
    roundingmode: RoundingMode,
    fpExceptionBehavior: FPExceptionBehavior,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    all_props.append(('fpExceptionBehavior', FPExceptionBehaviorAttr(fpExceptionBehavior)))
    return add_operation(
        name="llvm.intr.experimental.constrained.uitofp",
        result_type=res_type,
        operands=[arg_0],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CopySignOp(
    *,
    a: Value,
    b: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.copysign",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CoroAlignOp(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.coro.align",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CoroBeginOp(
    *,
    res_type: Type,
    token: Value,
    mem: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.coro.begin",
        result_type=res_type,
        operands=[token, mem],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CoroEndOp(
    *,
    res_type: Type,
    handle: Value,
    unwind: Value,
    retvals: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.coro.end",
        result_type=res_type,
        operands=[handle, unwind, retvals],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CoroFreeOp(
    *,
    res_type: Type,
    id: Value,
    handle: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.coro.free",
        result_type=res_type,
        operands=[id, handle],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CoroIdOp(
    *,
    res_type: Type,
    align: Value,
    promise: Value,
    coroaddr: Value,
    fnaddrs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.coro.id",
        result_type=res_type,
        operands=[align, promise, coroaddr, fnaddrs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CoroPromiseOp(
    *,
    res_type: LLVMPointerType,
    handle: Value,
    align: Value,
    from_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.coro.promise",
        result_type=res_type,
        operands=[handle, align, from_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CoroResumeOp(
    *,
    handle: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="llvm.intr.coro.resume",
        result_type=None,
        operands=[handle],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CoroSaveOp(
    *,
    res_type: Type,
    handle: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.coro.save",
        result_type=res_type,
        operands=[handle],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CoroSizeOp(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.coro.size",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CoroSuspendOp(
    *,
    res_type: Type,
    save: Value,
    final: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.coro.suspend",
        result_type=res_type,
        operands=[save, final],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CosOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.cos",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CoshOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.cosh",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CountLeadingZerosOp(
    *,
    in_: Value,
    is_zero_poison: bool,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('is_zero_poison',
                      IntegerAttr.make(IntegerType.signless(1), int(is_zero_poison))))
    return add_operation(
        name="llvm.intr.ctlz",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CountTrailingZerosOp(
    *,
    in_: Value,
    is_zero_poison: bool,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('is_zero_poison',
                      IntegerAttr.make(IntegerType.signless(1), int(is_zero_poison))))
    return add_operation(
        name="llvm.intr.cttz",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CtPopOp(
    *,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    return add_operation(
        name="llvm.intr.ctpop",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DbgDeclareOp(
    *,
    addr: Value,
    varInfo: DILocalVariableAttr,
    locationExpr: DIExpressionAttr = DIExpressionAttr(),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('varInfo', varInfo))
    all_props.append(('locationExpr', locationExpr))
    return add_operation(
        name="llvm.intr.dbg.declare",
        result_type=None,
        operands=[addr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DbgLabelOp(
    *,
    label: DILabelAttr,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('label', label))
    return add_operation(
        name="llvm.intr.dbg.label",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DbgValueOp(
    *,
    value: Value,
    varInfo: DILocalVariableAttr,
    locationExpr: DIExpressionAttr = DIExpressionAttr(),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('varInfo', varInfo))
    all_props.append(('locationExpr', locationExpr))
    return add_operation(
        name="llvm.intr.dbg.value",
        result_type=None,
        operands=[value],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DebugTrap(
    *,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="llvm.intr.debugtrap",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_EhTypeidForOp(
    *,
    res_type: Type,
    type_info: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.eh.typeid.for",
        result_type=res_type,
        operands=[type_info],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Exp2Op(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.exp2",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Exp10Op(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.exp10",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ExpOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.exp",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ExpectOp(
    *,
    val: Value,
    expected: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = val.type
    all_props = []
    return add_operation(
        name="llvm.intr.expect",
        result_type=res_type,
        operands=[val, expected],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ExpectWithProbabilityOp(
    *,
    val: Value,
    expected: Value,
    prob: APFloat,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = expected.type
    all_props = []
    all_props.append(('prob', FloatAttr(type=Float64Type(), value=prob)))
    return add_operation(
        name="llvm.intr.expect.with.probability",
        result_type=res_type,
        operands=[val, expected],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FAbsOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.fabs",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FCeilOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.ceil",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FFloorOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.floor",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FMAOp(
    *,
    a: Value,
    b: Value,
    c: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.fma",
        result_type=res_type,
        operands=[a, b, c],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FMulAddOp(
    *,
    a: Value,
    b: Value,
    c: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.fmuladd",
        result_type=res_type,
        operands=[a, b, c],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FTruncOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.trunc",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FakeUseOp(
    *,
    args: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="llvm.intr.fake.use",
        result_type=None,
        operands=list(args),
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FractionExpOp(
    *,
    res_type: Type,
    val: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.frexp",
        result_type=res_type,
        operands=[val],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FshlOp(
    *,
    a: Value,
    b: Value,
    c: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    return add_operation(
        name="llvm.intr.fshl",
        result_type=res_type,
        operands=[a, b, c],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FshrOp(
    *,
    a: Value,
    b: Value,
    c: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    return add_operation(
        name="llvm.intr.fshr",
        result_type=res_type,
        operands=[a, b, c],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GetActiveLaneMaskOp(
    *,
    res_type: Type,
    base: Value,
    n: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.get.active.lane.mask",
        result_type=res_type,
        operands=[base, n],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_InvariantEndOp(
    *,
    start: Value,
    size: int,
    ptr: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('size', IntegerAttr.make(IntegerType.signless(64), size)))
    return add_operation(
        name="llvm.intr.invariant.end",
        result_type=None,
        operands=[start, ptr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_InvariantStartOp(
    *,
    size: int,
    ptr: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = LLVMPointerType(0)
    all_props = []
    all_props.append(('size', IntegerAttr.make(IntegerType.signless(64), size)))
    return add_operation(
        name="llvm.intr.invariant.start",
        result_type=res_type,
        operands=[ptr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_IsConstantOp(
    *,
    val: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = IntegerType.signless(1)
    all_props = []
    return add_operation(
        name="llvm.intr.is.constant",
        result_type=res_type,
        operands=[val],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_IsFPClass(
    *,
    res_type: Type,
    in_: Value,
    bit: int,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('bit', IntegerAttr.make(IntegerType.signless(32), bit)))
    return add_operation(
        name="llvm.intr.is.fpclass",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LaunderInvariantGroupOp(
    *,
    ptr: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = ptr.type
    all_props = []
    return add_operation(
        name="llvm.intr.launder.invariant.group",
        result_type=res_type,
        operands=[ptr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LifetimeEndOp(
    *,
    ptr: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="llvm.intr.lifetime.end",
        result_type=None,
        operands=[ptr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LifetimeStartOp(
    *,
    ptr: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="llvm.intr.lifetime.start",
        result_type=None,
        operands=[ptr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LlrintOp(
    *,
    res_type: Type,
    val: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.llrint",
        result_type=res_type,
        operands=[val],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LlroundOp(
    *,
    res_type: Type,
    val: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.llround",
        result_type=res_type,
        operands=[val],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LoadExpOp(
    *,
    res_type: Type,
    val: Value,
    power: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.ldexp",
        result_type=res_type,
        operands=[val, power],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Log2Op(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.log2",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Log10Op(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.log10",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LogOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.log",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LrintOp(
    *,
    res_type: Type,
    val: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.lrint",
        result_type=res_type,
        operands=[val],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LroundOp(
    *,
    res_type: Type,
    val: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.lround",
        result_type=res_type,
        operands=[val],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MaskedLoadOp(
    *,
    res_type: VectorType,
    data: Value,
    mask: Value,
    pass_thru: Optional[Value] = None,
    alignment: int,
    nontemporal: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(32), alignment)))
    if nontemporal:
        all_props.append(('nontemporal', UnitAttr()))
    return add_operation(
        name="llvm.intr.masked.load",
        result_type=res_type,
        operands=[data, mask, *([] if pass_thru is None else [pass_thru])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MaskedStoreOp(
    *,
    value: Value,
    data: Value,
    mask: Value,
    alignment: int,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(32), alignment)))
    return add_operation(
        name="llvm.intr.masked.store",
        result_type=None,
        operands=[value, data, mask],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MatrixColumnMajorLoadOp(
    *,
    res_type: VectorType,
    data: Value,
    stride: Value,
    isVolatile: bool,
    rows: int,
    columns: int,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('isVolatile', IntegerAttr.make(IntegerType.signless(1), int(isVolatile))))
    all_props.append(('rows', IntegerAttr.make(IntegerType.signless(32), rows)))
    all_props.append(('columns', IntegerAttr.make(IntegerType.signless(32), columns)))
    return add_operation(
        name="llvm.intr.matrix.column.major.load",
        result_type=res_type,
        operands=[data, stride],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MatrixColumnMajorStoreOp(
    *,
    matrix: Value,
    data: Value,
    stride: Value,
    isVolatile: bool,
    rows: int,
    columns: int,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('isVolatile', IntegerAttr.make(IntegerType.signless(1), int(isVolatile))))
    all_props.append(('rows', IntegerAttr.make(IntegerType.signless(32), rows)))
    all_props.append(('columns', IntegerAttr.make(IntegerType.signless(32), columns)))
    return add_operation(
        name="llvm.intr.matrix.column.major.store",
        result_type=None,
        operands=[matrix, data, stride],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MatrixMultiplyOp(
    *,
    res_type: VectorType,
    lhs: Value,
    rhs: Value,
    lhs_rows: int,
    lhs_columns: int,
    rhs_columns: int,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('lhs_rows', IntegerAttr.make(IntegerType.signless(32), lhs_rows)))
    all_props.append(('lhs_columns', IntegerAttr.make(IntegerType.signless(32), lhs_columns)))
    all_props.append(('rhs_columns', IntegerAttr.make(IntegerType.signless(32), rhs_columns)))
    return add_operation(
        name="llvm.intr.matrix.multiply",
        result_type=res_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MatrixTransposeOp(
    *,
    res_type: VectorType,
    matrix: Value,
    rows: int,
    columns: int,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('rows', IntegerAttr.make(IntegerType.signless(32), rows)))
    all_props.append(('columns', IntegerAttr.make(IntegerType.signless(32), columns)))
    return add_operation(
        name="llvm.intr.matrix.transpose",
        result_type=res_type,
        operands=[matrix],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MaxNumOp(
    *,
    a: Value,
    b: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.maxnum",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MaximumOp(
    *,
    a: Value,
    b: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.maximum",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MemcpyInlineOp(
    *,
    dst: Value,
    src: Value,
    len: IntegerAttr,
    isVolatile: bool,
    access_groups: Optional[ArrayAttr] = None,
    alias_scopes: Optional[ArrayAttr] = None,
    noalias_scopes: Optional[ArrayAttr] = None,
    tbaa: Optional[ArrayAttr] = None,
    arg_attrs: Optional[ArrayAttr] = None,
    res_attrs: Optional[ArrayAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('len', len))
    all_props.append(('isVolatile', IntegerAttr.make(IntegerType.signless(1), int(isVolatile))))
    if access_groups is not None:
        all_props.append(('access_groups', access_groups))
    if alias_scopes is not None:
        all_props.append(('alias_scopes', alias_scopes))
    if noalias_scopes is not None:
        all_props.append(('noalias_scopes', noalias_scopes))
    if tbaa is not None:
        all_props.append(('tbaa', tbaa))
    if arg_attrs is not None:
        all_props.append(('arg_attrs', arg_attrs))
    if res_attrs is not None:
        all_props.append(('res_attrs', res_attrs))
    return add_operation(
        name="llvm.intr.memcpy.inline",
        result_type=None,
        operands=[dst, src],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MemcpyOp(
    *,
    dst: Value,
    src: Value,
    len: Value,
    isVolatile: bool,
    access_groups: Optional[ArrayAttr] = None,
    alias_scopes: Optional[ArrayAttr] = None,
    noalias_scopes: Optional[ArrayAttr] = None,
    tbaa: Optional[ArrayAttr] = None,
    arg_attrs: Optional[ArrayAttr] = None,
    res_attrs: Optional[ArrayAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('isVolatile', IntegerAttr.make(IntegerType.signless(1), int(isVolatile))))
    if access_groups is not None:
        all_props.append(('access_groups', access_groups))
    if alias_scopes is not None:
        all_props.append(('alias_scopes', alias_scopes))
    if noalias_scopes is not None:
        all_props.append(('noalias_scopes', noalias_scopes))
    if tbaa is not None:
        all_props.append(('tbaa', tbaa))
    if arg_attrs is not None:
        all_props.append(('arg_attrs', arg_attrs))
    if res_attrs is not None:
        all_props.append(('res_attrs', res_attrs))
    return add_operation(
        name="llvm.intr.memcpy",
        result_type=None,
        operands=[dst, src, len],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MemmoveOp(
    *,
    dst: Value,
    src: Value,
    len: Value,
    isVolatile: bool,
    access_groups: Optional[ArrayAttr] = None,
    alias_scopes: Optional[ArrayAttr] = None,
    noalias_scopes: Optional[ArrayAttr] = None,
    tbaa: Optional[ArrayAttr] = None,
    arg_attrs: Optional[ArrayAttr] = None,
    res_attrs: Optional[ArrayAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('isVolatile', IntegerAttr.make(IntegerType.signless(1), int(isVolatile))))
    if access_groups is not None:
        all_props.append(('access_groups', access_groups))
    if alias_scopes is not None:
        all_props.append(('alias_scopes', alias_scopes))
    if noalias_scopes is not None:
        all_props.append(('noalias_scopes', noalias_scopes))
    if tbaa is not None:
        all_props.append(('tbaa', tbaa))
    if arg_attrs is not None:
        all_props.append(('arg_attrs', arg_attrs))
    if res_attrs is not None:
        all_props.append(('res_attrs', res_attrs))
    return add_operation(
        name="llvm.intr.memmove",
        result_type=None,
        operands=[dst, src, len],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MemsetInlineOp(
    *,
    dst: Value,
    val: Value,
    len: IntegerAttr,
    isVolatile: bool,
    access_groups: Optional[ArrayAttr] = None,
    alias_scopes: Optional[ArrayAttr] = None,
    noalias_scopes: Optional[ArrayAttr] = None,
    tbaa: Optional[ArrayAttr] = None,
    arg_attrs: Optional[ArrayAttr] = None,
    res_attrs: Optional[ArrayAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('len', len))
    all_props.append(('isVolatile', IntegerAttr.make(IntegerType.signless(1), int(isVolatile))))
    if access_groups is not None:
        all_props.append(('access_groups', access_groups))
    if alias_scopes is not None:
        all_props.append(('alias_scopes', alias_scopes))
    if noalias_scopes is not None:
        all_props.append(('noalias_scopes', noalias_scopes))
    if tbaa is not None:
        all_props.append(('tbaa', tbaa))
    if arg_attrs is not None:
        all_props.append(('arg_attrs', arg_attrs))
    if res_attrs is not None:
        all_props.append(('res_attrs', res_attrs))
    return add_operation(
        name="llvm.intr.memset.inline",
        result_type=None,
        operands=[dst, val],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MemsetOp(
    *,
    dst: Value,
    val: Value,
    len: Value,
    isVolatile: bool,
    access_groups: Optional[ArrayAttr] = None,
    alias_scopes: Optional[ArrayAttr] = None,
    noalias_scopes: Optional[ArrayAttr] = None,
    tbaa: Optional[ArrayAttr] = None,
    arg_attrs: Optional[ArrayAttr] = None,
    res_attrs: Optional[ArrayAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('isVolatile', IntegerAttr.make(IntegerType.signless(1), int(isVolatile))))
    if access_groups is not None:
        all_props.append(('access_groups', access_groups))
    if alias_scopes is not None:
        all_props.append(('alias_scopes', alias_scopes))
    if noalias_scopes is not None:
        all_props.append(('noalias_scopes', noalias_scopes))
    if tbaa is not None:
        all_props.append(('tbaa', tbaa))
    if arg_attrs is not None:
        all_props.append(('arg_attrs', arg_attrs))
    if res_attrs is not None:
        all_props.append(('res_attrs', res_attrs))
    return add_operation(
        name="llvm.intr.memset",
        result_type=None,
        operands=[dst, val, len],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MinNumOp(
    *,
    a: Value,
    b: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.minnum",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MinimumOp(
    *,
    a: Value,
    b: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.minimum",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_NearbyintOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.nearbyint",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_NoAliasScopeDeclOp(
    *,
    scope: AliasScopeAttr,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('scope', scope))
    return add_operation(
        name="llvm.intr.experimental.noalias.scope.decl",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_PowIOp(
    *,
    res_type: Type,
    val: Value,
    power: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.powi",
        result_type=res_type,
        operands=[val, power],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_PowOp(
    *,
    a: Value,
    b: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.pow",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Prefetch(
    *,
    addr: Value,
    rw: int,
    hint: int,
    cache: int,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('rw', IntegerAttr.make(IntegerType.signless(32), rw)))
    all_props.append(('hint', IntegerAttr.make(IntegerType.signless(32), hint)))
    all_props.append(('cache', IntegerAttr.make(IntegerType.signless(32), cache)))
    return add_operation(
        name="llvm.intr.prefetch",
        result_type=None,
        operands=[addr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_PtrAnnotation(
    *,
    ptr: Value,
    annotation: Value,
    fileName: Value,
    line: Value,
    attr: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = ptr.type
    all_props = []
    return add_operation(
        name="llvm.intr.ptr.annotation",
        result_type=res_type,
        operands=[ptr, annotation, fileName, line, attr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_PtrMaskOp(
    *,
    ptr: Value,
    mask: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = ptr.type
    all_props = []
    return add_operation(
        name="llvm.intr.ptrmask",
        result_type=res_type,
        operands=[ptr, mask],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_RintOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.rint",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_RoundEvenOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.roundeven",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_RoundOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.round",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SAddSat(
    *,
    a: Value,
    b: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    return add_operation(
        name="llvm.intr.sadd.sat",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SAddWithOverflowOp(
    *,
    res_type: Type,
    operand_0: Value,
    operand_1: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.sadd.with.overflow",
        result_type=res_type,
        operands=[operand_0, operand_1],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SCmpOp(
    *,
    res_type: Type,
    a: Value,
    b: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.scmp",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SMaxOp(
    *,
    a: Value,
    b: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    return add_operation(
        name="llvm.intr.smax",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SMinOp(
    *,
    a: Value,
    b: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    return add_operation(
        name="llvm.intr.smin",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SMulWithOverflowOp(
    *,
    res_type: Type,
    operand_0: Value,
    operand_1: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.smul.with.overflow",
        result_type=res_type,
        operands=[operand_0, operand_1],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SSACopyOp(
    *,
    operand: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = operand.type
    all_props = []
    return add_operation(
        name="llvm.intr.ssa.copy",
        result_type=res_type,
        operands=[operand],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SSHLSat(
    *,
    a: Value,
    b: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    return add_operation(
        name="llvm.intr.sshl.sat",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SSubSat(
    *,
    a: Value,
    b: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    return add_operation(
        name="llvm.intr.ssub.sat",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SSubWithOverflowOp(
    *,
    res_type: Type,
    operand_0: Value,
    operand_1: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.ssub.with.overflow",
        result_type=res_type,
        operands=[operand_0, operand_1],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SinOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.sin",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SincosOp(
    *,
    res_type: Type,
    val: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.sincos",
        result_type=res_type,
        operands=[val],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SinhOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.sinh",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SqrtOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.sqrt",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_StackRestoreOp(
    *,
    ptr: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="llvm.intr.stackrestore",
        result_type=None,
        operands=[ptr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_StackSaveOp(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.stacksave",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_StepVectorOp(
    *,
    res_type: VectorType,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.stepvector",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_StripInvariantGroupOp(
    *,
    ptr: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = ptr.type
    all_props = []
    return add_operation(
        name="llvm.intr.strip.invariant.group",
        result_type=res_type,
        operands=[ptr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_TanOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.tan",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_TanhOp(
    *,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = in_.type
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.tanh",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ThreadlocalAddressOp(
    *,
    res_type: Type,
    global_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.threadlocal.address",
        result_type=res_type,
        operands=[global_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_Trap(
    *,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="llvm.intr.trap",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_UAddSat(
    *,
    a: Value,
    b: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    return add_operation(
        name="llvm.intr.uadd.sat",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_UAddWithOverflowOp(
    *,
    res_type: Type,
    operand_0: Value,
    operand_1: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.uadd.with.overflow",
        result_type=res_type,
        operands=[operand_0, operand_1],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_UBSanTrap(
    *,
    failureKind: int,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('failureKind', IntegerAttr.make(IntegerType.signless(8), failureKind)))
    return add_operation(
        name="llvm.intr.ubsantrap",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_UCmpOp(
    *,
    res_type: Type,
    a: Value,
    b: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.ucmp",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_UMaxOp(
    *,
    a: Value,
    b: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    return add_operation(
        name="llvm.intr.umax",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_UMinOp(
    *,
    a: Value,
    b: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    return add_operation(
        name="llvm.intr.umin",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_UMulWithOverflowOp(
    *,
    res_type: Type,
    operand_0: Value,
    operand_1: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.umul.with.overflow",
        result_type=res_type,
        operands=[operand_0, operand_1],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_USHLSat(
    *,
    a: Value,
    b: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    return add_operation(
        name="llvm.intr.ushl.sat",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_USubSat(
    *,
    a: Value,
    b: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = a.type
    all_props = []
    return add_operation(
        name="llvm.intr.usub.sat",
        result_type=res_type,
        operands=[a, b],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_USubWithOverflowOp(
    *,
    res_type: Type,
    operand_0: Value,
    operand_1: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.usub.with.overflow",
        result_type=res_type,
        operands=[operand_0, operand_1],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPAShrOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.ashr",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPAddOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.add",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPAndOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.and",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPFAddOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.fadd",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPFDivOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.fdiv",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPFMulAddOp(
    *,
    res_type: Type,
    op1: Value,
    op2: Value,
    op3: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.fmuladd",
        result_type=res_type,
        operands=[op1, op2, op3, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPFMulOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.fmul",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPFNegOp(
    *,
    res_type: Type,
    op: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.fneg",
        result_type=res_type,
        operands=[op, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPFPExtOp(
    *,
    res_type: Type,
    src: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.fpext",
        result_type=res_type,
        operands=[src, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPFPToSIOp(
    *,
    res_type: Type,
    src: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.fptosi",
        result_type=res_type,
        operands=[src, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPFPToUIOp(
    *,
    res_type: Type,
    src: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.fptoui",
        result_type=res_type,
        operands=[src, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPFPTruncOp(
    *,
    res_type: Type,
    src: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.fptrunc",
        result_type=res_type,
        operands=[src, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPFRemOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.frem",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPFSubOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.fsub",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPFmaOp(
    *,
    res_type: Type,
    op1: Value,
    op2: Value,
    op3: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.fma",
        result_type=res_type,
        operands=[op1, op2, op3, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPIntToPtrOp(
    *,
    res_type: Type,
    src: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.inttoptr",
        result_type=res_type,
        operands=[src, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPLShrOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.lshr",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPLoadOp(
    *,
    res_type: Type,
    ptr: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.load",
        result_type=res_type,
        operands=[ptr, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPMergeMinOp(
    *,
    res_type: Type,
    cond: Value,
    true_val: Value,
    false_val: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.merge",
        result_type=res_type,
        operands=[cond, true_val, false_val, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPMulOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.mul",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPOrOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.or",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPPtrToIntOp(
    *,
    res_type: Type,
    src: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.ptrtoint",
        result_type=res_type,
        operands=[src, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPReduceAddOp(
    *,
    res_type: Type,
    satrt_value: Value,
    val: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.reduce.add",
        result_type=res_type,
        operands=[satrt_value, val, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPReduceAndOp(
    *,
    res_type: Type,
    satrt_value: Value,
    val: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.reduce.and",
        result_type=res_type,
        operands=[satrt_value, val, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPReduceFAddOp(
    *,
    res_type: Type,
    satrt_value: Value,
    val: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.reduce.fadd",
        result_type=res_type,
        operands=[satrt_value, val, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPReduceFMaxOp(
    *,
    res_type: Type,
    satrt_value: Value,
    val: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.reduce.fmax",
        result_type=res_type,
        operands=[satrt_value, val, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPReduceFMinOp(
    *,
    res_type: Type,
    satrt_value: Value,
    val: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.reduce.fmin",
        result_type=res_type,
        operands=[satrt_value, val, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPReduceFMulOp(
    *,
    res_type: Type,
    satrt_value: Value,
    val: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.reduce.fmul",
        result_type=res_type,
        operands=[satrt_value, val, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPReduceMulOp(
    *,
    res_type: Type,
    satrt_value: Value,
    val: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.reduce.mul",
        result_type=res_type,
        operands=[satrt_value, val, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPReduceOrOp(
    *,
    res_type: Type,
    satrt_value: Value,
    val: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.reduce.or",
        result_type=res_type,
        operands=[satrt_value, val, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPReduceSMaxOp(
    *,
    res_type: Type,
    satrt_value: Value,
    val: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.reduce.smax",
        result_type=res_type,
        operands=[satrt_value, val, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPReduceSMinOp(
    *,
    res_type: Type,
    satrt_value: Value,
    val: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.reduce.smin",
        result_type=res_type,
        operands=[satrt_value, val, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPReduceUMaxOp(
    *,
    res_type: Type,
    satrt_value: Value,
    val: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.reduce.umax",
        result_type=res_type,
        operands=[satrt_value, val, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPReduceUMinOp(
    *,
    res_type: Type,
    satrt_value: Value,
    val: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.reduce.umin",
        result_type=res_type,
        operands=[satrt_value, val, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPReduceXorOp(
    *,
    res_type: Type,
    satrt_value: Value,
    val: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.reduce.xor",
        result_type=res_type,
        operands=[satrt_value, val, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPSDivOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.sdiv",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPSExtOp(
    *,
    res_type: Type,
    src: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.sext",
        result_type=res_type,
        operands=[src, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPSIToFPOp(
    *,
    res_type: Type,
    src: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.sitofp",
        result_type=res_type,
        operands=[src, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPSMaxOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.smax",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPSMinOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.smin",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPSRemOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.srem",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPSelectMinOp(
    *,
    res_type: Type,
    cond: Value,
    true_val: Value,
    false_val: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.select",
        result_type=res_type,
        operands=[cond, true_val, false_val, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPShlOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.shl",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPStoreOp(
    *,
    val: Value,
    ptr: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.store",
        result_type=None,
        operands=[val, ptr, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPStridedLoadOp(
    *,
    res_type: Type,
    ptr: Value,
    stride: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.experimental.vp.strided.load",
        result_type=res_type,
        operands=[ptr, stride, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPStridedStoreOp(
    *,
    val: Value,
    ptr: Value,
    stride: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="llvm.intr.experimental.vp.strided.store",
        result_type=None,
        operands=[val, ptr, stride, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPSubOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.sub",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPTruncOp(
    *,
    res_type: Type,
    src: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.trunc",
        result_type=res_type,
        operands=[src, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPUDivOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.udiv",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPUIToFPOp(
    *,
    res_type: Type,
    src: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.uitofp",
        result_type=res_type,
        operands=[src, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPUMaxOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.umax",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPUMinOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.umin",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPURemOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.urem",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPXorOp(
    *,
    res_type: Type,
    lhs: Value,
    rhs: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.xor",
        result_type=res_type,
        operands=[lhs, rhs, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VPZExtOp(
    *,
    res_type: Type,
    src: Value,
    mask: Value,
    evl: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vp.zext",
        result_type=res_type,
        operands=[src, mask, evl],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VaCopyOp(
    *,
    dest_list: Value,
    src_list: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="llvm.intr.vacopy",
        result_type=None,
        operands=[dest_list, src_list],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VaEndOp(
    *,
    arg_list: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="llvm.intr.vaend",
        result_type=None,
        operands=[arg_list],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VaStartOp(
    *,
    arg_list: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="llvm.intr.vastart",
        result_type=None,
        operands=[arg_list],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_VarAnnotation(
    *,
    val: Value,
    annotation: Value,
    fileName: Value,
    line: Value,
    attr: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="llvm.intr.var.annotation",
        result_type=None,
        operands=[val, annotation, fileName, line, attr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_masked_compressstore(
    *,
    value: Value,
    ptr: Value,
    mask: Value,
    arg_attrs: Optional[ArrayAttr] = None,
    res_attrs: Optional[ArrayAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if arg_attrs is not None:
        all_props.append(('arg_attrs', arg_attrs))
    if res_attrs is not None:
        all_props.append(('res_attrs', res_attrs))
    return add_operation(
        name="llvm.intr.masked.compressstore",
        result_type=None,
        operands=[value, ptr, mask],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_masked_expandload(
    *,
    res_type: Type,
    ptr: Value,
    mask: Value,
    passthru: Value,
    arg_attrs: Optional[ArrayAttr] = None,
    res_attrs: Optional[ArrayAttr] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if arg_attrs is not None:
        all_props.append(('arg_attrs', arg_attrs))
    if res_attrs is not None:
        all_props.append(('res_attrs', res_attrs))
    return add_operation(
        name="llvm.intr.masked.expandload",
        result_type=res_type,
        operands=[ptr, mask, passthru],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_masked_gather(
    *,
    res_type: VectorType,
    ptrs: Value,
    mask: Value,
    pass_thru: Sequence[Value],
    alignment: int,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(32), alignment)))
    return add_operation(
        name="llvm.intr.masked.gather",
        result_type=res_type,
        operands=[ptrs, mask, *pass_thru],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_masked_scatter(
    *,
    value: Value,
    ptrs: Value,
    mask: Value,
    alignment: int,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(32), alignment)))
    return add_operation(
        name="llvm.intr.masked.scatter",
        result_type=None,
        operands=[value, ptrs, mask],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_deinterleave2(
    *,
    res_type: Type,
    vec: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vector.deinterleave2",
        result_type=res_type,
        operands=[vec],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_extract(
    *,
    res_type: VectorType,
    srcvec: Value,
    pos: int,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('pos', IntegerAttr.make(IntegerType.signless(64), pos)))
    return add_operation(
        name="llvm.intr.vector.extract",
        result_type=res_type,
        operands=[srcvec],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_insert(
    *,
    dstvec: Value,
    srcvec: Value,
    pos: int,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    res_type = dstvec.type
    all_props = []
    all_props.append(('pos', IntegerAttr.make(IntegerType.signless(64), pos)))
    return add_operation(
        name="llvm.intr.vector.insert",
        result_type=res_type,
        operands=[dstvec, srcvec],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_interleave2(
    *,
    res_type: Type,
    vec1: Value,
    vec2: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vector.interleave2",
        result_type=res_type,
        operands=[vec1, vec2],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_reduce_add(
    *,
    res_type: Type,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vector.reduce.add",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_reduce_and(
    *,
    res_type: Type,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vector.reduce.and",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_reduce_fadd(
    *,
    res_type: Type,
    start_value: Value,
    input: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.vector.reduce.fadd",
        result_type=res_type,
        operands=[start_value, input],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_reduce_fmax(
    *,
    res_type: Type,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.vector.reduce.fmax",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_reduce_fmaximum(
    *,
    res_type: Type,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.vector.reduce.fmaximum",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_reduce_fmin(
    *,
    res_type: Type,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.vector.reduce.fmin",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_reduce_fminimum(
    *,
    res_type: Type,
    in_: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.vector.reduce.fminimum",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_reduce_fmul(
    *,
    res_type: Type,
    start_value: Value,
    input: Value,
    fastmathFlags: FastmathFlags = FastmathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('fastmathFlags', FastmathFlagsAttr(value=fastmathFlags)))
    return add_operation(
        name="llvm.intr.vector.reduce.fmul",
        result_type=res_type,
        operands=[start_value, input],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_reduce_mul(
    *,
    res_type: Type,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vector.reduce.mul",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_reduce_or(
    *,
    res_type: Type,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vector.reduce.or",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_reduce_smax(
    *,
    res_type: Type,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vector.reduce.smax",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_reduce_smin(
    *,
    res_type: Type,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vector.reduce.smin",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_reduce_umax(
    *,
    res_type: Type,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vector.reduce.umax",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_reduce_umin(
    *,
    res_type: Type,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vector.reduce.umin",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vector_reduce_xor(
    *,
    res_type: Type,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vector.reduce.xor",
        result_type=res_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_vscale(
    *,
    res_type: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="llvm.intr.vscale",
        result_type=res_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )
