# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import enum
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar, Generic

from cuda.tile._exception import Loc, TileSyntaxError
from cuda.tile._ir import hir
from cuda.tile._ir.hir import ResolvedName
from cuda.tile._ir.ir import Operation, Var, IRContext
from cuda.tile._ir.type import InvalidType


@dataclass
class JumpInfo:
    jump_op: Operation | None
    outputs: tuple[Var, ...]


@dataclass
class ControlFlowInfo:
    stored_locals: tuple[int, ...]
    flatten: bool = False
    jumps: list[JumpInfo] = dataclasses.field(default_factory=list)


class LocalScope:
    def __init__(self, local_names: tuple[str, ...], ir_ctx: IRContext):
        self._local_names = local_names
        self._ir_ctx = ir_ctx
        self._map: list[Var | None] = [None] * len(local_names)
        self.frozen = False
        self._dead = False

    @staticmethod
    def create_frozen(local_names: tuple[str, ...],
                      frozen_indices: tuple[int, ...],
                      frozen_vars: tuple[Var, ...],
                      ir_ctx: IRContext):
        ret = LocalScope(local_names, ir_ctx)
        for idx, var in zip(frozen_indices, frozen_vars, strict=True):
            ret._map[idx] = var
        ret.frozen = True
        return ret

    def mark_dead(self):
        self._dead = True

    def redefine(self, index: int, loc: Loc) -> Var:
        assert not self._dead
        assert not self.frozen
        assert index >= 0
        var = self._ir_ctx.make_var(self._local_names[index], loc)
        self._map[index] = var
        return var

    def __getitem__(self, index: int) -> Var:
        assert not self._dead
        assert index >= 0
        var = self._map[index]
        if var is None:
            raise TileSyntaxError(f"Undefined variable {self._local_names[index]} used")
        return var

    def get(self, index: int, loc: Loc):
        assert not self._dead
        assert index >= 0
        var = self._map[index]
        if var is None:
            name = self._local_names[index]
            var = self._ir_ctx.make_var(name, loc)
            var.set_type(InvalidType(f"Use of potentially undefined variable `{name}`", loc=loc))
        return var

    @contextmanager
    def enter_branch(self):
        assert not self._dead
        old = self._map
        self._map = list(old)
        try:
            yield
        finally:
            self._map = old


class _CurrentScope(threading.local):
    scope = None


_current_scope = _CurrentScope()


class _MissingItem(enum.IntEnum):
    INSTANCE = 0


V = TypeVar("V")


class IntMap(Generic[V]):
    def __init__(self):
        self._items = []

    def __getitem__(self, idx: int):
        assert isinstance(idx, int)
        assert idx >= 0
        try:
            val = self._items[idx]
        except IndexError:
            raise KeyError()
        if val is _MissingItem:
            raise KeyError()
        return val

    def __setitem__(self, idx, value):
        assert isinstance(idx, int)
        assert idx >= 0
        size = len(self._items)
        if idx < size:
            self._items[idx] = value
        else:
            if idx > size:
                self._items.extend((_MissingItem.INSTANCE,) * (idx - size))
            self._items.append(value)


@dataclass(eq=False)
class Scope:
    local_scopes: tuple[LocalScope, ...]
    loop_info: ControlFlowInfo | None
    if_else_info: ControlFlowInfo | None
    call_site: Loc | None
    hir2ir_varmap: IntMap[Var]
    func_hir: hir.Function

    def get_local_index(self, name: str) -> int:
        rn: ResolvedName = self.func_hir.used_names[name]
        assert rn.depth == len(self.local_scopes) - 1
        return rn.index

    @property
    def local(self) -> LocalScope:
        return self.local_scopes[-1]

    @property
    def depth(self) -> int:
        return len(self.local_scopes) - 1

    @contextmanager
    def make_current(self):
        old = _current_scope.scope
        _current_scope.scope = self
        try:
            yield
        finally:
            _current_scope.scope = old

    @staticmethod
    def get_current() -> "Scope | None":
        return _current_scope.scope

    @contextmanager
    def change_loop_info(self, new: ControlFlowInfo):
        old = self.loop_info
        self.loop_info = new
        try:
            yield
        finally:
            self.loop_info = old

    @contextmanager
    def change_if_else_info(self, new: ControlFlowInfo):
        old = self.if_else_info
        try:
            self.if_else_info = new
            yield
        finally:
            self.if_else_info = old
