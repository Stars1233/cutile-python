# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import typing
from dataclasses import dataclass
from types import FunctionType
from typing import (get_origin, get_args, Annotated, Any, Sequence)

from cuda.tile._stub import ConstantAnnotation, ArrayAnnotation, ScalarAnnotation, ListAnnotation
from cuda.tile._datatype import int64


@dataclass(frozen=True)
class LeafAnnotationNode:
    KIND = "leaf"

    constant: bool                 # ct.Constant: compile-time constant parameter.
    int64_index: bool              # 64-bit array index dtype.
    int64_scalar: bool             # int64 scalar parameter.
    static_shape: tuple[int, ...]  # array shape dims specialized to launch-time values.


@dataclass(frozen=True)
class HomogeneousTupleNode:
    KIND = "homogeneous_tuple"

    each: "ParameterAnnotationNode"


@dataclass(frozen=True)
class HeterogeneousTupleNode:
    KIND = "heterogeneous_tuple"

    items: tuple["ParameterAnnotationNode", ...]


ParameterAnnotationNode = LeafAnnotationNode | HomogeneousTupleNode | HeterogeneousTupleNode


@dataclass
class AnnotatedFunction:
    pyfunc: FunctionType
    pysig: inspect.Signature
    parameter_annotations: Sequence[ParameterAnnotationNode]


def get_annotated_function(pyfunc: FunctionType) -> AnnotatedFunction:
    sig = inspect.signature(pyfunc)
    # Resolves string annotations produced by `from __future__ import annotations`.
    hints = typing.get_type_hints(pyfunc, include_extras=True)
    annotations = [hints.get(name, param.annotation) for name, param in sig.parameters.items()]
    parameter_annotations = tuple(_build_annotation_node(ann) for ann in annotations)
    return AnnotatedFunction(pyfunc=pyfunc,
                             pysig=sig,
                             parameter_annotations=parameter_annotations)


def _build_tuple_node(annotation: Any, outer_constant: bool) -> ParameterAnnotationNode:
    args = get_args(annotation)
    if len(args) == 2 and args[1] is ...:
        return HomogeneousTupleNode(_build_annotation_node(args[0], outer_constant))
    return HeterogeneousTupleNode(
        tuple(_build_annotation_node(arg, outer_constant) for arg in args))


def _build_annotation_node(annotation: Any,
                           outer_constant: bool = False) -> ParameterAnnotationNode:
    if get_origin(annotation) is Annotated:
        inner, *metadata = get_args(annotation)
        is_constant = outer_constant or any(isinstance(m, ConstantAnnotation) for m in metadata)
        if get_origin(inner) is tuple:
            return _build_tuple_node(inner, is_constant)
        return LeafAnnotationNode(constant=is_constant,
                                  int64_index=_has_int64_index_annotation(metadata),
                                  int64_scalar=_has_int64_scalar_annotation(metadata),
                                  static_shape=_get_static_shape_annotation(metadata))
    if get_origin(annotation) is tuple:
        return _build_tuple_node(annotation, outer_constant)
    return LeafAnnotationNode(constant=outer_constant, int64_index=False,
                              int64_scalar=False, static_shape=())


def _has_int64_index_annotation(metadata: Sequence[Any]) -> bool:
    for m in metadata:
        if isinstance(m, ArrayAnnotation) and m.index_dtype is int64:
            return True
        if (isinstance(m, ListAnnotation)
                and isinstance(m.element, ArrayAnnotation)
                and m.element.index_dtype is int64):
            return True
    return False


def _has_int64_scalar_annotation(metadata: Sequence[Any]) -> bool:
    return any(isinstance(m, ScalarAnnotation) and m.dtype is int64 for m in metadata)


def _get_static_shape_annotation(metadata: Sequence[Any]) -> tuple[int, ...]:
    for m in metadata:
        if isinstance(m, ArrayAnnotation) and m.static_shape_dims:
            return m.static_shape_dims
        if (isinstance(m, ListAnnotation)
                and isinstance(m.element, ArrayAnnotation)
                and m.element.static_shape_dims):
            return m.element.static_shape_dims
    return ()
