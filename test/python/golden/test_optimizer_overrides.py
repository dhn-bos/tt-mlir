# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional
import re

from ttmlir import optimizer_overrides
from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer, _is_opmodel_enabled
from builder.base.builder_test_utils import *
import os


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (16, 32, 32, 64),
            (64, 64, 3, 3),
            (1, 1, 1, 64),
            (1, 1, 1, 64),
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 4])
@pytest.mark.parametrize(
    "stride,padding,dilation,groups", [([1, 1], [1, 1], [1, 1], 1)]
)
def test_conv2d_sharding(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    request,
):
    def conv2d(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        conv2d_0 = builder.conv2d(
            in0,
            weight,
            bias,
            in1,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=unit_attrs,
        )
        builder.set_conv2d_config_override()
        return conv2d_0

    output_file_mlir = compile_ttir_to_flatbuffer(
        conv2d,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
    if _is_opmodel_enabled():
        assert check_sharded_input_output(output_file_mlir, "conv2d")


@pytest.mark.subprocess
@pytest.mark.parametrize(
    "shapes",
    [(1, 32, 32, 64)],
)
@pytest.mark.parametrize("dtypes", [torch.float32], ids=["f32"])
@pytest.mark.parametrize(
    "optimization_policy",
    [optimizer_overrides.MemoryLayoutAnalysisPolicyType.DFSharding],
)
def test_optimization_policies(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    optimization_policy: optimizer_overrides.MemoryLayoutAnalysisPolicyType,
    request,
):
    def model(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
    ):
        return builder.add(in0, in0)

    output_file_mlir = compile_ttir_to_flatbuffer(
        model,
        [shapes, shapes],
        [dtypes, dtypes],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        optimization_policy=optimization_policy,
    )
    check_overrides_policy(output_file_mlir, optimization_policy)


@pytest.mark.subprocess
@pytest.mark.parametrize(
    "shapes",
    [
        (
            32,
            32,
        )
    ],
)
@pytest.mark.parametrize("dtypes", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("configs", [{"buffer_type": "l1"}])
def test_output_layouts(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    configs: dict,
    request,
):
    def model(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
    ):
        add_0 = builder.add(in0, in0)
        sub_0 = builder.multiply(in0, in0)
        builder.set_output_layout_override(configs, sub_0)
        return sub_0

    output_file_mlir = compile_ttir_to_flatbuffer(
        model,
        [shapes, shapes],
        [dtypes, dtypes],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
    check_output_layouts(
        output_file_mlir,
        "multiply",
        configs,
    )
