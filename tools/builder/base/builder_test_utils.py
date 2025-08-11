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

# from builder.base.builder_utils import compile_ttir_to_flatbuffer, _is_opmodel_enabled
import os


def check_sharded_input_output(mlir_file: str, op_name: str):
    sharded_layouts = []
    with open(mlir_file, "r") as f:
        for line in f:
            if line.startswith("#ttnn_layout") and "sharded" in line:
                layout = line.split("=", 1)[0].strip()
                sharded_layouts.append(layout)

            if len(sharded_layouts) > 0:
                pattern = re.compile(
                    rf".*{op_name}.*({'|'.join(sharded_layouts)}).*->.*({'|'.join(sharded_layouts)}).*"
                )
                if pattern.search(line):
                    return True
    return False


# NOT Scalable RN
def check_overrides_policy(mlir_file: str):
    l1 = False
    layout1 = False
    with open(mlir_file, "r") as f:
        for line in f:
            if line.startswith("#l1"):
                l1 = True

            if line.startswith("#ttnn_layout1"):
                layout1 = True
    assert l1, "L1 buffer type not found in the MLIR file"
    assert layout1, "TTNN layout1 not found in the MLIR file"


def check_output_layouts(mlir_file: str, config: dict = {}):
    l1 = False
    layout1 = False
    with open(mlir_file, "r") as f:
        for override, value in config.items():
            if override == "buffer_type" and value == "l1":
                l1 = True
            if override == "layout" and value == "layout1":
                layout1 = True
        for line in f:
            if line.startswith("#l1"):
                l1 = True

            if line.startswith("#ttnn_layout1"):
                layout1 = True
    assert l1, "L1 buffer type not found in the MLIR file"
    assert layout1, "TTNN layout1 not found in the MLIR file"
