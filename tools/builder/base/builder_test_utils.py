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
                print(pattern, line)
                if pattern.search(line):
                    print("HERE444")
                    return True

    return False


def check_overrides_policy(
    mlir_file: str, policy: optimizer_overrides.MemoryLayoutAnalysisPolicyType
):
    if policy == optimizer_overrides.MemoryLayoutAnalysisPolicyType.BFInterleaved:
        # BFInterleaved policy uses L1 memory layout, and is the only non-default policy supported
        memory_layout = "l1"
    else:
        # Policy defaults to DRAM
        memory_layout = "dram"
    layouts = []
    with open(mlir_file, "r") as f:
        for line in f:
            if line.startswith("#ttnn_layout"):
                if memory_layout in line:
                    layout = line.split("=", 1)[0].strip()
                    layouts.append(layout)
            if "return" in line and len(layouts) > 0:
                substrs = re.split(r"(?=#)", line)[1:]  # Skip first empty part
                return_layouts = ["#" + substr.split(">")[0] for substr in substrs]
                for layout in return_layouts:
                    if layout not in layouts:  # if each return tensor is in layouts
                        assert (
                            layout not in layouts
                        ), f"Return {layout} doesn't use {memory_layout} memory layout"


def check_output_layouts(mlir_file: str, op_name: str, configs: dict):
    output_layout_override = optimizer_overrides.OutputLayoutOverrideParams()
    layouts = []
    items_not_found = []

    with open(mlir_file, "r") as f:
        keys_in_fb = False
        for key, value in configs.items():
            if not hasattr(output_layout_override, key):
                raise ValueError(f"Invalid override attribute: {key}")
        for line in f:
            if line.startswith("#ttnn_layout"):
                key_in_fb2 = True
                for value in configs.values():
                    if value not in line:
                        key_in_fb2 = False
                        break
                if key_in_fb2:
                    layout = line.split("=", 1)[0].strip()
                    layouts.append(layout)

            if len(layouts) > 0:
                pattern = re.compile(rf".*{op_name}.*->.*({'|'.join(layouts)}).*")
                if pattern.search(line):
                    keys_in_fb = True
                    break

        assert (
            keys_in_fb
        ), f"'{configs}' not found in the output layout for op '{op_name}'"
