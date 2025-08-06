# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import inspect
import subprocess
import torch
import pytest
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict
from collections import OrderedDict

from ttmlir.compile_and_run import stablehlo_to_ttir
from ttmlir.dialects import func, sdy
from ttmlir.passes import (
    tt_populate_argument_types,
    ttir_to_ttnn_backend_pipeline,
    ttnn_to_flatbuffer_file,
    ttir_to_ttmetal_backend_pipeline,
    ttmetal_to_flatbuffer_file,
    translate_to_cpp,
    MLIRModuleLogger,
)

from builder.base.builder import *
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.stablehlo.stablehlo_utils import build_stablehlo_module

# ----- Private APIs -----


def _get_target_path(output_path, filename, target):
    target_dir = os.path.join(output_path, target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return os.path.join(target_dir, filename)


def run_pipeline(
    module,
    pipeline_fn: Callable = ttir_to_ttnn_backend_pipeline,
    pipeline_options: Optional[List[str]] = None,
    dump_to_file: bool = True,
    output_file_name: str = "test.mlir",
    system_desc_path: Optional[str] = None,
    mesh_shape: Optional[Tuple[int, int]] = None,
    argument_types_string: Optional[str] = None,
):
    """
    Runs a pipeline over a module and optionally dumps to file.

    Arguments
    ---------
    module :
        TTIR module on which pipeline is run

    pipeline_fn : Callable
        Pipeline function to run. pipeline_fn(module, options)

    pipeline_options : *Optional[List[str]]*
        Pipeline options to be added to the pass

    dump_to_file : bool
        Flag which indicates that generated TTNN module will be dumped to file.

    output_file_name : str
        Name of the output file.

    mesh_shape : *Optional[Tuple[int, int]]*
        A list that contains shape of the mesh to be applied on ttir to ttnn
        conversion path.

    argument_types_string : *Optional[str]*

    Returns
    -------
    MLIR module containing MLIR op graph defined by `module` and pipeline_fn.
    """

    if pipeline_options is None:
        pipeline_options = []

    if argument_types_string:
        tt_populate_argument_types(module, argument_types_string)

    # Default to the `SYSTEM_DESC_PATH` envvar
    if system_desc_path is None:
        system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")

    # Generate option string
    if system_desc_path:
        pipeline_options.append(f"system-desc-path={system_desc_path}")
    if mesh_shape and len(mesh_shape) == 2:
        pipeline_options.append(f"mesh-shape={mesh_shape[0]},{mesh_shape[1]}")
    if argument_types_string:
        pipeline_options.append("enable-const-eval=true")

    # Now, pass it through the pipeline. Module gets modified in place.
    pipeline_fn(module, " ".join(pipeline_options))

    # Optionally dump to file.
    if dump_to_file:
        with open(output_file_name, "w") as f:
            f.write(str(module))

    return module


def compile_ttir_to_flatbuffer(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    dialect: Literal["ttir", "stablehlo"] = "ttir",
    target: Literal["ttnn", "ttmetal", "ttnn-standalone"] = "ttnn",
    mesh_shape: Optional[Tuple[int, int]] = None,
    module_dump: bool = True,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: Optional[List[str]] = None,
    print_ir: Union[bool, str] = False,
):
    """
    Compiles a TTIRBuilder function `fn` to TTIR MLIR -> TT{Metal,NN} MLIR -> Flatbuffer.

    This decorator is mainly a wrapper around the following functions, with
    each next function called on the output of the last:

    1. `build_ttir_module`
    2. `run_pipeline`
    3. `to_target`

    The choice of TTNN vs. TTMetal is controlled by the `target` parameter.

    Parameters
    ----------
    fn : Callable
        The TTIRBuilder function to compile. Must take `builder : TTIRBuilder` as a kwarg.

    inputs_shapes : *List[Shape]*
        Shapes of the respective ranked tensor inputs of the test function.

    inputs_types : *Optional[List[torch.dtype]]*, optional
        The dtypes to use for the inputs to `fn`. Note that if supplied,
        `len(inputs_shapes) == len(inputs_types)` must be true.
        Default is None.

    test_base : str
        The string to be used as the base name for dumped files throughout the
        process. If `None` is provided, then the `__name__` of `fn` will be used.

    output_root : str
        The path to dump all generated arguments under. If this path doesn't
        exist, it will be created.

    target : *Literal["ttnn", "ttmetal", "ttnn-standalone"]*
        Either "ttnn" or "ttmetal". This controls which backend to use.

    argument_types_string : *Optional[str]*

    custom_pipeline : *Union[Callable, str]*, optional
        Pipeline function to run.
        Can be either:

        - A Callable: custom_pipeline(module, options)
        - A str: "ttir-lower-to-layout,ttir-bufferization-pipeline"

    mesh_shape : *Optional[Tuple[int, int]]*, optional
        A list that contains shape of the mesh to be applied on ttir to ttnn
        conversion path.
        Default is None.

    module_dump : bool
        Set to True to print out generated TTIR MLIR module.
        Default is False.

    pipeline_options : *Optional[List[str]]*
        Pipeline options to be added to the pass

    print_ir : *Union[bool, str]*, optional
        Set to True to print IR to stdout. Set to dir path to print IR after
        each pass to its own file under that directory.
        Default is False.

    Returns
    -------
    str
        The path to the generated TT{Metal,NN} MLIR file.
    """

    if inputs_types is not None:
        if len(inputs_shapes) != len(inputs_types):
            raise ValueError("inputs_shapes and inputs_types must have the same length")

    if type(custom_pipeline) is str:
        custom_pipeline = create_custom_ttir_pipeline_fn(
            custom_pipeline, print_ir=print_ir
        )

    if pipeline_options is None:
        pipeline_options = []

    pipeline_fn: Callable
    to_target: Callable
    mlir_suffix: str
    target_extension: str

    if target == "ttnn":
        pipeline_fn = (
            custom_pipeline if custom_pipeline else ttir_to_ttnn_backend_pipeline
        )
        to_target = ttnn_to_flatbuffer_file
        mlir_suffix = "_ttnn.mlir"
        target_extension = "ttnn"
    elif target == "ttmetal":
        pipeline_fn = (
            custom_pipeline if custom_pipeline else ttir_to_ttmetal_backend_pipeline
        )
        to_target = ttmetal_to_flatbuffer_file
        mlir_suffix = "_ttm.mlir"
        target_extension = "ttm"
    elif target == "ttnn-standalone":
        ttir_to_ttnn_emitc_pipeline = create_custom_ttir_pipeline_fn(
            "ttir-to-emitc-pipeline", print_ir=print_ir
        )
        pipeline_fn = (
            custom_pipeline if custom_pipeline else ttir_to_ttnn_emitc_pipeline
        )
        to_target = _emitc_to_executable
        mlir_suffix = "_ttnn.mlir"
        target_extension = "cpp"
    else:
        raise ValueError("Unsupported target: " + target)

    if dialect == "ttir":
        # Compile model to TTIR MLIR
        module, builder = build_ttir_module(
            fn,
            inputs_shapes,
            inputs_types,
            mesh_shape=mesh_shape,
            module_dump=module_dump,
            output_root=output_root,
            dialect=dialect,
        )
    elif dialect == "stablehlo":
        # Compile model to StableHLO and run stablehlo pipeline to TTIR MLIR
        module, builder = build_stablehlo_module(
            fn,
            inputs_shapes,
            inputs_types,
            module_dump=module_dump,
            output_root=output_root,
        )
        module = stablehlo_to_ttir(module)
        print(module)
        builder.populate_goldens()

    output_file_mlir = _get_target_path(output_root, test_base + mlir_suffix, target)
    output_file_fbb = ".".join([output_file_mlir, target_extension])

    # Compile TTIR MLIR -> TT{Metal,NN} MLIR
    module = run_pipeline(
        module,
        pipeline_fn,
        pipeline_options=pipeline_options,
        dump_to_file=module_dump,
        output_file_name=output_file_mlir,
        system_desc_path=system_desc_path,
        mesh_shape=mesh_shape,
        argument_types_string=argument_types_string,
    )
    print(f"{target} pipeline ran successfully.")

    module_logger = MLIRModuleLogger()
    module_logger.attach_context(module.context)

    # Compile TT{Metal,NN} MLIR -> flatbuffer
    to_target(
        module,
        output_file_fbb,
        builder.golden_map,
        module_logger.module_log if module_logger.module_log else [],
    )
    print(f"{target} flatbuffer created successfully at: {output_file_fbb}")
    return output_file_mlir
