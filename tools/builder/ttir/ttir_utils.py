# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import inspect
import subprocess
import torch
import pytest
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict

from ttmlir.dialects import func
from ttmlir.ir import *
from ttmlir.passmanager import PassManager
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
from builder.ttir.ttir_builder import TTIRBuilder

# ----- Private APIs -----


def _get_target_path(output_path, filename, target):
    target_dir = os.path.join(output_path, target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return os.path.join(target_dir, filename)


def _emitc_to_executable(module, filepath: str, golden_map, module_cache):
    cpp = translate_to_cpp(module)
    with open(filepath, "w") as f:
        f.write(cpp)


# ----- Public APIs -----


def create_custom_ttir_pipeline_fn(
    pipeline: str, verify: bool = True, print_ir: Union[bool, str] = False
) -> Callable:
    """
    Creates a custom pipeline function.

    Parameters
    ----------
    pipeline : str
        Pipeline string specification
    verify : bool, optional
        Whether to enable verification (default: True)
    print_ir : *Union[bool, str]*, optional
        If True or a path string, enables IR printing (default: False)

    Returns
    -------
    Callable
        Function that runs the custom pipeline on a module
    """

    def wrapper(module, device_register_options):
        register_device = "ttcore-register-device"
        if device_register_options:
            register_device = f"{register_device}{{{device_register_options}}}"

        pipeline_str = f"builtin.module({','.join([register_device, pipeline])})"
        with module.context:
            pm = PassManager.parse(pipeline_str)
            pm.enable_verifier(verify)
            print("Running custom pipeline:", pm)
            if print_ir:
                print_ir_path = print_ir if isinstance(print_ir, str) else None
                pm.enable_ir_printing(tree_printing_dir_path=print_ir_path)
            pm.run(module.operation)

    return wrapper


def build_ttir_module(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    mesh_shape: Optional[Tuple[int, int]] = None,
    module_dump: bool = False,
    base: Optional[str] = None,
    output_root: str = ".",
):
    """
    Define a MLIR module specified as a python function.

    It will wrap `fn` in a MLIR FuncOp and then wrap that in a MLIR
    module, and finally tie arguments of that FuncOp to test function inputs. It will
    also pass a `TTIRBuilder` object as the last argument of test function.

    Parameters
    ----------
    fn : Callable
        Python function to be converted to MLIR

    inputs_shapes : *List[Shape]*
        Shapes of the respective ranked tensor inputs of the test function.

    inputs_types: *Optional[List[Union[torch.dtype, TypeInfo]]]*
        Data types of the input tensors

    mesh_shape: *Optional[Tuple[int, int]]*
        A list that contains shape of the mesh to be applied on ttir to ttnn
        conversion path.

    module_dump : bool
        Set to True to print out generated MLIR module.

    golden_dump : bool
        Set to True to dump golden info to flatbuffer file.

    base : *Optional[str]*
        Output file name

    output_root: str = ".",
        Output file path

    Returns
    -------
    Module
        MLIR module containing MLIR op graph defined by `fn`

    Example
    -------
    >>> def test_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
    ...     return builder.add(in0, in1)
    ...
    >>> build_ttir_module(test_add, ((32, 32), (32, 32)))

    This returns:

    .. code-block:: mlir

        #any = #ttcore.operand_constraint<...>
        module {
            func.func @test_add(
                %arg0: tensor<32x32xf32>,
                %arg1: tensor<32x32xf32>
            ) -> tensor<32x32xf32> {
                %0 = ttir.empty() : tensor<32x32xf32>
                %1 = "ttir.add"(%arg0, %arg1, %0) ...
                return %1 : tensor<32x32xf32>
            }
        }

    Check out:
    https://github.com/llvm/llvm-project/blob/main/mlir/test/python/dialects/tensor.py
    """

    ctx = Context()

    # Grab the location of the test function in python for later debugging
    try:
        fname = inspect.getfile(fn)
        line_no = inspect.getsourcelines(fn)[1]
        loc = Location.file(fname, line_no, 0, ctx)
    except (OSError, TypeError):
        loc = Location.unknown(ctx)

    # Instantiate builder which is passed as the last argument to
    # `fn` so the user can use it to build ops.
    ttir_builder = TTIRBuilder(ctx, loc, mesh_shape)

    # Default to all f32s
    if inputs_types is None:
        inputs_types = [torch.float32] * len(inputs_shapes)

    if len(inputs_shapes) != len(inputs_types):
        raise ValueError(
            f"inputs_shapes and inputs_types must have the same length: "
            f"{len(inputs_shapes)} != {len(inputs_types)}"
        )

    with ctx, loc:
        fn_input_types = [
            ttir_builder._create_ranked_tensor_type(
                shape,
                ttir_builder._get_type_from_torch_dtype(
                    dtype if isinstance(dtype, torch.dtype) else dtype
                ),
            )
            for (shape, dtype) in zip(inputs_shapes, inputs_types)
        ]

        # Wrap everything in a mlir module.
        module = Module.create()
        with InsertionPoint(module.body):
            # Wrap everything in a mlir function.
            @func.func(*fn_input_types, name=fn.__name__)
            def decorated_func(*inputs):
                # Randomly generate golden tensors for function inputs.
                input_goldens = []
                for index, (operand, dtype) in enumerate(zip(inputs, inputs_types)):
                    input_goldens.append(
                        ttir_builder._generate_input_golden(
                            operand, dtype, index
                        ).tensor
                    )
                result = fn(*inputs, ttir_builder)
                output_ops = result if hasattr(result, "__iter__") else (result,)
                output_goldens = [
                    ttir_builder._get_golden_tensor(op) for op in output_ops
                ]
                ttir_builder.set_graph_input_output(input_goldens, output_goldens)
                return result

        print(f"`{fn.__name__}` sucessfully transformed into a MLIR module.")

        base = fn.__name__ if base is None else base

        filename = _get_target_path(output_root, base + "_ttir.mlir", "ttir")

        if module_dump:
            with open(filename, "w") as f:
                f.write(module.operation.get_asm(enable_debug_info=True))
                print(module.operation.get_asm(enable_debug_info=True))

        return module, ttir_builder


def run_ttir_pipeline(
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

    print(module)
    print(pipeline_options)
    # Now, pass it through the pipeline. Module gets modified in place.
    pipeline_fn(module, " ".join(pipeline_options))

    # Optionally dump to file.
    if dump_to_file:
        with open(output_file_name, "w") as f:
            f.write(str(module))

    return module
