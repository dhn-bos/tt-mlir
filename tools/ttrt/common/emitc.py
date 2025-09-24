# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import importlib.machinery
import sys
import signal
import io
import subprocess
import time
import socket
from pkg_resources import get_distribution
import shutil
import atexit
import traceback
from pathlib import Path
import csv
import ast

from ttrt.common.util import *
from ttrt.common.query import Query


class EmitC:
    registered_args = {}

    @staticmethod
    def initialize_api():
        EmitC.register_arg(
            name="--clean-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="clean all artifacts from previous runs",
        )
        EmitC.register_arg(
            name="--log-file",
            type=str,
            default="",
            choices=None,
            help="log file to dump ttrt output to",
        )
        EmitC.register_arg(
            name="--artifact-dir",
            type=str,
            default=f"{os.getcwd()}/ttrt-artifacts",
            choices=None,
            help="provides a directory path to save artifacts to",
        )
        EmitC.register_arg(
            name="--program-index",
            type=str,
            default="all",
            choices=["all"] + [str(i) for i in range(0, 5)],
            help="the program inside the fbb to run",
        )
        EmitC.register_arg(
            name="--loops",
            type=int,
            default=1,
            choices=None,
            help="number of loops",
        )
        EmitC.register_arg(
            name="--host-only",
            type=bool,
            default=False,
            choices=[True, False],
            help="collect performance trace on host only",
        )
        EmitC.register_arg(
            name="--result-file",
            type=str,
            default="perf_results.json",
            choices=None,
            help="test file to save results to",
        )
        EmitC.register_arg(
            name="--disable-golden",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable golden comparison for intermediate and output tensors",
        )
        EmitC.register_arg(
            name="--memory",
            type=bool,
            default=False,
            choices=[True, False],
            help="dump memory reports after every op execution",
        )
        EmitC.register_arg(
            name="--disable-eth-dispatch",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable putting dispatch on ethernet cores - place it on worker cores instead",
        )
        EmitC.register_arg(
            name="--ignore-version",
            type=bool,
            default=False,
            choices=[True, False],
            help="Ignore check for Major/Minor/Patch between flatbuffer and TTRT, use at your own risk.",
        )
        EmitC.register_arg(
            name="--enable-program-cache",
            type=bool,
            default=False,
            choices=[True, False],
            help="enable program cache in ttnn runtime",
        )
        EmitC.register_arg(
            name="--emitc",
            type=bool,
            default=False,
            choices=[True, False],
            help="toggles emitc testing",
        )
        EmitC.register_arg(
            name="--trace-region-size",
            type=int,
            default=0,
            choices=None,
            help="Device trace region size",
        )
        EmitC.register_arg(
            name="--dump-device-rate",
            type=int,
            default=1000,
            choices=None,
            help="Rate at which to flush device perf information",
        )
        EmitC.register_arg(
            name="--benchmark",
            type=bool,
            default=False,
            choices=[True, False],
            help="Enable benchmark mode with warmup and e2e time measurements (automatically enables program cache)",
        )
        EmitC.register_arg(
            name="binary",
            type=str,
            default="",
            choices=None,
            help="flatbuffer binary file",
        )
        EmitC.register_arg(
            name="--disable-ttrt-callbacks",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable ttrt callbacks",
        )

    def __init__(self, args={}, logger=None, artifacts=None):
        for name, attributes in EmitC.registered_args.items():
            if type(args) == dict:
                if name in args.keys():
                    self[name] = args[name]
                else:
                    self[name] = attributes["default"]
            else:
                # argument got parsed to hyphen's for underscrolls and leading hyphen's removed - need to put back
                converted_name = name
                if name != "binary":
                    converted_name = converted_name.lstrip("-")
                    converted_name = converted_name.replace("-", "_")
                self[name] = getattr(args, converted_name)

        self.logger = logger if logger != None else Logger(self["--log-file"])
        self.logging = self.logger.get_logger()
        self.globals = Globals(self.logger)
        self.file_manager = FileManager(self.logger)
        self.artifacts = (
            artifacts
            if artifacts != None
            else Artifacts(
                self.logger,
                self.file_manager,
                artifacts_folder_path=self["--artifact-dir"],
            )
        )
        self.ttnn_binaries = []
        self.results = Results(self.logger, self.file_manager)

    def preprocess(self):
        self.logging.debug(f"------preprocessing emitc API")

        if self["--clean-artifacts"]:
            self.artifacts.clean_artifacts()

        self.artifacts.create_artifacts()

        self.logging.debug(f"------finished preprocessing emitc API")

    def check_constraints(self):
        self.logging.debug(f"------checking constraints for emitc API")

        # is this potentially an unneccessary duplicate of run.py?
        if not hasattr(self, "binary"):
            # load from Capsule instead. only TTNN Path is supported for now
            bin = Binary(self.logger, self.file_manager, "", self["--capsule"])
            if not bin.check_version(ignore=self["--ignore-version"]):
                self.logger.warning(
                    "Flatbuffer version not present, are you sure that the binary is valid? - Skipped"
                )
                return

            if self["--program-index"] != "all":
                if not bin.check_program_index_exists(int(self["--program-index"])):
                    self.logging.warning(
                        f"program index={int(self['--program-index'])} is greater than number of programs in: {bin.file_path} - skipping this test"
                    )
                    return
            self.ttnn_binaries.append(bin)
        else:
            shared_object_paths = self.file_manager.find_shared_object_paths(
                self["binary"]
            )
            ttnn_binary_paths = self.file_manager.find_ttnn_binary_paths(self["binary"])

            self.logging.debug(f"shared_object_paths={shared_object_paths}")
            self.logging.debug(f"ttnn_binary_paths={ttnn_binary_paths}")

            for path in shared_object_paths:
                so = SharedObject(self.logger, self.file_manager, path)
                try:
                    so.check_version(ignore=self["--ignore-version"])
                except Exception as e:
                    test_result = {
                        "file_path": path,
                        "result": "skip",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                        "program_index": self["--program-index"],
                    }
                    self.logging.warning(
                        f"SKIP: test={path} was skipped with exception={str(e)}"
                    )
                    self.results.add_result(test_result)
                    continue

                self.shared_objects.append(so)
                corresponding_ttnn = self.file_manager.find_corresponding_ttnn(path)

                if corresponding_ttnn:
                    self.ttnn_binary_paths.append(corresponding_ttnn)

            for path in ttnn_binary_paths:
                bin = Binary(self.logger, self.file_manager, path)
                try:
                    bin.check_version(ignore=self["--ignore-version"])
                except Exception as e:
                    test_result = {
                        "file_path": path,
                        "result": "skip",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                        "program_index": self["--program-index"],
                    }
                    self.logging.warning(
                        f"SKIP: test={path} was skipped with exception={str(e)}"
                    )
                    self.results.add_result(test_result)
                    continue

                if self["--program-index"] != "all":
                    if not bin.check_program_index_exists(int(self["--program-index"])):
                        message = f"program index={int(self['--program-index'])} is greater than number of programs in: {bin.file_path} - skipping this test"
                        self.logging.warning(message)
                        test_result = {
                            "file_path": path,
                            "result": "skip",
                            "exception": message,
                            "log_file": self.logger.file_name,
                            "artifacts": self.artifacts.artifacts_folder_path,
                            "program_index": self["--program-index"],
                        }
                        self.logging.warning(
                            f"SKIP: test={path} was skipped with exception={message}"
                        )
                        self.results.add_result(test_result)
                        continue

                self.ttnn_binaries.append(bin)

            self.logging.debug(f"finished checking constraints for emitc API")

        self.logging.debug(f"------finished checking constraints for emitc API")

    def execute(self):
        self.logging.debug(f"------executing emitc API")

        def _execute(shared_objects):
            if len(binaries) == 0:
                self.logging.warning(f"no binaries found to run - returning early")
                return

            # change this: to for .so files in directory, for each, if there is a matching .ttnn file, run
            for so in self.shared_objects:
                try:
                    # .so are compiled such that they have the same name as flatbuffers, so we rename here
                    emitc_dylib_path = (
                        so.file_path
                    )  # ***** #bin.file_path.replace(".ttnn", ".so")

                    compare_to_ttnn = False
                    if so in self.ttnn_binaries:
                        bin = self.ttnn_binaries[so]
                        compare_to_ttnn = True

                    # Open the dylib
                    emitc_dylib_handle = ttrt.runtime.test.open_so(emitc_dylib_path)
                    self.logging.debug(f"opened emitc dylib={emitc_dylib_path}")

                    # Run to EmitC
                    for program_index in program_indices:
                        # Create symbol string to read from dylib
                        fwd_func_name = program.name

                        # pre-upload inputs
                        inputs = convert_input_layouts(
                            device, inputs, bin.fbb, program_index
                        )

                        for loop in range(self["--loops"]):
                            emitc_outs = ttrt.runtime.test.run_so_program(
                                emitc_dylib_handle,
                                fwd_func_name,
                                inputs,
                                device,
                            )

                    ttrt.runtime.test.close_so(emitc_dylib_handle)
                except Exception as e:
                    self.logging.error(f"Error during EmitC execution: {str(e)}")
                    raise e
                try:
                    if compare_to_ttnn:
                        command_options = f"--program-index {self['--program-index']} --loops {self['--loops']} --save-artifacts "

                        if self["--memory"]:
                            command_options += " --memory "

                        if self["--disable-eth-dispatch"]:
                            command_options += " --disable-eth-dispatch "

                        if self["--disable-golden"]:
                            command_options += " --disable-golden "

                        if self["--enable-program-cache"]:
                            command_options += " --enable-program-cache "

                        if self["--dump-device-rate"] != 1000:
                            command_options += (
                                f" --dump-device-rate {self['--dump-device-rate']} "
                            )

                        if self["--benchmark"]:
                            command_options += " --benchmark "

                        if self["--ignore-version"]:
                            command_options += " --ignore-version "

                        if self["--disable-ttrt-callbacks"]:
                            command_options += " --disable-ttrt-callbacks "

                        ttrt_executable_path = shutil.which("ttrt")
                        test_command = f"{ttrt_executable_path} run {bin.file_path} {command_options}"
                        self.logging.info(
                            f"test command for binary={bin.file_path} is: {test_command}"
                        )
                        testProcess = subprocess.Popen(
                            [test_command],
                            shell=True,
                            env=env_vars,
                            preexec_fn=os.setsid,
                        )

                        fbb_output_tensors = self.load_output_tensors_from_artifacts(
                            bin
                        )

                        # post-process test results
                        test_result = []
                        with open("run_results.json", "r") as file:
                            test_result = json.load(file)

                        for result in test_result:
                            if result["result"] != "pass":
                                if result["result"] == "test_error":
                                    raise TTRTTestException(str(result["exception"]))
                                raise Exception(f'{result["exception"]}')

                        if compare_to_ttnn:
                            emitc_outs = [
                                ttrt.runtime.to_host(emitc_out, untilize=True)[0]
                                for emitc_out in emitc_outs
                            ]
                            self.logging.debug(
                                f"got emitc outputs for program_index={program_index}, loop={loop}"
                            )

                            all_tensors_match = ttrt.runtime.test.compare_outs(
                                outputs, emitc_outs
                            )

                            if not all_tensors_match:
                                self.logging.error(
                                    "Failed: TTRT and EmitC outputs do not match! program_index={program_index}, loop={loop}"
                                )
                                self.logging.error(outputs, emitc_outs)
                                raise Exception(
                                    "Failed: TTRT and EmitC outputs do not match! program_index={program_index}, loop={loop}"
                                )
                            self.logging.info(
                                f"EmitC tensors match for {bin.file_path}"
                            )
                except Exception as e:
                    result = "error"
                    if isinstance(e, TTRTTestException):
                        result = "test_error"
                    test_result = {
                        "file_path": so.file_path,
                        "result": result,
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                        "program_index": self["--program-index"],
                    }
                    self.logging.error(
                        f"ERROR: test={so.file_path} experienced an error with exception={str(e)}"
                    )
                    self.results.add_result(test_result)
                    so.test_result = result
                    traceback.print_exc()
                    continue

        self.logging.debug(f"executing shared_objects")
        _execute(self.shared_objects)
        self.logging.debug(f"finished executing shared_objects")

        self.logging.debug(f"------finished executing emitc API")

    def postprocess(self):
        self.logging.debug(f"------postprocessing emitc API")

        for so in self.shared_objects:
            if so.test_result == "pass":
                test_result = {
                    "file_path": bin.file_path,
                    "result": "pass",
                    "exception": "",
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                    "program_index": self["--program-index"],
                    "program_results": so.program_results,
                }
                self.results.add_result(test_result)
                self.logging.info(f"PASS: test case={so.file_path}")
            else:
                self.logging.error(f"ERROR: test case={so.file_path}")

        self.results.save_results(self["--result-file"])

        self.logging.debug(f"------finished postprocessing emitc API")

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __call__(self):
        self.logging.debug(
            f"----------------------------starting emitc API----------------------------"
        )

        self.preprocess()
        self.check_constraints()
        self.execute()
        self.postprocess()

        self.logging.debug(
            f"----------------------------finished emitc API----------------------------"
        )

        return self.results.get_result_code(), self.results.get_results()

    @staticmethod
    def register_arg(name, type, default, choices, help):
        EmitC.registered_args[name] = {
            "type": type,
            "default": default,
            "choices": choices,
            "help": help,
        }

    @staticmethod
    def generate_subparser(subparsers):
        emitc_parser = subparsers.add_parser(
            "emitc", help="run performance trace and collect performance data"
        )
        emitc_parser.set_defaults(api=EmitC)

        for name, attributes in EmitC.registered_args.items():
            if name == "binary":
                emitc_parser.add_argument(f"{name}", help=attributes["help"])
            elif attributes["type"] == bool:
                emitc_parser.add_argument(
                    f"{name}",
                    action="store_true",
                    help=attributes["help"],
                )
            else:
                emitc_parser.add_argument(
                    f"{name}",
                    type=attributes["type"],
                    default=attributes["default"],
                    choices=attributes["choices"],
                    help=attributes["help"],
                )

        return emitc_parser


def load_output_tensors_from_artifacts(self, bin):
    """
    Open directory, loop through subdirectories, load all .pt files into torch tensors, and save them according to their respective program.
    """
    fbb_run_directory = self.artifacts.get_binary_run_folder_path(bin)
    program_tensors = {}
    saved_tensors = {}

    self.logging.debug(f"Loading .pt tensors from directory: {fbb_run_directory}")

    for root, dirs, files in os.walk(fbb_run_directory):

        # second for loop may be unnecessary
        for directory in dirs:
            tensors = []
            for pt_file in directory:
                if pt_file.endswith(".pt") and "output" in pt_file:

                    try:
                        self.logging.debug(
                            f"Loading output tensor from file: {pt_file}"
                        )
                        tensors.append(torch.load(pt_file))

                    except Exception as e:
                        self.logging.error(
                            f"Error loading .pt file {pt_file}: {str(e)}"
                        )
                        continue

            program_tensors[directory] = tensors

    return program_tensors
