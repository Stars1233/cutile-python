# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sys
from distutils import file_util

from setuptools import setup, Command
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build import build
import os


def guess_cuda_tile_build_dir():
    import subprocess
    new_env = dict(os.environ)
    new_env.pop("PYTHONPATH", None)
    path = subprocess.run([sys.executable, "-c", "import cuda.tile; print(cuda.tile.__file__)"],
                          env=new_env,
                          capture_output=True,
                          check=True).stdout.decode().strip()
    suffix = "src/cuda/tile/__init__.py"
    assert path.endswith(suffix), path
    path = path[:-len(suffix)]
    return os.path.join(path, "build")


class BuildBinaries(Command):
    editable_mode = False

    def initialize_options(self):
        self.editable_mode = False

    def finalize_options(self) -> None:
        pass

    def run(self):
        build_dir = os.getenv("CUDA_TILE_CEXT_BUILD_DIR")
        if build_dir is None:
            build_dir = guess_cuda_tile_build_dir()
        self.spawn(["cmake", "--build", build_dir])

        binary_name = "mlir2cubin"
        src_path = os.path.join(build_dir, "internal", "mlir2cubin", binary_name)

        if self.editable_mode:
            bin_dir = os.path.join(self.get_package_dir(), "bin")
        else:
            bin_dir = os.path.join(self.get_finalized_command('build').build_lib,
                                   "cuda", "lang", "bin")

        os.makedirs(bin_dir, exist_ok=True)
        dst_path = os.path.join(bin_dir, binary_name)
        # Create a symlink to the build directory if in editable mode, otherwise copy
        link = "sym" if self.editable_mode else None
        file_util.copy_file(src_path, dst_path, update=1, link=link)

    def get_package_dir(self) -> str:
        package = "cuda.lang"
        build_py = self.get_finalized_command('build_py')
        return os.path.abspath(build_py.get_package_dir(package))


class CustomBuild(build):
    sub_commands = [*build.sub_commands, ("build_binaries", None)]


# Force an "unpure" wheel name even though we are not including any C extensions
class CustomBdistWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False


setup(
    cmdclass=dict(
        build=CustomBuild,
        build_binaries=BuildBinaries,
        bdist_wheel=CustomBdistWheel,
    )
)
