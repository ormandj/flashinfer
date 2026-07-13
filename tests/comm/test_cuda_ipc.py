# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys
from io import StringIO
from pathlib import Path
from unittest import mock

import pytest


def _load_cuda_ipc_module():
    module_path = Path(__file__).resolve().parents[2] / "flashinfer/comm/cuda_ipc.py"
    spec = importlib.util.spec_from_file_location("_cuda_ipc_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cuda_ipc = _load_cuda_ipc_module()


@pytest.mark.parametrize(
    ("maps", "expected"),
    [
        (
            "7f00-7f01 r-xp 0 00:00 0 /usr/local/cuda/lib64/libcudart.so",
            "/usr/local/cuda/lib64/libcudart.so",
        ),
        (
            "7f00-7f01 r-xp 0 00:00 0 /usr/local/cuda/lib64/libcudart.so.13.0",
            "/usr/local/cuda/lib64/libcudart.so.13.0",
        ),
        (
            "\n".join(
                [
                    "7f00-7f01 r-xp 0 00:00 0 /pkg/tilelang/lib/libcudart_stub.so",
                    "7f01-7f02 r-xp 0 00:00 0 /usr/local/cuda/lib64/libcudart.so.13",
                ]
            ),
            "/usr/local/cuda/lib64/libcudart.so.13",
        ),
        (
            "7f00-7f01 r-xp 0 00:00 0 /pkg/torch/lib/libcudart-deadbeef.so.12",
            "/pkg/torch/lib/libcudart-deadbeef.so.12",
        ),
        (
            "7f00-7f01 r-xp 0 00:00 0 /pkg/torch/lib/libcudart-deadbeef.so",
            "/pkg/torch/lib/libcudart-deadbeef.so",
        ),
        (
            "7f00-7f01 r-xp 0 00:00 0 /pkg/libcudart_stub.so",
            None,
        ),
        (
            "7f00-7f01 r-xp 0 00:00 0 /pkg/libcudart-stub.so",
            None,
        ),
        (
            "7f00-7f01 r-xp 0 00:00 0 /pkg/libcudart.foo.so",
            None,
        ),
        (
            "7f00-7f01 r-xp 0 00:00 0 /pkg/libcudart.sofoo.so",
            None,
        ),
        (
            "7f00-7f01 r-xp 0 00:00 0 /pkg/libcudart-.so",
            None,
        ),
        (
            "7f00-7f01 r-xp 0 00:00 0 /pkg/libcudart/libcuda.so.1",
            None,
        ),
        (
            "7f00-7f01 r-xp 0 00:00 0 /pkg/libcuda.so.1",
            None,
        ),
    ],
)
def test_find_loaded_library_matches_only_runtime_filenames(maps, expected):
    with mock.patch("builtins.open", return_value=StringIO(maps)):
        assert cuda_ipc.find_loaded_library("libcudart") == expected
