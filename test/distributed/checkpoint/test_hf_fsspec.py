# Owner(s): ["oncall: distributed checkpointing"]

import importlib.util
import json
import os
import posixpath
import shutil
import tempfile
import unittest
import uuid
from unittest import mock

import fsspec

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files,
    consolidate_safetensors_files_on_every_rank,
)
from torch.distributed.checkpoint._hf_utils import (
    _get_fs_and_path,
    _list_safetensors_files,
    _metadata_fn,
    _open_safetensors,
)
from torch.distributed.checkpoint.filesystem import FileSystem
from torch.distributed.checkpoint.fsspec_filesystem import _FileSystem
from torch.distributed.checkpoint.quantized_hf_storage import (
    QuantizedHuggingFaceStorageReader,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None


class _MemoryFsTestCase(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.name = f"test-{uuid.uuid4().hex}"
        self.root = f"memory://{self.name}"
        self.memory_fs = fsspec.filesystem("memory")
        self.addCleanup(self._remove_root)
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def _remove_root(self) -> None:
        if self.memory_fs.exists(f"/{self.name}"):
            self.memory_fs.rm(f"/{self.name}", recursive=True)

    def _write_bytes(self, url: str, data: bytes) -> None:
        with fsspec.open(url, "wb") as f:
            f.write(data)

    def _read_bytes(self, url: str) -> bytes:
        with fsspec.open(url, "rb") as f:
            return f.read()


@unittest.skipIf(not HAS_SAFETENSORS, "safetensors not installed")
@instantiate_parametrized_tests
class TestFsspecSafetensorsFile(_MemoryFsTestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self) -> None:
        super().setUp()
        from safetensors.torch import save

        self.tensors = {
            "a": torch.arange(24, dtype=torch.float32).view(4, 6),
            "b": torch.arange(8, dtype=torch.bfloat16).view(2, 2, 2),
            "s": torch.tensor(3.0),
            "u8": torch.arange(12, dtype=torch.uint8).view(3, 4),
            "i32": torch.arange(-6, 6, dtype=torch.int32).view(4, 3),
            "f8": torch.arange(6, dtype=torch.float32).to(torch.float8_e4m3fn),
            "bool": torch.tensor([[True, False], [False, True]]),
        }
        data = save(self.tensors, metadata={"k": "v"})
        self.local_path = os.path.join(self.temp_dir, "f.safetensors")
        with open(self.local_path, "wb") as f:
            f.write(data)
        self.remote_path = f"{self.root}/f.safetensors"
        self._write_bytes(self.remote_path, data)
        self.remote_fs, _ = _get_fs_and_path(self.remote_path)

    def test_get_fs_and_path_picks_backend(self) -> None:
        self.assertIsInstance(_get_fs_and_path(self.local_path)[0], FileSystem)
        self.assertIsInstance(self.remote_fs, _FileSystem)

    def test_header_matches_safe_open(self) -> None:
        local_fs, _ = _get_fs_and_path(self.local_path)
        with (
            _open_safetensors(local_fs, self.local_path) as local,
            _open_safetensors(self.remote_fs, self.remote_path) as remote,
        ):
            self.assertEqual(sorted(remote.keys()), sorted(local.keys()))
            self.assertEqual(remote.metadata(), local.metadata())
            for key in self.tensors:
                self.assertEqual(
                    remote.get_slice(key).get_shape(), local.get_slice(key).get_shape()
                )
                self.assertEqual(
                    remote.get_slice(key).get_dtype(), local.get_slice(key).get_dtype()
                )

    @parametrize(
        "key,index",
        [
            ("a", ()),
            ("a", (slice(1, 3),)),
            ("a", (slice(None), slice(2, 5))),
            ("a", (slice(1, 3), slice(0, 4))),
            ("a", (slice(3, 3),)),
            ("a", (slice(3, 1),)),
            ("a", (slice(-2, None),)),
            ("a", (slice(0, 4, 2),)),
            ("a", (slice(None), slice(0, 6, 2))),
            ("a", 1),
            ("a", -1),
            ("a", (slice(None), 0)),
            ("a", (1, 2)),
            ("a", Ellipsis),
            ("a", (Ellipsis, slice(1, 3))),
            ("a", (0, Ellipsis)),
            ("b", (slice(1, 2), slice(None), slice(0, 1))),
            ("b", (Ellipsis, 0)),
            ("s", ()),
            ("s", Ellipsis),
            ("u8", (slice(1, 3), slice(1, 4))),
            ("i32", (slice(2, 4),)),
            ("f8", (slice(2, 5),)),
            ("bool", (slice(1, 2),)),
        ],
    )
    def test_index_matches_safe_open(self, key, index) -> None:
        local_fs, _ = _get_fs_and_path(self.local_path)
        with (
            _open_safetensors(local_fs, self.local_path) as local,
            _open_safetensors(self.remote_fs, self.remote_path) as remote,
        ):
            result = remote.get_slice(key)[index]
            self.assertEqual(result, local.get_slice(key)[index])
            self.assertEqual(result, self.tensors[key][index])

    @parametrize(
        "key,index",
        [
            ("a", 4),
            ("a", -5),
            ("a", (slice(None),) * 3),
            ("a", (slice(None, None, -1),)),
            ("s", 0),
        ],
    )
    def test_invalid_index_raises_like_safe_open(self, key, index) -> None:
        local_fs, _ = _get_fs_and_path(self.local_path)
        with (
            _open_safetensors(local_fs, self.local_path) as local,
            _open_safetensors(self.remote_fs, self.remote_path) as remote,
        ):
            with self.assertRaises(Exception) as expected:
                local.get_slice(key)[index]
            with self.assertRaises(type(expected.exception)):
                remote.get_slice(key)[index]

    def test_get_tensor(self) -> None:
        with _open_safetensors(self.remote_fs, self.remote_path) as remote:
            for key, tensor in self.tensors.items():
                self.assertEqual(remote.get_tensor(key), tensor)

    def test_list_safetensors_files(self) -> None:
        self._write_bytes(f"{self.root}/b.safetensors", b"")
        self._write_bytes(f"{self.root}/model.safetensors.index.json", b"{}")
        fs, path = _get_fs_and_path(self.root)
        files = _list_safetensors_files(fs, path)
        self.assertEqual(
            [posixpath.basename(f) for f in files], ["b.safetensors", "f.safetensors"]
        )

    def test_list_invalidates_fsspec_cache(self) -> None:
        fs, path = _get_fs_and_path(self.root)
        with mock.patch.object(fs.fs, "invalidate_cache") as invalidate:
            _list_safetensors_files(fs, path)
        invalidate.assert_called_once_with()

    def test_remote_reads_disable_readahead(self) -> None:
        with mock.patch.object(
            self.remote_fs.fs, "open", wraps=self.remote_fs.fs.open
        ) as fs_open:
            with _open_safetensors(self.remote_fs, self.remote_path) as remote:
                remote.get_slice("a")[(slice(1, 3),)]
        fs_open.assert_called_once_with(self.remote_path, "rb", cache_type="none")

    def test_truncated_file_raises(self) -> None:
        from safetensors.torch import save

        path = f"{self.root}/truncated.safetensors"
        self._write_bytes(path, save({"x": torch.arange(100.0)})[:-8])
        with _open_safetensors(self.remote_fs, path) as remote:
            with self.assertRaisesRegex(ValueError, "Truncated safetensors file"):
                remote.get_tensor("x")


@unittest.skipIf(not HAS_SAFETENSORS, "safetensors not installed")
@instantiate_parametrized_tests
class TestConsolidateOnFsspec(_MemoryFsTestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self) -> None:
        super().setUp()
        from safetensors.torch import save

        self.w = torch.arange(16, dtype=torch.float32).view(4, 4)
        self.b = torch.arange(4, dtype=torch.float32)

        def sharding(offsets: dict[str, list[int]]) -> dict[str, str]:
            info = {k: {"saved_offsets": v} for k, v in offsets.items()}
            return {"DCP_SHARDING_INFO": json.dumps(info)}

        self.shards = {
            "shard-00001-model-00001-of-00001.safetensors": save(
                {"w": self.w[:2].contiguous(), "b": self.b},
                metadata=sharding({"w": [0, 0], "b": [0]}),
            ),
            "shard-00002-model-00001-of-00001.safetensors": save(
                {"w": self.w[2:].contiguous()}, metadata=sharding({"w": [2, 0]})
            ),
        }
        self.mapping = {"w": 1, "b": 2}

    def _write_remote_shards(self) -> None:
        for name, data in self.shards.items():
            self._write_bytes(f"{self.root}/sharded/{name}", data)

    def test_consolidate_to_remote(self) -> None:
        from safetensors.torch import load

        self._write_remote_shards()
        with mock.patch.object(
            self.memory_fs, "open", wraps=self.memory_fs.open
        ) as fs_open:
            consolidate_safetensors_files(
                input_dir=f"{self.root}/sharded",
                output_dir=self.root,
                fqn_to_index_mapping=self.mapping,
            )

        # shard-00002 holds only "w" (output file 1), so output file 2 ("b") must
        # not open it: 1 header read + 1 data read = 2 opens.
        shard2_opens = [
            call for call in fs_open.call_args_list if "shard-00002" in call.args[0]
        ]
        self.assertEqual(len(shard2_opens), 2)

        w_file = self._read_bytes(f"{self.root}/model-00001-of-00002.safetensors")
        b_file = self._read_bytes(f"{self.root}/model-00002-of-00002.safetensors")
        self.assertEqual(load(w_file), {"w": self.w})
        self.assertEqual(load(b_file), {"b": self.b})
        index = json.loads(self._read_bytes(f"{self.root}/{_metadata_fn}"))
        self.assertEqual(
            index["weight_map"],
            {
                "w": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            },
        )

    def test_consolidate_several_tensors_per_file(self) -> None:
        from safetensors.torch import load

        self._write_remote_shards()
        consolidate_safetensors_files(
            input_dir=f"{self.root}/sharded",
            output_dir=self.root,
            fqn_to_index_mapping={"w": 1, "b": 1},
        )

        data = self._read_bytes(f"{self.root}/model-00001-of-00001.safetensors")
        self.assertEqual(load(data), {"w": self.w, "b": self.b})

    def test_local_and_remote_outputs_are_identical(self) -> None:
        local_in = os.path.join(self.temp_dir, "sharded")
        os.makedirs(local_in)
        for name, data in self.shards.items():
            with open(os.path.join(local_in, name), "wb") as f:
                f.write(data)
        self._write_remote_shards()
        # Inputs and outputs may live on different filesystems.
        local_from_remote = os.path.join(self.temp_dir, "from_remote")
        os.makedirs(local_from_remote)
        for input_dir, output_dir in (
            (local_in, self.temp_dir),
            (f"{self.root}/sharded", self.root),
            (local_in, f"{self.root}/from_local"),
            (f"{self.root}/sharded", local_from_remote),
        ):
            consolidate_safetensors_files(input_dir, output_dir, self.mapping)

        for name in (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            _metadata_fn,
        ):
            with open(os.path.join(self.temp_dir, name), "rb") as f:
                expected = f.read()
            with open(os.path.join(local_from_remote, name), "rb") as f:
                self.assertEqual(f.read(), expected)
            self.assertEqual(self._read_bytes(f"{self.root}/{name}"), expected)
            self.assertEqual(
                self._read_bytes(f"{self.root}/from_local/{name}"), expected
            )

    def test_consolidate_on_every_rank_without_dist(self) -> None:
        from safetensors.torch import load

        self._write_remote_shards()
        consolidate_safetensors_files_on_every_rank(
            input_dir=f"{self.root}/sharded",
            output_dir=self.root,
            fqn_to_index_mapping=self.mapping,
        )
        w_file = self._read_bytes(f"{self.root}/model-00001-of-00002.safetensors")
        b_file = self._read_bytes(f"{self.root}/model-00002-of-00002.safetensors")
        self.assertEqual(load(w_file), {"w": self.w})
        self.assertEqual(load(b_file), {"b": self.b})
        index = json.loads(self._read_bytes(f"{self.root}/{_metadata_fn}"))
        self.assertEqual(
            index["weight_map"],
            {
                "w": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            },
        )

    @parametrize("remote_input", [True, False])
    @parametrize("use_file_uri", [False, True])
    def test_consolidate_on_every_rank_across_filesystems(
        self, remote_input: bool, use_file_uri: bool
    ) -> None:
        from safetensors.torch import load

        local_path = os.path.join(self.temp_dir, "dir")
        local_dir = f"file://{local_path}" if use_file_uri else local_path
        if remote_input:
            self._write_remote_shards()
            input_dir, output_dir = f"{self.root}/sharded", local_dir
        else:
            os.makedirs(local_path)
            for name, data in self.shards.items():
                with open(os.path.join(local_path, name), "wb") as f:
                    f.write(data)
            input_dir, output_dir = local_dir, f"{self.root}/out"

        consolidate_safetensors_files_on_every_rank(input_dir, output_dir, self.mapping)

        def read(name: str) -> bytes:
            if remote_input:
                with open(os.path.join(local_path, name), "rb") as f:
                    return f.read()
            return self._read_bytes(f"{output_dir}/{name}")

        self.assertEqual(load(read("model-00001-of-00002.safetensors")), {"w": self.w})
        self.assertEqual(load(read("model-00002-of-00002.safetensors")), {"b": self.b})
        self.assertEqual(
            json.loads(read(_metadata_fn))["weight_map"].keys(), {"w", "b"}
        )


@unittest.skipIf(not HAS_SAFETENSORS, "safetensors not installed")
class TestHfStorageOnFsspec(_MemoryFsTestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self) -> None:
        super().setUp()
        self.path = f"{self.root}/ckpt"
        self.state_dict = {"w": torch.rand(4, 6), "b": torch.rand(6)}

    def _load(self, reader) -> dict[str, torch.Tensor]:
        loaded = {k: torch.zeros_like(v) for k, v in self.state_dict.items()}
        dist_cp.load(loaded, storage_reader=reader)
        return loaded

    def test_save_load_round_trip(self) -> None:
        dist_cp.save(
            self.state_dict,
            storage_writer=dist_cp.HuggingFaceStorageWriter(path=self.path),
        )
        for name in ("model-00001-of-00001.safetensors", _metadata_fn):
            self.assertTrue(self.memory_fs.exists(f"/{self.name}/ckpt/{name}"))
        loaded = self._load(dist_cp.HuggingFaceStorageReader(path=self.path))
        self.assertEqual(loaded, self.state_dict)

    def test_save_distributed_with_consolidation(self) -> None:
        dist_cp.save(
            self.state_dict,
            storage_writer=dist_cp.HuggingFaceStorageWriter(
                path=self.path,
                fqn_to_index_mapping={"w": 1, "b": 2},
                save_distributed=True,
                enable_consolidation=True,
            ),
        )
        for name in (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            _metadata_fn,
        ):
            self.assertTrue(self.memory_fs.exists(f"/{self.name}/ckpt/{name}"))
        loaded = self._load(
            dist_cp.HuggingFaceStorageReader(path=self.path, thread_count=2)
        )
        self.assertEqual(loaded, self.state_dict)

    def test_checkpoint_id_switches_filesystem(self) -> None:
        # Storage built on a local path, then pointed at a URI via checkpoint_id.
        writer = dist_cp.HuggingFaceStorageWriter(
            path=self.temp_dir,
            fqn_to_index_mapping={"w": 1, "b": 2},
            save_distributed=True,
            enable_consolidation=True,
        )
        dist_cp.save(self.state_dict, checkpoint_id=self.path, storage_writer=writer)
        for name in (
            "sharded/shard-00001-model-00001-of-00002.safetensors",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            _metadata_fn,
        ):
            self.assertTrue(self.memory_fs.exists(f"/{self.name}/ckpt/{name}"))
        self.assertEqual(os.listdir(self.temp_dir), [])

        loaded = {k: torch.zeros_like(v) for k, v in self.state_dict.items()}
        reader = dist_cp.HuggingFaceStorageReader(path=self.temp_dir)
        dist_cp.load(loaded, checkpoint_id=self.path, storage_reader=reader)
        self.assertEqual(loaded, self.state_dict)

        # Switching back from remote to local does not re-initialize the old remote fs.
        local_ckpt = os.path.join(self.temp_dir, "local_ckpt")
        with mock.patch.object(writer.fs, "init_path") as remote_writer_init:
            dist_cp.save(
                self.state_dict, checkpoint_id=local_ckpt, storage_writer=writer
            )
        remote_writer_init.assert_not_called()

        loaded_local = {k: torch.zeros_like(v) for k, v in self.state_dict.items()}
        with mock.patch.object(reader.fs, "init_path") as remote_reader_init:
            dist_cp.load(loaded_local, checkpoint_id=local_ckpt, storage_reader=reader)
        remote_reader_init.assert_not_called()
        self.assertEqual(loaded_local, self.state_dict)

    def test_validate_checkpoint_id(self) -> None:
        for cls in (
            dist_cp.HuggingFaceStorageWriter,
            dist_cp.HuggingFaceStorageReader,
            QuantizedHuggingFaceStorageReader,
        ):
            self.assertTrue(cls.validate_checkpoint_id(self.temp_dir))
            self.assertTrue(cls.validate_checkpoint_id(self.path))


@unittest.skipIf(not HAS_SAFETENSORS, "safetensors not installed")
class TestQuantizedHfStorageOnFsspec(_MemoryFsTestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_scale_in_separate_file(self) -> None:
        from safetensors.torch import save

        # Each weight's scale lives in the other file, so with two files the
        # reader's worker threads open scale files while reading weights.
        original1 = torch.randn(256, 128)
        original2 = torch.randn(256, 128)
        file1 = "model-00001-of-00002.safetensors"
        file2 = "model-00002-of-00002.safetensors"
        self._write_bytes(
            f"{self.root}/{file1}",
            save(
                {
                    "linear1.weight": original1 / 2.0,
                    "linear2.weight_scale_inv": torch.full((2, 1), 4.0),
                }
            ),
        )
        self._write_bytes(
            f"{self.root}/{file2}",
            save(
                {
                    "linear2.weight": original2 / 4.0,
                    "linear1.weight_scale_inv": torch.full((2, 1), 2.0),
                }
            ),
        )
        weight_map = {
            "linear1.weight": file1,
            "linear2.weight_scale_inv": file1,
            "linear2.weight": file2,
            "linear1.weight_scale_inv": file2,
        }
        self._write_bytes(
            f"{self.root}/{_metadata_fn}",
            json.dumps({"weight_map": weight_map}).encode(),
        )

        loaded = {
            "linear1.weight": torch.zeros(256, 128),
            "linear2.weight": torch.zeros(256, 128),
        }
        dist_cp.load(
            loaded,
            storage_reader=QuantizedHuggingFaceStorageReader(
                path=self.root, block_size=128, thread_count=2
            ),
        )
        self.assertEqual(loaded["linear1.weight"], original1)
        self.assertEqual(loaded["linear2.weight"], original2)


if __name__ == "__main__":
    run_tests()
