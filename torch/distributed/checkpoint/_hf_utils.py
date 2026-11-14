import functools
import io
import json
import math
import os
import struct
from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from types import EllipsisType
from typing import Any, cast, NamedTuple, TYPE_CHECKING, TypeVar

import torch


if TYPE_CHECKING:
    from torch.distributed.checkpoint.filesystem import FileSystem, FileSystemBase
    from torch.distributed.checkpoint.fsspec_filesystem import _FileSystem

_Index = int | slice | EllipsisType


_metadata_fn: str = "model.safetensors.index.json"

FILE_NAME = "model-{cpt_idx}-of-{num_files}"
SHARDED_FILE_NAME = "shard-{shard_idx}-model-{cpt_idx}-of-{num_files}"
SUFFIX = ".safetensors"

# metadata keys
CUSTOM_METADATA_KEY = "DCP_SHARDING_INFO"
DEFAULT_EXTRA_METADATA_KEY = "__metadata__"
SAVED_OFFSETS_KEY = "saved_offsets"
SHAPE_KEY = "shape"
DATA_KEY = "data"
DTYPE_KEY = "dtype"
DATA_OFFSETS_KEY = "data_offsets"

DTYPE_MAP = {
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BF16": torch.bfloat16,
}

HF_DCP_VERSION: float = 1.0
DCP_VERSION_KEY = "DCP_VERSION"
DCP_SHARDING_INFO_KEY = "DCP_SHARDING_INFO"

FORMAT_KEY = "format"
FORMAT_VALUE = "pt"

NUM_BYTES_FOR_HEADER_LEN = 8

SHARDED_DIR_NAME = "sharded"

_MAX_REMOTE_READ_THREADS = 16
# iter_slices fetches about this many bytes per batch of concurrent ranged reads,
# split into chunks so a single large tensor is also fetched concurrently. It fetches
# the next batch while turning the current one into tensors.
_RANGED_READ_BATCH_BYTES = 512 * 1024 * 1024
_RANGED_READ_CHUNK_BYTES = 32 * 1024 * 1024

_T = TypeVar("_T")


@dataclass
class _HFStorageInfo:
    """This is the per entry storage info."""

    relative_path: str
    shape: torch.Size
    dtype: torch.dtype


def _gen_file_name(
    index: int, largest_index: int, shard_index: int | None = None
) -> str:
    if shard_index is not None:
        return (
            SHARDED_FILE_NAME.format(
                shard_idx=f"{shard_index}".zfill(5),
                cpt_idx=f"{index}".zfill(5),
                num_files=f"{largest_index}".zfill(5),
            )
            + SUFFIX
        )
    else:
        return (
            FILE_NAME.format(
                cpt_idx=f"{index}".zfill(5), num_files=f"{largest_index}".zfill(5)
            )
            + SUFFIX
        )


def _get_safetensors_file_metadata(file_bytes: io.IOBase) -> tuple[Any, int]:
    # this uses the same logic that's done in HF code base
    # https://github.com/2404589803/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py#L5308
    # and follows their documentation on how their files are serialized
    # https://huggingface.co/docs/safetensors/index#format

    header_len_bytes = file_bytes.read(NUM_BYTES_FOR_HEADER_LEN)
    header_len = struct.unpack("<Q", header_len_bytes)[0]
    header_json = file_bytes.read(header_len)
    metadata = json.loads(header_json)
    return (metadata, header_len + NUM_BYTES_FOR_HEADER_LEN)


def _get_fs_and_path(
    path: str | os.PathLike,
) -> tuple["FileSystem | _FileSystem", str | os.PathLike]:
    """Return the DCP filesystem for ``path`` (fsspec for URIs) and the initialized path."""
    from torch.distributed.checkpoint.filesystem import FileSystem
    from torch.distributed.checkpoint.fsspec_filesystem import _FileSystem

    if "://" not in str(path):
        fs = FileSystem()
        return fs, fs.init_path(path)
    remote_fs = _FileSystem()
    # Other ranks and processes write the files listed and opened here, and
    # fsspec's listing cache only sees writes made through the same instance.
    return remote_fs, remote_fs.init_path(path, use_listings_cache=False)


def _list_safetensors_files(
    fs: "FileSystem | _FileSystem", path: str | os.PathLike
) -> list[str]:
    return sorted(f for f in fs.ls(path) if f.endswith(SUFFIX))


@contextmanager
def _open_for_ranged_reads(
    fs: "FileSystemBase", path: str | os.PathLike
) -> Generator[io.IOBase, None, None]:
    """Open ``path`` for reads whose byte ranges the caller already knows."""
    from torch.distributed.checkpoint.fsspec_filesystem import _FileSystem

    if isinstance(fs, _FileSystem) and fs.fs is not None:
        # fsspec's default readahead cache fetches a whole extra block (5 MiB
        # on gcsfs) per read, so a header read would download 5 MiB per file.
        with fs.fs.open(os.fspath(path), "rb", cache_type="none") as stream:
            yield stream
    else:
        with fs.create_stream(path, "rb") as stream:
            yield stream


def _map_over_files(
    fs: "FileSystemBase", fn: Callable[[str], _T], files: list[str]
) -> list[_T]:
    """Apply ``fn`` to each file, in parallel on remote filesystems where each call is a round trip."""
    from torch.distributed.checkpoint.filesystem import FileSystem

    if isinstance(fs, FileSystem) or len(files) <= 1:
        return [fn(f) for f in files]
    with ThreadPoolExecutor(
        max_workers=min(_MAX_REMOTE_READ_THREADS, len(files))
    ) as ex:
        return list(ex.map(fn, files))


class _RangePlan(NamedTuple):
    """The data bytes to read for one index, and how to turn them into the result."""

    offset: int
    nbytes: int
    dtype: torch.dtype
    rows_shape: list[int]
    rows_index: _Index | tuple[_Index, ...]


class _FsspecSafetensorsSlice:
    def __init__(self, file: "_FsspecSafetensorsFile", name: str) -> None:
        self._file = file
        self._info = file._header[name]

    def get_shape(self) -> list[int]:
        return list(self._info[SHAPE_KEY])

    def get_dtype(self) -> str:
        return self._info[DTYPE_KEY]

    def __getitem__(self, index: _Index | tuple[_Index, ...]) -> torch.Tensor:
        plan = self._plan(index)
        return self._file._read_tensor(plan)[plan.rows_index]

    def _plan(self, index: _Index | tuple[_Index, ...]) -> _RangePlan:
        from safetensors.torch import _getdtype  # type: ignore[import]

        shape = self.get_shape()
        dtype = _getdtype(self.get_dtype())
        begin, end = self._info[DATA_OFFSETS_KEY]
        if not shape:
            return _RangePlan(begin, end - begin, dtype, [], index)
        index = index if isinstance(index, tuple) else (index,)
        if any(i is Ellipsis for i in index):
            e = next(n for n, i in enumerate(index) if i is Ellipsis)
            fill = (slice(None),) * (len(shape) - len(index) + 1)
            index = index[:e] + fill + index[e + 1 :]
        first, rest = (index[0], index[1:]) if index else (slice(None), ())
        if isinstance(first, int):
            # range() normalizes negative indices and raises IndexError like torch.
            start = range(shape[0])[first]
            stop, first = start + 1, 0
        elif isinstance(first, slice):
            start, stop, step = first.indices(shape[0])
            stop, first = max(stop, start), slice(None, None, step)
        else:
            raise TypeError(f"Unsupported index: {index!r}")
        # Read only the dim-0 rows covering the request, then index them in memory.
        row_nbytes = math.prod(shape[1:]) * dtype.itemsize
        rows_shape = [stop - start, *shape[1:]]
        return _RangePlan(
            begin + start * row_nbytes,
            (stop - start) * row_nbytes,
            dtype,
            rows_shape,
            (first, *rest),
        )


class _FsspecSafetensorsFile:
    """The ``safetensors.safe_open`` API that DCP uses, over an fsspec stream."""

    def __init__(
        self,
        stream: io.IOBase,
        cat_ranges: Callable[[list[str], list[int], list[int]], list[bytes]]
        | None = None,
        path: str | None = None,
    ) -> None:
        # Binary fsspec and local file streams provide readinto.
        self._stream = cast(io.BufferedIOBase, stream)
        # With fsspec's cat_ranges and the path, iter_slices fetches ranges concurrently.
        self._cat_ranges = cat_ranges
        self._path = path
        header, self._data_start = _get_safetensors_file_metadata(stream)
        self._metadata = header.pop(DEFAULT_EXTRA_METADATA_KEY, None)
        self._header = header

    def __enter__(self) -> "_FsspecSafetensorsFile":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def keys(self) -> list[str]:
        return sorted(self._header)

    def metadata(self) -> dict[str, str] | None:
        return self._metadata

    def get_slice(self, name: str) -> _FsspecSafetensorsSlice:
        return _FsspecSafetensorsSlice(self, name)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.get_slice(name)[()]

    def iter_slices(
        self, requests: Sequence[tuple[str, _Index | tuple[_Index, ...]]]
    ) -> Iterator[torch.Tensor]:
        """Yield ``get_slice(name)[index]`` for each request, fetching the byte ranges in concurrent batches."""
        if self._cat_ranges is None or not requests:
            yield from (self.get_slice(name)[index] for name, index in requests)
            return
        batches: list[list[_RangePlan]] = [[]]
        batch_bytes = 0
        for name, index in requests:
            if batch_bytes >= _RANGED_READ_BATCH_BYTES:
                batches.append([])
                batch_bytes = 0
            plan = self.get_slice(name)._plan(index)
            batches[-1].append(plan)
            batch_bytes += plan.nbytes
        with ThreadPoolExecutor(max_workers=1) as prefetcher:
            pending = prefetcher.submit(self._fetch_batch, batches[0])
            for n, batch in enumerate(batches):
                parts = pending.result()
                # Fetch the next batch while this one is turned into tensors.
                if n + 1 < len(batches):
                    pending = prefetcher.submit(self._fetch_batch, batches[n + 1])
                for plan, chunks in zip(batch, parts):
                    data = bytearray().join(chunks)
                    yield self._to_tensor(data, len(data), plan)[plan.rows_index]

    def _fetch_batch(self, plans: list[_RangePlan]) -> list[list[bytes]]:
        """Fetch the data bytes of ``plans`` concurrently, as chunks per plan."""
        if self._cat_ranges is None or self._path is None:
            raise AssertionError("_fetch_batch needs cat_ranges and the path")
        owners, starts, ends = [], [], []
        for n, plan in enumerate(plans):
            stop = plan.offset + plan.nbytes
            for lo in range(plan.offset, stop, _RANGED_READ_CHUNK_BYTES):
                owners.append(n)
                starts.append(self._data_start + lo)
                ends.append(self._data_start + min(lo + _RANGED_READ_CHUNK_BYTES, stop))
        parts: list[list[bytes]] = [[] for _ in plans]
        chunks = self._cat_ranges([self._path] * len(starts), starts, ends)
        for n, chunk in zip(owners, chunks):
            parts[n].append(chunk)
        return parts

    def _read_tensor(self, plan: _RangePlan) -> torch.Tensor:
        self._stream.seek(self._data_start + plan.offset)
        # readinto lets filesystems that support it write straight into the result.
        data = bytearray(plan.nbytes)
        view, got = memoryview(data), 0
        while got < plan.nbytes and (n := self._stream.readinto(view[got:])):
            got += n
        return self._to_tensor(data, got, plan)

    @staticmethod
    def _to_tensor(data: bytearray, got: int, plan: _RangePlan) -> torch.Tensor:
        if got != plan.nbytes:
            raise ValueError(
                f"Truncated safetensors file: expected {plan.nbytes} bytes at data "
                f"offset {plan.offset}, got {got}"
            )
        if plan.nbytes == 0:
            return torch.empty(plan.rows_shape, dtype=plan.dtype)
        return torch.frombuffer(data, dtype=plan.dtype).view(plan.rows_shape)


@contextmanager
def _open_safetensors(
    fs: "FileSystemBase", path: str | os.PathLike
) -> Generator[Any, None, None]:
    from torch.distributed.checkpoint.filesystem import FileSystem

    if isinstance(fs, FileSystem):
        from safetensors import safe_open  # type: ignore[import]

        with safe_open(path, framework="pt") as f:
            yield f
    else:
        from torch.distributed.checkpoint.fsspec_filesystem import _FileSystem

        with _open_for_ranged_reads(fs, path) as stream:
            remote = fs.fs if isinstance(fs, _FileSystem) else None
            cat_ranges = (
                functools.partial(remote.cat_ranges, on_error="raise")
                if remote is not None
                else None
            )
            yield _FsspecSafetensorsFile(stream, cat_ranges, os.fspath(path))


def _open_safetensors_headers(
    fs: "FileSystem | _FileSystem", path: str | os.PathLike
) -> list[tuple[str, Any]]:
    from torch.distributed.checkpoint.filesystem import FileSystem

    files = _list_safetensors_files(fs, path)
    if isinstance(fs, FileSystem):
        return [(f, _open_safetensors(fs, f)) for f in files]

    def read_header(f: str) -> _FsspecSafetensorsFile:
        with _open_for_ranged_reads(fs, f) as stream:
            return _FsspecSafetensorsFile(stream)

    return list(zip(files, _map_over_files(fs, read_header, files)))


def _get_dtype(dtype_str: str) -> torch.dtype:
    try:
        dtype = DTYPE_MAP[dtype_str]
    except KeyError:
        dtype = torch.get_default_dtype()

    return dtype


def _get_dcp_custom_metadata(metadata: Any) -> Any | None:
    if DEFAULT_EXTRA_METADATA_KEY in metadata:
        custom_metadata = metadata[DEFAULT_EXTRA_METADATA_KEY]
        if CUSTOM_METADATA_KEY in custom_metadata:
            return json.loads(custom_metadata[CUSTOM_METADATA_KEY])
    return None
