"""Unit tests for _plan_tasks, _plan_container_tasks, and FileInfo.n_images routing."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import pytest

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.processing import (
    BatchTask, ContainerTask, MemoryChunkTask, Task,
    _plan_container_tasks, _plan_tasks,
)
from pixel_patrol_base.core.processing_config import ProcessingConfig
from _processing_mocks import MockEntry, MockLoader, capture_warnings


def _stream(pairs: List[Tuple[str, int]]) -> Iterator[Tuple[Path, dict]]:
    """Inject a synthetic file_stream from (path_str, on_disk_bytes) pairs."""
    for path_str, size in pairs:
        p = Path(path_str)
        yield p, {
            "path":              path_str,
            "name":              p.name,
            "size_bytes":        size,
            "file_extension":    p.suffix.lower().lstrip("."),
            "modification_date": datetime(2026, 1, 1),
            "common_base":       p.parent.name or str(p.parent),
        }


def _run(pairs: List[Tuple[str, int]], loader: MockLoader,
         config: ProcessingConfig) -> Tuple[List[Task], List[dict]]:
    files_meta: List[dict] = []
    tasks = list(_plan_tasks(_stream(pairs), config, loader, files_meta))
    return tasks, files_meta


# ── _plan_container_tasks unit tests ─────────────────────────────────────────

def test_container_tasks_all_fit_in_one_task():
    info = FileInfo(shape=(10, 10), dtype=np.float32, dim_order=("Y", "X"), n_images=5)
    tasks = _plan_container_tasks(0, "/f.lmdb", info, 102400, 1000)
    assert len(tasks) == 1
    assert tasks[0].image_slice == (0, 5)


def test_container_tasks_splits_evenly():
    info = FileInfo(shape=(40, 40), dtype=np.float32, dim_order=("Y", "X"), n_images=20)
    tasks = _plan_container_tasks(0, "/f.lmdb", info, 102400, 1000)
    assert len(tasks) == 2
    assert tasks[0].image_slice == (0, 16)
    assert tasks[1].image_slice == (16, 20)


def test_container_tasks_one_per_image_when_large():
    info = FileInfo(shape=(128, 128), dtype=np.float32, dim_order=("Y", "X"), n_images=100)
    tasks = _plan_container_tasks(0, "/f.lmdb", info, 102400, 1000)
    assert len(tasks) == 100
    assert [t.image_slice[0] for t in tasks] == list(range(100))


def test_container_tasks_image_exceeds_budget():
    info = FileInfo(shape=(512, 512), dtype=np.float32, dim_order=("Y", "X"), n_images=1)
    tasks = _plan_container_tasks(0, "/f.lmdb", info, 10, 1000)
    assert len(tasks) == 1
    assert tasks[0].image_slice == (0, 1)


def test_container_tasks_file_index_and_path():
    info = FileInfo(shape=(10, 10), dtype=np.float32, dim_order=("Y", "X"), n_images=3)
    tasks = _plan_container_tasks(42, "/custom/path.lmdb", info, 102400, 1000)
    assert all(t.file_index == 42 for t in tasks)
    assert all(t.file_path == "/custom/path.lmdb" for t in tasks)


def test_container_tasks_capped_by_max_images_per_task():
    info = FileInfo(shape=(10, 10), dtype=np.float32, dim_order=("Y", "X"), n_images=100)
    tasks = _plan_container_tasks(0, "/f.lmdb", info, 102400, 10)
    assert len(tasks) == 10
    assert tasks[0].image_slice == (0, 10)
    assert tasks[-1].image_slice == (90, 100)


# ── _plan_tasks: batch routing ───────────────────────────────────────────────

def test_small_files_batched_into_one_task():
    loader = MockLoader({
        "/a.npy": MockEntry((64, 64), np.float32, ("Y", "X")),
        "/b.npy": MockEntry((64, 64), np.float32, ("Y", "X")),
        "/c.npy": MockEntry((64, 64), np.float32, ("Y", "X")),
    })
    config = ProcessingConfig(mb_per_task=0.1)
    tasks, fi = _run([("/a.npy", 10240), ("/b.npy", 10240), ("/c.npy", 10240)], loader, config)
    assert len(tasks) == 1
    assert isinstance(tasks[0], BatchTask)
    assert len(tasks[0].files) == 3
    assert len(fi) == 3


def test_batch_flushes_when_on_disk_budget_reached():
    loader = MockLoader({
        "/a.npy": MockEntry((80, 80), np.float32, ("Y", "X")),
        "/b.npy": MockEntry((80, 80), np.float32, ("Y", "X")),
        "/c.npy": MockEntry((80, 80), np.float32, ("Y", "X")),
    })
    config = ProcessingConfig(mb_per_task=50 / 1024)
    tasks, _ = _run([("/a.npy", 25600), ("/b.npy", 25600), ("/c.npy", 25600)], loader, config)
    assert len(tasks) == 2
    assert all(isinstance(t, BatchTask) for t in tasks)
    assert len(tasks[0].files) == 2
    assert len(tasks[1].files) == 1


def test_files_meta_assigned_sequentially():
    loader = MockLoader({
        "/x.npy": MockEntry((64, 64), np.float32, ("Y", "X")),
        "/y.npy": MockEntry((64, 64), np.float32, ("Y", "X")),
    })
    config = ProcessingConfig(mb_per_task=0.1)
    tasks, fi = _run([("/x.npy", 5000), ("/y.npy", 5000)], loader, config)
    assert fi[0]["name"] == "x.npy"
    assert fi[1]["name"] == "y.npy"
    assert tasks[0].files[0].file_index == 0
    assert tasks[0].files[1].file_index == 1


def test_header_failure_skips_file():
    loader = MockLoader({"/mystery.npy": MockEntry((64, 64), np.float32, ("Y", "X"), fail=True)})
    config = ProcessingConfig(mb_per_task=0.1)
    with capture_warnings() as warnings:
        # 20 KB > fast-path threshold (budget/8 = 12.8 KB) so read_header is called
        tasks, fi = _run([("/mystery.npy", 20000)], loader, config)
    assert len(tasks) == 0
    assert len(fi) == 0
    assert any("skipping" in w for w in warnings)


def test_header_failure_large_file_skipped():
    loader = MockLoader({"/big.npy": MockEntry((512, 512), np.float32, ("Y", "X"), fail=True)})
    config = ProcessingConfig(mb_per_task=0.1, leaf_block_shape={"Y": 64})
    tasks, fi = _run([("/big.npy", 1024 * 1024)], loader, config)
    assert len(tasks) == 0
    assert len(fi) == 0


# ── _plan_tasks: MemoryChunkTask routing ─────────────────────────────────────

def test_large_single_image_yields_chunk_tasks():
    loader = MockLoader({"/big.npy": MockEntry((512, 512), np.float32, ("Y", "X"))})
    config = ProcessingConfig(mb_per_task=0.1, leaf_block_shape={"Y": 64})
    tasks, fi = _run([("/big.npy", 512 * 512 * 4)], loader, config)
    assert all(isinstance(t, MemoryChunkTask) for t in tasks)
    assert len(tasks) > 1
    assert all(t.file_index == 0 for t in tasks)
    assert tasks[0].n_memory_chunks == len(tasks)


def test_large_file_flushes_pending_batch_first():
    loader = MockLoader({
        "/small.npy": MockEntry((64, 64), np.float32, ("Y", "X")),
        "/big.npy":   MockEntry((512, 512), np.float32, ("Y", "X")),
    })
    config = ProcessingConfig(mb_per_task=0.1, leaf_block_shape={"Y": 64})
    tasks, _ = _run([("/small.npy", 16384), ("/big.npy", 512 * 512 * 4)], loader, config)
    assert isinstance(tasks[0], BatchTask)
    assert len(tasks[0].files) == 1
    assert all(isinstance(t, MemoryChunkTask) for t in tasks[1:])


def test_unsplittable_large_file_falls_to_batch():
    loader = MockLoader({"/pinned.npy": MockEntry((512, 512), np.float32, ("Y", "X"))})
    config = ProcessingConfig(mb_per_task=0.1, leaf_block_shape={"Y": -1, "X": -1})
    tasks, _ = _run([("/pinned.npy", 512 * 512 * 4)], loader, config)
    assert len(tasks) == 1
    assert isinstance(tasks[0], BatchTask)


def test_non_divisible_large_file_falls_to_batch_with_warning():
    loader = MockLoader({"/weird.npy": MockEntry((100, 512), np.float32, ("Y", "X"))})
    config = ProcessingConfig(mb_per_task=0.1, leaf_block_shape={"Y": 32})
    with capture_warnings() as warnings:
        tasks, _ = _run([("/weird.npy", 100 * 512 * 4)], loader, config)
    assert len(tasks) == 1
    assert isinstance(tasks[0], BatchTask)
    assert any("could not be split" in w for w in warnings)


def test_chunk_tasks_followed_by_more_batching():
    loader = MockLoader({
        "/big.npy":    MockEntry((512, 512), np.float32, ("Y", "X")),
        "/small1.npy": MockEntry((64, 64), np.float32, ("Y", "X")),
        "/small2.npy": MockEntry((64, 64), np.float32, ("Y", "X")),
    })
    config = ProcessingConfig(mb_per_task=0.1, leaf_block_shape={"Y": 64})
    tasks, _ = _run([
        ("/big.npy", 512 * 512 * 4),
        ("/small1.npy", 16384),
        ("/small2.npy", 16384),
    ], loader, config)
    chunk_tasks = [t for t in tasks if isinstance(t, MemoryChunkTask)]
    batch_tasks = [t for t in tasks if isinstance(t, BatchTask)]
    assert len(chunk_tasks) > 0
    assert len(batch_tasks) == 1
    assert len(batch_tasks[0].files) == 2


# ── _plan_tasks: ContainerTask routing ────────────────────────────────────────

def test_container_file_all_images_one_task():
    loader = MockLoader({"/c.lmdb": MockEntry((10, 10), np.float32, ("Y", "X"), n_images=5)})
    config = ProcessingConfig(mb_per_task=0.1)
    tasks, fi = _run([("/c.lmdb", 5000)], loader, config)
    assert len(tasks) == 1
    assert isinstance(tasks[0], ContainerTask)
    assert tasks[0].image_slice == (0, 5)
    assert fi[0]["name"] == "c.lmdb"


def test_container_file_split_into_multiple_tasks():
    loader = MockLoader({"/c.lmdb": MockEntry((40, 40), np.float32, ("Y", "X"), n_images=20)})
    config = ProcessingConfig(mb_per_task=0.1)
    tasks, _ = _run([("/c.lmdb", 20 * 40 * 40 * 4)], loader, config)
    assert len(tasks) == 2
    assert all(isinstance(t, ContainerTask) for t in tasks)
    assert tasks[0].image_slice == (0, 16)
    assert tasks[1].image_slice == (16, 20)
    assert all(t.file_index == 0 for t in tasks)


def test_container_file_one_task_per_image_when_large():
    loader = MockLoader({"/c.lmdb": MockEntry((200, 200), np.float32, ("Y", "X"), n_images=10)})
    config = ProcessingConfig(mb_per_task=0.1)
    tasks, _ = _run([("/c.lmdb", 10 * 200 * 200 * 4)], loader, config)
    assert len(tasks) == 10
    assert all(isinstance(t, ContainerTask) for t in tasks)
    for i, t in enumerate(tasks):
        assert t.image_slice == (i, i + 1)


def test_container_flushes_pending_batch_before_sub_image_tasks():
    loader = MockLoader({
        "/small.npy":     MockEntry((64, 64), np.float32, ("Y", "X")),
        "/container.lmdb": MockEntry((10, 10), np.float32, ("Y", "X"), n_images=5),
    })
    config = ProcessingConfig(mb_per_task=0.1)
    tasks, _ = _run([("/small.npy", 10240), ("/container.lmdb", 5000)], loader, config)
    assert len(tasks) == 2
    assert isinstance(tasks[0], BatchTask)
    assert isinstance(tasks[1], ContainerTask)


def test_container_header_failure_skips_file():
    loader = MockLoader({"/c.lmdb": MockEntry((64, 64), np.float32, ("Y", "X"), fail=True)})
    config = ProcessingConfig(mb_per_task=0.1)
    tasks, fi = _run([("/c.lmdb", 5000)], loader, config)
    assert len(tasks) == 0
    assert len(fi) == 0


# ── Mixed scenarios ───────────────────────────────────────────────────────────

def test_mixed_small_large_container_ordering():
    loader = MockLoader({
        "/small.npy":      MockEntry((64, 64), np.float32, ("Y", "X")),
        "/big.npy":        MockEntry((512, 512), np.float32, ("Y", "X")),
        "/container.lmdb": MockEntry((10, 10), np.float32, ("Y", "X"), n_images=4),
        "/final.npy":      MockEntry((64, 64), np.float32, ("Y", "X")),
    })
    config = ProcessingConfig(mb_per_task=0.1, leaf_block_shape={"Y": 64})
    tasks, fi = _run([
        ("/small.npy",      16384),
        ("/big.npy",        512 * 512 * 4),
        ("/container.lmdb", 4 * 400),
        ("/final.npy",      16384),
    ], loader, config)

    assert len(fi) == 4
    batch_tasks   = [t for t in tasks if isinstance(t, BatchTask)]
    chunk_tasks   = [t for t in tasks if isinstance(t, MemoryChunkTask)]
    container_tasks = [t for t in tasks if isinstance(t, ContainerTask)]

    assert len(batch_tasks) >= 2
    assert len(chunk_tasks) > 0
    assert len(container_tasks) == 1

    assert isinstance(tasks[0], BatchTask)
    assert all(isinstance(t, MemoryChunkTask) for t in tasks[1: 1 + len(chunk_tasks)])
    assert isinstance(tasks[-2], ContainerTask)
    assert isinstance(tasks[-1], BatchTask)
    assert all(t.file_index == 1 for t in chunk_tasks)
    assert container_tasks[0].file_index == 2


def test_all_task_types_have_correct_file_indices():
    loader = MockLoader({
        "/f0.npy":  MockEntry((64, 64), np.float32, ("Y", "X")),
        "/f1.lmdb": MockEntry((10, 10), np.float32, ("Y", "X"), n_images=3),
        "/f2.npy":  MockEntry((512, 512), np.float32, ("Y", "X")),
    })
    config = ProcessingConfig(mb_per_task=0.1, leaf_block_shape={"Y": 64})
    tasks, fi = _run([("/f0.npy", 16384), ("/f1.lmdb", 3000), ("/f2.npy", 512 * 512 * 4)], loader, config)

    assert len(fi) == 3
    assert fi[0]["name"] == "f0.npy"
    assert fi[1]["name"] == "f1.lmdb"
    assert fi[2]["name"] == "f2.npy"

    batch_tasks   = [t for t in tasks if isinstance(t, BatchTask)]
    container_tasks = [t for t in tasks if isinstance(t, ContainerTask)]
    chunk_tasks   = [t for t in tasks if isinstance(t, MemoryChunkTask)]

    assert any(ip.file_index == 0 for bt in batch_tasks for ip in bt.files)
    assert all(t.file_index == 1 for t in container_tasks)
    assert all(t.file_index == 2 for t in chunk_tasks)
