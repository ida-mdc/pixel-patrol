"""Unit tests for _task_image_count and _count_failed_images: how many images
a failed task's error should count as."""
from __future__ import annotations

from pixel_patrol_base.core.processing import (
    BatchTask,
    ContainerTask,
    MemoryChunkTask,
    _IndexedPath,
    _count_failed_images,
    _task_image_count,
)


def test_batch_task_counts_all_its_files():
    task = BatchTask(files=(
        _IndexedPath(file_index=0, file_path="a.tif"),
        _IndexedPath(file_index=1, file_path="b.tif"),
        _IndexedPath(file_index=2, file_path="c.tif"),
    ))
    assert _task_image_count(task) == 3


def test_container_task_counts_its_image_slice_span():
    task = ContainerTask(file_index=0, file_path="x.lmdb", image_slice=(10, 15))
    assert _task_image_count(task) == 5


def test_container_task_single_image_slice():
    task = ContainerTask(file_index=0, file_path="x.lif", image_slice=(3, 4))
    assert _task_image_count(task) == 1


def test_memory_chunk_task_counts_as_one_image():
    task = MemoryChunkTask(file_index=0, file_path="big.tif", spec=None, n_memory_chunks=6)
    assert _task_image_count(task) == 1


def test_two_failing_chunks_of_same_image_count_once():
    failed: set = set()
    t1 = MemoryChunkTask(file_index=5, file_path="big.tif", spec=None, n_memory_chunks=6)
    t2 = MemoryChunkTask(file_index=5, file_path="big.tif", spec=None, n_memory_chunks=6)
    assert _count_failed_images(t1, failed) == 1
    assert _count_failed_images(t2, failed) == 0
    assert failed == {5}


def test_failing_chunks_of_different_images_both_count():
    failed: set = set()
    t1 = MemoryChunkTask(file_index=5, file_path="a.tif", spec=None, n_memory_chunks=3)
    t2 = MemoryChunkTask(file_index=6, file_path="b.tif", spec=None, n_memory_chunks=3)
    assert _count_failed_images(t1, failed) == 1
    assert _count_failed_images(t2, failed) == 1
    assert failed == {5, 6}


def test_container_and_batch_tasks_unaffected_by_dedup_set():
    failed: set = set()
    container = ContainerTask(file_index=0, file_path="x.lmdb", image_slice=(0, 4))
    batch = BatchTask(files=(_IndexedPath(file_index=1, file_path="a.tif"),))
    assert _count_failed_images(container, failed) == 4
    assert _count_failed_images(batch, failed) == 1
    assert failed == set()
