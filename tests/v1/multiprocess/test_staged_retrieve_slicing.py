# SPDX-License-Identifier: Apache-2.0
"""Staged and direct retrieve must hand the H2D planners the same chunk plan.

The staged path re-slices GPU block ids per bounce batch, so this compares it
with the direct path where slicing matters: two object groups (full attention
and a sliding window whose prefix chunks are skipped), a sub-chunk window
(blocks per window differ from blocks per chunk), several bounce batches for
one group, ``skip_first_n_tokens`` landing inside a later batch, and a
``max_batch_size`` of 1 that makes the planners re-batch inside one staged
bounce batch (``start_object_idx > 0``). Both planners run for real
(``cudaMemcpyBatchAsync`` direct copy and the block-transfer kernel); only
``device_ops``, pinning and ``torch_dev`` are faked, so no GPU is required.
"""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock
import ctypes

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.dax_coordinated_l1.staged_retrieve import StagedRetrievePool
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as mod
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.object_group_transfer import (
    downsample_and_stage_block_ids,
)
import lmcache.v1.distributed.dax_coordinated_l1.staged_retrieve as staged_mod
import lmcache.v1.multiprocess.object_group_transfer as planner

pytestmark = pytest.mark.no_shared_allocator

CHUNK = 4  # tokens per chunk
BLOCK = 2  # tokens per block -> 2 blocks per chunk
NUM_CHUNKS = 5
OBJ_BYTES = 100  # slot 256 -> offsets 0 and 128 -> two objects per batch
WINDOW = [-1, 2]  # group 0 full attention, group 1 keeps the last 2 chunks
SUBCHUNK = [CHUNK, 2]  # group 1 keeps only the last block of each chunk
BPW = [CHUNK // BLOCK, 2 // BLOCK]  # blocks per window per kernel group
KG_OF_GROUP = [[0], [1]]

# (object group, first payload byte, bytes, ((block ids, skip blocks), ...))
PlanRecord = tuple[int, int, int, tuple[tuple[tuple[int, ...], int], ...]]


def _value(ptr: int) -> int:
    return ctypes.c_ubyte.from_address(ptr).value


def _obj(group: int, chunk: int) -> TensorMemoryObj:
    value = 10 * group + chunk + 1
    raw = torch.full((OBJ_BYTES,), value, dtype=torch.uint8)
    meta = MemoryObjMetadata(
        shape=torch.Size([OBJ_BYTES]),
        dtype=torch.uint8,
        address=(1 << 30) + chunk * 4096,  # DAX-like logical offset
        phy_size=OBJ_BYTES,
        ref_count=1,
        fmt=MemoryFormat.KV_2LTD,
        shapes=[torch.Size([OBJ_BYTES])],
        dtypes=[torch.uint8],
    )
    return TensorMemoryObj(raw, meta, None)


def _context(staged_tensors: list[torch.Tensor], batch: int) -> MagicMock:
    ctx = MagicMock()
    ctx.lmcache_tokens_per_chunk = CHUNK
    ctx.max_batch_size = batch
    ctx.calculate_num_blocks.side_effect = lambda tokens, g: tokens // BLOCK

    def stage(ids: list[list[int]]) -> list[torch.Tensor]:
        tensors = [torch.tensor(list(x), dtype=torch.long) for x in ids]
        staged_tensors.extend(tensors)
        return tensors

    ctx.stage_block_ids.side_effect = stage
    ctx.kv_layer_groups_manager = SimpleNamespace(
        num_object_groups=2,
        num_kernel_groups=2,
        object_groups=[ObjectGroupInfo(kernel_group_indices=k) for k in KG_OF_GROUP],
        get_attn_desc=lambda: SimpleNamespace(
            num_chunks_in_sw=WINDOW,
            group_kinds=("attention", "recurrent"),
            is_full_attention=lambda gid: WINDOW[gid] < 0,
        ),
        get_subchunk_sw_size_tokens=lambda g: SUBCHUNK[g],
    )
    return ctx


def _resolve_ids(staged_tensors: list[torch.Tensor], ptr: int, numel: int) -> list:
    for tensor in staged_tensors:
        base = tensor.data_ptr()
        if base <= ptr < base + tensor.numel() * 8:
            start = (ptr - base) // 8
            assert start + numel <= tensor.numel()
            return tensor[start : start + numel].tolist()
    raise AssertionError("block id pointer does not belong to a staged tensor")


def _run(
    monkeypatch: pytest.MonkeyPatch,
    staged: bool,
    path: str,
    skip_first_n_tokens: int,
    batch: int,
) -> tuple[list[PlanRecord], list[tuple[int, list]]]:
    """Retrieve both groups and capture the per-chunk plan the planner built."""
    objects = {g: [_obj(g, c) for c in range(NUM_CHUNKS)] for g in range(2)}
    keys = [[(g, c) for c in range(NUM_CHUNKS)] for g in range(2)]
    by_key = {k: o for g in range(2) for k, o in zip(keys[g], objects[g], strict=True)}
    plan: list[PlanRecord] = []
    current_group: list[int] = []
    transfer_calls: list[tuple[int, list]] = []
    staged_tensors: list[torch.Tensor] = []

    @contextmanager
    def read(ks: list) -> Iterator[list[TensorMemoryObj]]:
        yield [by_key[k] for k in ks]

    storage = SimpleNamespace(
        read_prefetched_results=read,
        finish_read_prefetched=MagicMock(),
        uses_dax_coordinated_l1=staged,
        dax_coordinated_l1_backend=None,
    )
    pool = None
    if staged:
        empty = torch.empty
        monkeypatch.setattr(
            staged_mod.torch,
            "empty",
            lambda *a, **kw: empty(
                *a, **{k: v for k, v in kw.items() if k != "pin_memory"}
            ),
        )
        pool = StagedRetrievePool(1, 256, 1)
        storage.dax_coordinated_l1_backend = SimpleNamespace(staged_retrieve=pool)

    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._ctx = SimpleNamespace(
        chunk_size=CHUNK,
        storage_manager=storage,
        event_bus=MagicMock(),
        resolve_obj_keys=lambda *_: keys,
    )
    ctx = _context(staged_tensors, batch)
    module.get_and_touch_context_entry = lambda _: SimpleNamespace(
        cache_context=ctx, model_name="m", event_backend=MagicMock()
    )
    device = MagicMock()
    # Recorded completions report done, so every bounce slot is reclaimed.
    device.Event.return_value.query.return_value = True
    monkeypatch.setattr(mod, "torch_dev", device)
    monkeypatch.setattr(mod, "submit_callback_to_stream", MagicMock())
    monkeypatch.setattr(
        mod, "downsample_and_stage_block_ids", downsample_and_stage_block_ids
    )

    real_transfer = planner.transfer_kv_per_object_group

    def transfer(cache_context, block_ids_gpu, memory_objs, object_group_id, *a, **kw):
        current_group.append(object_group_id)
        transfer_calls.append((object_group_id, list(memory_objs)))
        if staged:
            for obj in memory_objs:
                assert 0 <= obj.meta.address < 256, obj.meta.address  # bounce
        return real_transfer(
            cache_context, block_ids_gpu, memory_objs, object_group_id, *a, **kw
        )

    monkeypatch.setattr(mod, "transfer_kv_per_object_group", transfer)

    device_ops = MagicMock()
    device_ops.direct_copy_format_supported.return_value = True
    device_ops.DirectCopyGroupSpec.side_effect = lambda *a: tuple(a)
    device_ops.DirectCopyObject.side_effect = lambda *a: tuple(a)
    device_ops.KernelGroupSpec.side_effect = lambda *a: tuple(a)
    device_ops.LaunchVar.side_effect = lambda *a: tuple(a)
    device_ops.BatchStep.side_effect = lambda staging, launches: (staging, launches)

    def direct_capture(direction, device, align, specs, objs):
        group = current_group[-1]
        kgs = KG_OF_GROUP[group]
        for host_ptr, host_offset, nbytes, chunk_idx, skip_blocks in objs:
            per_kg = []
            for pos, kg in enumerate(kgs):
                ids = list(specs[pos][5])
                b = BPW[kg]
                per_kg.append(
                    (tuple(ids[chunk_idx * b : (chunk_idx + 1) * b]), skip_blocks[pos])
                )
            plan.append((group, _value(host_ptr), nbytes, tuple(per_kg)))

    def kernel_capture(direction, device, align, specs, steps, **_):
        group = current_group[-1]
        kgs = KG_OF_GROUP[group]
        for staging, launches in steps:
            per_chunk: list[list] = [[] for _ in staging]
            for spec_idx, start, total, batch_len, skip in launches:
                kg = kgs[spec_idx]
                b = BPW[kg]
                assert total == batch_len * b == len(staging) * b
                ids = _resolve_ids(
                    staged_tensors, specs[spec_idx][5], specs[spec_idx][6]
                )
                ids = ids[start : start + total]
                for i in range(batch_len):
                    per_chunk[i].append(
                        (tuple(ids[i * b : (i + 1) * b]), min(b, max(0, skip - i * b)))
                    )
            for (value, _offset, nbytes), kg_plan in zip(
                staging, per_chunk, strict=True
            ):
                plan.append((group, value, nbytes, tuple(kg_plan)))

    device_ops.execute_direct_copy_transfer.side_effect = direct_capture
    device_ops.execute_object_group_transfer.side_effect = kernel_capture
    monkeypatch.setattr(planner, "device_ops", device_ops)
    monkeypatch.setattr(planner, "_HAS_BATCH_MEMCPY_ASYNC", path == "direct")
    monkeypatch.setattr(planner, "_HAS_NATIVE_OBJECT_GROUP_TRANSFER", path == "kernel")
    monkeypatch.setattr(
        planner,
        "build_staging_copies",
        lambda batch, buffers, is_h2d: [
            (_value(mo.data_ptr), mo.meta.address, mo.get_size()) for mo in batch
        ],
    )

    gpu_block_ids = [list(range(10, 20)), list(range(20, 30))]
    handle, ok = module.retrieve(
        SimpleNamespace(request_id="r", cache_salt=None),
        1,
        gpu_block_ids,
        b"producer",
        skip_first_n_tokens=skip_first_n_tokens,
    )
    assert ok is True
    if pool is not None:
        status = pool.report_status()
        assert status["slots_in_use"] == 0 and status["quarantined_slots"] == 0
        pool.close()
        storage.finish_read_prefetched.assert_any_call(keys[0])
        storage.finish_read_prefetched.assert_any_call(
            keys[1][NUM_CHUNKS - WINDOW[1] :]
        )
    return sorted(plan), transfer_calls


def _effective(plan: list[PlanRecord]) -> list[PlanRecord]:
    """Keep chunks that receive bytes in at least one kernel group."""
    return [
        record
        for record in plan
        if any(
            skip < BPW[kg]
            for (_, skip), kg in zip(record[3], KG_OF_GROUP[record[0]], strict=True)
        )
    ]


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("path", ["direct", "kernel"])
@pytest.mark.parametrize("skip", [0, 6, 14, 20])
def test_staged_plan_equals_direct_plan(
    monkeypatch: pytest.MonkeyPatch, path: str, skip: int, batch: int
) -> None:
    direct, direct_calls = _run(monkeypatch, False, path, skip, batch)
    staged, staged_calls = _run(monkeypatch, True, path, skip, batch)
    # Same per-chunk (object bytes, block ids, skip blocks) plan for every
    # chunk that receives bytes. The staged path batches by slot packing (2 per
    # slot) and the direct path by max_batch_size; the kernel planner stages a
    # whole batch when any token of it is unskipped, so fully skipped chunks may
    # appear in one plan and not the other (they write nothing). The staged
    # path never stages more than the direct path.
    assert _effective(staged) == _effective(direct)
    assert set(staged) <= set(direct)
    assert [g for g, _ in staged_calls] == [0, 0, 0, 1]
    assert [len(objs) for _, objs in staged_calls] == [2, 2, 1, 2]
    assert [g for g, _ in direct_calls] == [0, 1]
    assert all(all(o is not None for o in objs) for _, objs in staged_calls)
    # Chunks that actually receive bytes: group 0 chunks >= first chunk with
    # unskipped tokens, group 1 only its two window chunks (3, 4).
    written = {(g, v - 10 * g - 1) for g, v, _, _ in _effective(direct)}
    first_chunk = skip // CHUNK
    expected = {(0, c) for c in range(first_chunk, NUM_CHUNKS)} | {
        (1, c) for c in range(max(3, first_chunk), NUM_CHUNKS)
    }
    assert written == expected
    if skip == 6:  # chunk 1 keeps one of its two blocks in the full group
        assert any(v == 2 and per[0][1] == 1 for g, v, _, per in direct if g == 0)
