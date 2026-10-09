# SPDX-License-Identifier: Apache-2.0
"""End-to-end GPU test of the direct copy path through the Python planner.

Unlike ``test_direct_copy_transfer_gpu.py`` (which drives the native executor
with hand-built descriptors), this test goes through the public
``transfer_kv_per_object_group`` entry point with a real ``GPUCacheContext``
and real ``LazyMemoryAllocator`` objects, once with the kernel policy and once
with the direct policy, and requires bit-identical results. It covers the
planner's use of the host pointer lists, the kernel-group byte offset inside
the object, the allocator's virtual offsets (objects straddling a 64 MB pin
chunk are split), ``skip_first_n_tokens`` with a batch of 4, and a Kimi-like
1024-token MLA geometry.
"""

# Standard
import random

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_device_type

pytest.importorskip("cupy", reason="GPU cache context tests require cupy")
pytest.importorskip("lmcache.cuda_ops", reason="Requires lmcache.cuda_ops")

# First Party
from lmcache.v1.memory_allocators.lazy_memory_allocator import (  # noqa: E402
    LazyMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryObj  # noqa: E402
from lmcache.v1.multiprocess.object_group_transfer import (  # noqa: E402
    direct_transfer_supported,
    downsample_and_stage_block_ids,
    transfer_kv_per_object_group,
)
from lmcache.v1.platform.devices.cuda.cache_context import (  # noqa: E402
    GPUCacheContext,
)
import lmcache.cuda_ops as cuda_ops  # noqa: E402
import lmcache.lmcache_native as lmcache_native  # noqa: E402

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(torch_device_type != "cuda", reason="Requires the CUDA backend"),
    pytest.mark.skipif(
        not hasattr(cuda_ops, "execute_direct_copy_transfer")
        or not cuda_ops.batch_memcpy_supported(),
        reason="cudaMemcpyBatchAsync unavailable (needs CUDA >= 12.8 build/driver)",
    ),
]

H2D = lmcache_native.TransferDirection.H2D
D2H = lmcache_native.TransferDirection.D2H
_DEVICE = torch.device("cuda:0")
_DTYPE = torch.bfloat16


class _FakeIPCWrapper:
    """Hands ``GPUCacheContext`` a local CUDA tensor (same-process CUDA IPC
    cannot reimport its own handle)."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def close(self) -> None:
        return None


# name -> (per-layer paged shape, expected format, chunk tokens, num chunks)
_CASES = {
    # Kimi-Linear-like: vLLM inflates the MLA page to 1024 tokens, one block
    # per LMCache chunk, 1.1 MB per (layer, block) entry. 16 chunks of 4.7 MB
    # straddle the allocator's 64 MB pin chunk.
    "mla_1024": (
        [20, 1024, 576],
        lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
        1024,
        16,
    ),
    # Ordinary vLLM NHD flash-attention layout, 16 chunks of 16-token blocks.
    "nhd_16": (
        [2, 96, 16, 8, 128],
        lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        256,
        6,
    ),
}
_NUM_LAYERS = 4


def _make_context(name: str) -> tuple[GPUCacheContext, list[torch.Tensor]]:
    shape, expected_fmt, chunk_tokens, _ = _CASES[name]
    tensors = [
        torch.rand(shape, dtype=_DTYPE, device=_DEVICE) for _ in range(_NUM_LAYERS)
    ]
    ctx = GPUCacheContext(
        [_FakeIPCWrapper(t) for t in tensors],  # type: ignore
        lmcache_tokens_per_chunk=chunk_tokens,
    )
    assert ctx.get_engine_kv_format(0) == expected_fmt
    return ctx, tensors


def _allocate_objects(
    allocator: LazyMemoryAllocator, ctx: GPUCacheContext, num_chunks: int, fill: bool
) -> list[MemoryObj]:
    shape, dtype = ctx.get_kernel_group_shape_dtype(ctx.lmcache_tokens_per_chunk, 0)
    objs: list[MemoryObj] = []
    for _ in range(num_chunks):
        obj = allocator.allocate(shape, dtype)
        if obj is None:
            raise RuntimeError("allocator out of space")
        raw = obj.raw_tensor
        assert raw is not None
        view = raw.view(torch.uint8)[: obj.get_size()]
        if fill:
            view.copy_(torch.randint(0, 256, (obj.get_size(),), dtype=torch.uint8))
        else:
            view.zero_()
        objs.append(obj)
    return objs


def _object_bytes(obj: MemoryObj) -> torch.Tensor:
    raw = obj.raw_tensor
    assert raw is not None
    return raw.view(torch.uint8)[: obj.get_size()].clone()


def _run(
    ctx: GPUCacheContext,
    block_ids: list[list[int]],
    objs: list[MemoryObj],
    direction: "lmcache_native.TransferDirection",
    mode: str,
    batch_size: int,
    skip_first_n_tokens: int,
) -> None:
    host_ids = [list(ids) for ids in block_ids]
    with torch.cuda.device(ctx.device), torch.cuda.stream(ctx.stream):
        block_ids_gpu = downsample_and_stage_block_ids(ctx, host_ids)
        assert direct_transfer_supported(ctx, 0, objs, host_ids) is True
        # Withholding the host block ids is what forces the kernel path now
        # that the direct path is taken automatically whenever it can be.
        transfer_kv_per_object_group(
            ctx,
            block_ids_gpu,
            objs,
            object_group_id=0,
            batch_size=batch_size,
            skip_first_n_tokens=skip_first_n_tokens,
            direction=direction,
            transfer_key=f"test-{mode}",
            block_ids_host=host_ids if mode == "direct" else [],
        )
    ctx.stream.synchronize()


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(7)
    torch.manual_seed(7)


@pytest.fixture
def allocator():
    alloc = LazyMemoryAllocator(init_size=256 << 20, final_size=256 << 20)
    yield alloc
    alloc.close()


@pytest.mark.parametrize("name", sorted(_CASES))
def test_store_direct_matches_kernel(name: str, allocator: LazyMemoryAllocator):
    """D2H through the planner: identical object bytes on both paths."""
    ctx, _tensors = _make_context(name)
    _, _, _chunk, num_chunks = _CASES[name]
    blocks_per_chunk = ctx.calculate_num_blocks(ctx.lmcache_tokens_per_chunk, 0)
    block_ids = [random.sample(range(ctx.num_blocks), num_chunks * blocks_per_chunk)]
    try:
        kernel_objs = _allocate_objects(allocator, ctx, num_chunks, fill=False)
        direct_objs = _allocate_objects(allocator, ctx, num_chunks, fill=False)
        _run(ctx, block_ids, kernel_objs, D2H, "kernel", 1, 0)
        _run(ctx, block_ids, direct_objs, D2H, "direct", 1, 0)
        for k, d in zip(kernel_objs, direct_objs, strict=True):
            assert torch.equal(_object_bytes(k), _object_bytes(d))
            assert int(_object_bytes(d).sum()) != 0  # something was copied
        for obj in kernel_objs + direct_objs:
            allocator.free(obj)
    finally:
        ctx.close()


@pytest.mark.parametrize("name", sorted(_CASES))
def test_retrieve_direct_matches_kernel(name: str, allocator: LazyMemoryAllocator):
    """H2D through the planner with batch size 4 and a skipped token prefix:
    identical pages on both paths."""
    ctx, tensors = _make_context(name)
    _, _, chunk, num_chunks = _CASES[name]
    blocks_per_chunk = ctx.calculate_num_blocks(chunk, 0)
    block_size = chunk // blocks_per_chunk
    block_ids = [random.sample(range(ctx.num_blocks), num_chunks * blocks_per_chunk)]
    # Skip one whole chunk plus one block of the next one.
    skip_first_n_tokens = chunk + block_size
    try:
        objs = _allocate_objects(allocator, ctx, num_chunks, fill=True)
        before = [t.clone() for t in tensors]

        _run(
            ctx, block_ids, objs, H2D, "kernel", ctx.max_batch_size, skip_first_n_tokens
        )
        after_kernel = [t.clone() for t in tensors]
        for t, snap in zip(tensors, before, strict=True):
            t.copy_(snap)

        _run(
            ctx, block_ids, objs, H2D, "direct", ctx.max_batch_size, skip_first_n_tokens
        )
        for k, d in zip(after_kernel, tensors, strict=True):
            assert _bitwise_equal(k, d)
        # The skipped prefix was left alone, the rest was written.
        assert any(
            not _bitwise_equal(k, snap)
            for k, snap in zip(after_kernel, before, strict=True)
        )
        for obj in objs:
            allocator.free(obj)
    finally:
        ctx.close()


@pytest.mark.no_shared_allocator
@pytest.mark.parametrize("name", ["mla_1024", "nhd_16"])
@pytest.mark.parametrize("force_kernel", [False, True])
def test_dax_staged_retrieve_matches_direct_bytes_and_skip(
    name, force_kernel, monkeypatch
):
    """Real CPU gather + pinned bounce + GPU planner, with ordinary DRAM input.

    A spin kernel queued ahead of the staged copies keeps their DMA pending
    while the retrieve path returns: the response must not wait for the GPU.
    Device-DAX/peer protection is covered separately; this test requires CUDA
    and deliberately claims no DAX qualification.
    """
    # Standard
    from contextlib import contextmanager
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    # First Party
    from lmcache.v1.distributed.dax_coordinated_l1.staged_retrieve import (
        StagedRetrievePool,
    )
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as transfer_mod
    from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
        LMCacheDrivenTransferModule,
    )
    import lmcache.v1.multiprocess.object_group_transfer as planner

    ctx, tensors = _make_context(name)
    num_chunks = 4
    shape, dtype = ctx.get_kernel_group_shape_dtype(ctx.lmcache_tokens_per_chunk, 0)
    object_bytes = shape.numel() * dtype.itemsize
    # Small registration boundaries exercise the nonzero bounce offset.
    monkeypatch.setattr(LazyMemoryAllocator, "PIN_CHUNK_SIZE", 1 << 20)
    allocator = LazyMemoryAllocator(
        init_size=16 * object_bytes, final_size=16 * object_bytes
    )
    pool = StagedRetrievePool(2, 2 * object_bytes, 2)
    objects = _allocate_objects(allocator, ctx, num_chunks, True)
    ids = [
        list(
            range(
                num_chunks * ctx.calculate_num_blocks(ctx.lmcache_tokens_per_chunk, 0)
            )
        )
    ]
    initial = [tensor.clone() for tensor in tensors]
    skip = ctx.lmcache_tokens_per_chunk + ctx.get_shape_desc(0).bs
    finish = MagicMock()

    @contextmanager
    def read(_keys):
        yield objects

    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._ctx = SimpleNamespace(
        storage_manager=SimpleNamespace(
            read_prefetched_results=read,
            finish_read_prefetched=finish,
        )
    )
    # Real completion events, captured so the test can poll them itself.
    handles: list[torch.cuda.Event] = []
    real_event = transfer_mod.torch_dev.Event

    def make_event() -> torch.cuda.Event:
        handle = real_event()
        handles.append(handle)
        return handle

    monkeypatch.setattr(transfer_mod, "torch_dev", SimpleNamespace(Event=make_event))
    # About half a second of GPU spin ahead of the staged H2D copies; the
    # retrieve path returns long before it, so its completion events must
    # still be pending then. Timing is generous: a failure here means the
    # response waited for the DMA, not a flaky clock.
    clock_khz = torch.cuda.get_device_properties(ctx.device).clock_rate
    sleep_cycles = int(clock_khz * 1000 * 0.5)
    try:
        _run(ctx, ids, objects, H2D, "direct", ctx.max_batch_size, skip)
        expected = [tensor.clone() for tensor in tensors]
        for tensor, original in zip(tensors, initial, strict=True):
            tensor.copy_(original)
        torch.cuda.synchronize()
        if force_kernel:
            monkeypatch.setattr(planner, "_HAS_BATCH_MEMCPY_ASYNC", False)
        with torch.cuda.device(ctx.device), torch.cuda.stream(ctx.stream):
            gpu_ids = downsample_and_stage_block_ids(ctx, ids)
            torch.cuda._sleep(sleep_cycles)
            copied = module._retrieve_staged_group(
                pool,
                ctx,
                list(range(num_chunks)),
                gpu_ids,
                ids,
                0,
                0,
                skip,
                "staged-gpu",
            )
            # Back before the DMA: the readers are released, the slots are
            # in flight and no completion event has fired yet.
            assert len(handles) == 2
            assert not any(handle.query() for handle in handles)
            status = pool.report_status()
            assert status["active_sessions"] == 0
            assert status["in_flight_slots"] == 2
            assert status["slots_in_use"] == 2
        finish.assert_called_once_with(list(range(num_chunks)))
        ctx.stream.synchronize()
        assert all(handle.query() for handle in handles)
        assert copied == num_chunks * object_bytes
        assert all(
            _bitwise_equal(actual, want)
            for actual, want in zip(tensors, expected, strict=True)
        )
        status = pool.report_status()
        assert status["in_flight_slots"] == 0
        assert status["slots_in_use"] == 0
        assert status["event_timeouts"] == 0
        # The second batch could not reuse the first batch's slot, whose DMA
        # was still in flight, so both bounded slots were allocated.
        assert status["allocated_bytes"] == status["capacity_bytes"]
        assert status["capacity_bytes"] == 4 * object_bytes
    finally:
        ctx.stream.synchronize()
        pool.close()
        for obj in objects:
            obj.ref_count_down()
        ctx.close()
        allocator.close()
