from dataclasses import dataclass
from typing import Any, Optional, Sequence

import asyncio
import threading
from concurrent.futures import Future as ThreadFuture

import torch

from .core import forward_distributions, prepare_model_for_inference


@dataclass(frozen=True)
class EngineInput:
    tokens: torch.Tensor  # shape: (S,) long
    legal_mask: Optional[torch.Tensor] = None  # shape: (4,) bool
    ctx: Any = None  # user metadata for routing


@dataclass
class EngineItemOutput:
    head_probs: list[torch.Tensor]  # each (n_bins,) for this item
    ctx: Any


class InferenceEngine:
    """Thin wrapper around the model to run batch forwards.

    Responsibilities: move to device/dtype, optional compile, and forward.
    Does not perform selection or masking.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: Optional[torch.device | str] = None,
        prefer_bf16: bool = True,
        compile_mode: Optional[str] = "reduce-overhead",
    ) -> None:
        self.model, self.used_dtype = prepare_model_for_inference(
            model, device=device, prefer_bf16=prefer_bf16, compile_mode=compile_mode
        )

    @torch.inference_mode()
    def predict_batch(self, items: Sequence[EngineInput]) -> list[EngineItemOutput]:
        if len(items) == 0:
            return []
        # Stack tokens to (B, S)
        tokens = torch.stack([it.tokens for it in items], dim=0).to(
            next(self.model.parameters()).device, dtype=torch.long
        )
        head_probs = forward_distributions(self.model, tokens, set_eval=True)
        # Split per item as (n_bins,) per head
        outs: list[EngineItemOutput] = []
        for b in range(tokens.size(0)):
            per_item = [hp[b].detach() for hp in head_probs]
            outs.append(EngineItemOutput(head_probs=per_item, ctx=items[b].ctx))
        return outs

class AsyncBatchPool:
    """Async batching pool that coalesces requests and runs efficient batches.

    Submit returns an asyncio.Future; a single background worker gathers pending
    requests up to `max_batch` or until `flush_interval_ms` elapses, deduplicates
    by token key, serves from cache, runs a single forward for remaining items,
    then fulfills all futures.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        *,
        max_batch: Optional[int] = None,
        flush_interval_ms: float = 1.0,
        static_batch: Optional[int] = None,
        enable_cache: bool = True,
    ) -> None:
        self.engine = engine
        self.max_batch = max_batch
        self.flush_interval_ms = max(0.0, float(flush_interval_ms))
        self.static_batch = static_batch
        self.enable_cache = enable_cache

        self._queue: asyncio.Queue[tuple[EngineInput, "asyncio.Future[EngineItemOutput]"]] = asyncio.Queue()
        self._task: Optional[asyncio.Task[None]] = None
        self._closed = False
        self._cache: dict[tuple[int, ...], EngineItemOutput] = {}

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._worker())

    async def close(self) -> None:
        self._closed = True
        if self._task is not None:
            await self._queue.put((EngineInput(tokens=torch.tensor([], dtype=torch.long)), asyncio.get_running_loop().create_future()))  # sentinel
            try:
                await self._task
            finally:
                self._task = None

    def submit(self, req: EngineInput) -> "asyncio.Future[EngineItemOutput]":
        if self._closed:
            raise RuntimeError("AsyncBatchPool is closed")
        fut: "asyncio.Future[EngineItemOutput]" = asyncio.get_running_loop().create_future()
        self._queue.put_nowait((req, fut))
        return fut

    def _key(self, tokens: torch.Tensor) -> tuple[int, ...]:
        return tuple(int(x) for x in tokens.view(-1).to("cpu", copy=False).tolist())

    async def _worker(self) -> None:
        while not self._closed:
            try:
                first_req, first_fut = await self._queue.get()
            except asyncio.CancelledError:
                break
            # If sentinel: break
            if first_req.tokens.numel() == 0 and self._closed:
                break

            batch: list[tuple[EngineInput, "asyncio.Future[EngineItemOutput]"]] = [(first_req, first_fut)]
            # Timed window to collect more
            deadline = asyncio.get_running_loop().time() + (self.flush_interval_ms / 1000.0)
            while True:
                if self.max_batch is not None and len(batch) >= self.max_batch:
                    break
                timeout = max(0.0, deadline - asyncio.get_running_loop().time())
                if timeout == 0.0 and self.flush_interval_ms > 0.0:
                    break
                try:
                    req, fut = await asyncio.wait_for(self._queue.get(), timeout=timeout if self.flush_interval_ms > 0.0 else 0.0)
                    batch.append((req, fut))
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    return

            # Dedup and serve from cache
            dedup_items: list[EngineInput] = []
            futures_by_index: list[list["asyncio.Future[EngineItemOutput]"]] = []
            index_by_key: dict[tuple[int, ...], int] = {}

            # Serve cached results immediately
            for req, fut in batch:
                k = self._key(req.tokens)
                if self.enable_cache and k in self._cache:
                    cached = self._cache[k]
                    if not fut.done():
                        fut.set_result(
                            EngineItemOutput([t.clone() for t in cached.head_probs], req.ctx)
                        )
                    continue
                idx = index_by_key.get(k)
                if idx is None:
                    index_by_key[k] = len(dedup_items)
                    dedup_items.append(req)
                    futures_by_index.append([fut])
                else:
                    futures_by_index[idx].append(fut)

            if not dedup_items:
                continue

            # Optionally pad to static_batch by duplicating last item
            real_count = len(dedup_items)
            pad_count = 0
            if self.static_batch is not None and real_count < self.static_batch:
                pad_count = self.static_batch - real_count
                last = dedup_items[-1]
                dedup_items.extend([last] * pad_count)

            # Run engine in thread to avoid blocking event loop
            outs = await asyncio.to_thread(self.engine.predict_batch, dedup_items)

            # Only process real outputs (ignore padding)
            outs = outs[:real_count]

            # Populate cache and complete futures
            for i, out in enumerate(outs):
                k = self._key(dedup_items[i].tokens)
                if self.enable_cache:
                    self._cache[k] = EngineItemOutput([t.to("cpu") for t in out.head_probs], ctx=None)
                for fut in futures_by_index[i]:
                    if not fut.done():
                        fut.set_result(EngineItemOutput([t.clone() for t in out.head_probs], dedup_items[i].ctx))


class _LoopThread(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def run(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_coro(self, coro) -> ThreadFuture:
        if self.loop is None:
            raise RuntimeError("Event loop not started")
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


class SyncBatchPool:
    """Synchronous adapter over AsyncBatchPool using a background event loop.

    submit() returns a blocking-handle compatible with .done()/.result().
    """

    class Handle:
        def __init__(self, tfut: ThreadFuture) -> None:
            self._tfut = tfut

        def done(self) -> bool:
            return self._tfut.done()

        def result(self) -> EngineItemOutput:
            return self._tfut.result()

    def __init__(
        self,
        engine: InferenceEngine,
        *,
        max_batch: Optional[int] = None,
        flush_interval_ms: float = 1.0,
        static_batch: Optional[int] = None,
        enable_cache: bool = True,
    ) -> None:
        self._loop_thread = _LoopThread()
        self._loop_thread.start()
        # Build async pool in the background loop
        async def _build_pool() -> AsyncBatchPool:
            pool = AsyncBatchPool(
                engine,
                max_batch=max_batch,
                flush_interval_ms=flush_interval_ms,
                static_batch=static_batch,
                enable_cache=enable_cache,
            )
            await pool.start()
            return pool

        self._pool: AsyncBatchPool = self._loop_thread.run_coro(_build_pool()).result()

    def submit(self, req: EngineInput) -> "SyncBatchPool.Handle":
        async def _submit(req: EngineInput):
            fut = self._pool.submit(req)
            return await fut
        tfut = self._loop_thread.run_coro(_submit(req))
        return SyncBatchPool.Handle(tfut)

    def close(self) -> None:
        # Close async pool and stop loop
        async def _close():
            await self._pool.close()
        self._loop_thread.run_coro(_close()).result()
        if self._loop_thread.loop is not None:
            self._loop_thread.loop.call_soon_threadsafe(self._loop_thread.loop.stop)


__all__ = [
    "EngineInput",
    "EngineItemOutput",
    "InferenceEngine",
    "AsyncBatchPool",
    "SyncBatchPool",
]
