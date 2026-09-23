"""Optional local embedding backend; model weights load only on first embed call."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from ..errors import CapabilityError


class SentenceTransformerBackend:
    def __init__(self, model_name: str, *, revision: str | None = None) -> None:
        if not model_name:
            raise ValueError("model_name cannot be empty")
        self.model_name = model_name
        self.revision = revision
        self._model: Any = None
        self._lock = asyncio.Lock()
        self._jobs: set[asyncio.Task[list[list[float]]]] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def backend_id(self) -> str:
        return "sentence-transformers"

    @property
    def cache_identity(self) -> str:
        return f"sentence-transformers:{self.model_name}:{self.revision or 'unpinned'}"

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return await self._embed(texts, task="generic")

    async def embed_query(self, query: str) -> Sequence[float]:
        vectors = await self._embed((query,), task="query")
        return vectors[0]

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return await self._embed(texts, task="document")

    async def _embed(self, texts: Sequence[str], *, task: str) -> list[list[float]]:
        if self._closed:
            raise CapabilityError("embedding backend is closed")
        if not texts:
            return []
        job = asyncio.create_task(self._run_embed(tuple(texts), task))
        self._jobs.add(job)
        job.add_done_callback(self._finished_job)
        return await asyncio.shield(job)

    def _finished_job(self, job: asyncio.Task[list[list[float]]]) -> None:
        self._jobs.discard(job)
        if not job.cancelled():
            job.exception()  # retrieve errors after a caller stops waiting

    async def _run_embed(self, texts: tuple[str, ...], task: str) -> list[list[float]]:
        async with self._lock:
            if self._closed:
                raise CapabilityError("embedding backend is closed")
            return await asyncio.to_thread(self._encode, texts, task)

    def _encode(self, texts: tuple[str, ...], task: str) -> list[list[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise CapabilityError(
                    "Install typedrank[embeddings] to use SentenceTransformerBackend"
                ) from exc
            self._model = SentenceTransformer(
                self.model_name, revision=self.revision, trust_remote_code=False
            )
        method = (
            getattr(self._model, "encode_query", self._model.encode)
            if task == "query"
            else getattr(self._model, "encode_document", self._model.encode)
            if task == "document"
            else self._model.encode
        )
        vectors = method(list(texts), convert_to_numpy=True)
        return [[float(value) for value in vector] for vector in vectors]

    async def aclose(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._finish_close())
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        if self._jobs:
            await asyncio.gather(*self._jobs, return_exceptions=True)
        self._model = None
