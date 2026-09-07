"""One resident pipeline, two bounded concurrent requests, explicit overload."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings
from .pipeline import RAG, Variant


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=12000)


def create_app(settings: Settings, variant: Variant) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.rag = RAG(settings, variant)
        app.state.active = 0
        app.state.pending = set()
        try:
            yield
        finally:
            if app.state.pending:
                await asyncio.gather(*app.state.pending, return_exceptions=True)
            app.state.rag.close()

    app = FastAPI(title="Trust RAG", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ready", "variant": variant.name}

    @app.post("/ask")
    async def ask(request: Question):
        if app.state.active >= 2:
            raise HTTPException(429, "Two queries are already running")
        app.state.active += 1

        async def execute():
            try:
                return await asyncio.to_thread(app.state.rag.ask, request.question)
            finally:
                app.state.active -= 1

        def finished(task):
            app.state.pending.discard(task)
            if not task.cancelled():
                task.exception()  # Consume a failure even if the HTTP client disconnected.

        task = asyncio.create_task(execute())
        app.state.pending.add(task)
        task.add_done_callback(finished)
        try:
            # Cancelling an HTTP request cannot stop its worker thread. Keep its
            # admission slot occupied until the actual RAG call finishes.
            return await asyncio.shield(task)
        except Exception as exc:
            raise HTTPException(502, f"Query failed: {type(exc).__name__}") from None

    return app
