import asyncio
import threading

import pytest
from fastapi import HTTPException

from applications.trust_rag import serving
from applications.trust_rag.pipeline import Variant


def test_disconnected_requests_keep_capacity_until_workers_finish(corpus, monkeypatch):
    release = threading.Event()

    class BlockingRAG:
        def __init__(self, *args):
            self.closed = False

        def ask(self, question):
            if not release.wait(10):
                raise TimeoutError("test worker was not released")
            return {"answer": question}

        def close(self):
            self.closed = True

    monkeypatch.setattr(serving, "RAG", BlockingRAG)
    app = serving.create_app(corpus, Variant.stage("B2"))
    endpoint = next(route.endpoint for route in app.routes if route.path == "/ask")

    async def scenario():
        async with app.router.lifespan_context(app):
            requests = [asyncio.create_task(endpoint(serving.Question(question=str(i)))) for i in range(2)]
            try:
                await asyncio.sleep(0)
                assert app.state.active == 2
                requests[0].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await requests[0]
                with pytest.raises(HTTPException) as rejected:
                    await endpoint(serving.Question(question="third"))
                assert rejected.value.status_code == 429
            finally:
                release.set()
                await asyncio.gather(*requests, return_exceptions=True)
                await asyncio.gather(*app.state.pending, return_exceptions=True)
            assert app.state.active == 0
            assert await endpoint(serving.Question(question="next")) == {"answer": "next"}
        assert app.state.rag.closed

    asyncio.run(scenario())
