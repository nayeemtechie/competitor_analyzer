import asyncio
from typing import Optional

import pytest

from src.llm.provider import LLMProvider, LLMResponse


class StubManager:
    """Test double for :class:`LLMProviderManager`."""

    def __init__(self, response: LLMResponse):
        self._response = response
        self.calls = []

    async def generate(self, system_prompt, user_prompt, **kwargs):
        self.calls.append((system_prompt, user_prompt, kwargs))
        return self._response


def make_response(content: str, *, success: bool = True, error: Optional[str] = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        provider="stub",
        model="stub-model",
        success=success,
        error=error,
    )


def test_achat_delegates_to_manager_arguments():
    response = make_response("async result")
    manager = StubManager(response)
    provider = LLMProvider(manager=manager)

    result = asyncio.run(
        provider.achat(
            system="system prompt",
            user="user prompt",
            model="model-x",
            temperature=0.5,
            max_tokens=256,
            json_mode=True,
            retry_attempts=2,
        )
    )

    assert result == "async result"
    assert manager.calls == [
        (
            "system prompt",
            "user prompt",
            {
                "model": "model-x",
                "temperature": 0.5,
                "max_tokens": 256,
                "json_mode": True,
                "retry_attempts": 2,
            },
        )
    ]


def test_achat_raises_on_unsuccessful_response():
    manager = StubManager(make_response("", success=False, error="boom"))
    provider = LLMProvider(manager=manager)

    with pytest.raises(RuntimeError) as exc:
        asyncio.run(provider.achat(system="sys", user="user"))

    assert "boom" in str(exc.value)


def test_chat_runs_without_event_loop():
    manager = StubManager(make_response("sync result"))
    provider = LLMProvider(manager=manager)

    result = provider.chat(system="sys", user="user", model="model-y")

    assert result == "sync result"
    assert manager.calls[0][2]["model"] == "model-y"


def test_chat_from_async_context_uses_background_thread():
    manager = StubManager(make_response("threaded"))
    provider = LLMProvider(manager=manager)

    async def _call_chat():
        return provider.chat(system="sys", user="user")

    result = asyncio.run(_call_chat())

    assert result == "threaded"
    assert manager.calls  # call recorded from background execution


def test_chat_raises_on_error():
    manager = StubManager(make_response("", success=False, error="failure"))
    provider = LLMProvider(manager=manager)

    with pytest.raises(RuntimeError):
        provider.chat(system="sys", user="user")
