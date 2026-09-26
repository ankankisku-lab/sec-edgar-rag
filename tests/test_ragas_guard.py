import asyncio

import pytest

from src.evaluation import ragas_eval
from src.evaluation.ragas_eval import BudgetExhausted, CallGuard

PER_MINUTE = ("Error code: 429 - Rate limit reached for model `llama-3.3-70b-versatile` on tokens per "
              "minute (TPM): Limit 12000, Used 11500, Requested 3200. Please try again in 7.5s.")
PER_DAY = ("Error code: 429 - Rate limit reached for model `llama-3.3-70b-versatile` on tokens per "
           "day (TPD): Limit 100000, Used 99000, Requested 3000. Please try again in 14m2.5s.")


class FakeLLM:
    def __init__(self, errors):
        self.errors, self.n = list(errors), 0

    async def agenerate(self, *a, **kw):
        self.n += 1
        if self.errors:
            raise RuntimeError(self.errors.pop(0))
        return "ok"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def fake_sleep(_):
        return None
    monkeypatch.setattr(ragas_eval.asyncio, "sleep", fake_sleep)


def test_per_minute_limit_waits_then_succeeds():
    llm = FakeLLM([PER_MINUTE, PER_MINUTE])
    guard = CallGuard(llm, rpm=6000, max_calls=10, max_waits=5)
    assert asyncio.run(llm.agenerate("p")) == "ok"
    assert guard.waits == 2 and guard.calls == 3 and guard.stopped is None


def test_daily_limit_stops_cleanly():
    llm = FakeLLM([PER_DAY])
    guard = CallGuard(llm, rpm=6000, max_calls=10, max_waits=5)
    with pytest.raises(BudgetExhausted):
        asyncio.run(llm.agenerate("p"))
    assert "daily" in guard.stopped
    with pytest.raises(BudgetExhausted):          # nothing more is sent after a stop
        asyncio.run(llm.agenerate("p"))
    assert guard.calls == 1


def test_call_cap_is_hard():
    llm = FakeLLM([])
    guard = CallGuard(llm, rpm=6000, max_calls=2, max_waits=5)
    asyncio.run(llm.agenerate("p"))
    asyncio.run(llm.agenerate("p"))
    with pytest.raises(BudgetExhausted):
        asyncio.run(llm.agenerate("p"))
    assert guard.calls == 2


def test_other_errors_are_not_swallowed():
    llm = FakeLLM(["Error code: 404 - model not found"])
    CallGuard(llm, rpm=6000, max_calls=5, max_waits=5)
    with pytest.raises(RuntimeError, match="404"):
        asyncio.run(llm.agenerate("p"))
