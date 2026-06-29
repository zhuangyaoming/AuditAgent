"""Robustness tests for the checkpoint / failure-handling changes.

Covers the behaviour that protects against an unstable / exhausted API:

  * LLM calls RAISE (classified) on failure instead of silently returning "",
    so a failed case is never persisted as an empty "success".
  * Terminal billing/quota/auth errors (incl. quota carried on HTTP 429) trip a
    global circuit breaker; transient errors (429 rate-limit / 5xx) are retried.
  * CN `_write_sheet` writes atomically and REFUSES to overwrite an existing but
    unreadable workbook (which would wipe sibling sheets).

Run with either:
    pytest tests/test_robustness.py
    python tests/test_robustness.py
No third-party test plugins required (async cases use asyncio.run internally).
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Module loaders (load by path; stub heavy pipeline/eval imports for the CN file)
# ---------------------------------------------------------------------------

def _load_us_base_agent():
    path = REPO_ROOT / "us_pipeline_v2" / "agents" / "base_agent.py"
    spec = importlib.util.spec_from_file_location("us_base_agent_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_cn_module():
    # Stub the heavy imports the CN entry script pulls in at import time.
    for name in ("SingleReportAnalyzer", "CrossReportAnalyzer", "prompt", "llm_judge"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["SingleReportAnalyzer"].SingleAnalyzer = object
    sys.modules["CrossReportAnalyzer"].CrossAnalyzer = object
    for fn in ("construct_aggre_prompt", "construct_cross_report_prompt"):
        setattr(sys.modules["prompt"], fn, lambda *a, **k: "")
    for fn in ("build_gt_for_case", "compute_case_metrics", "convert_gt_to_label_format",
               "get_dedup_stats", "llm_eval_risk_fact"):
        setattr(sys.modules["llm_judge"], fn, lambda *a, **k: {})
    sys.modules["llm_judge"].LLM_CONCURRENCY = 3

    path = REPO_ROOT / "run_accounting_cn_eval_all.py"
    spec = importlib.util.spec_from_file_location("cn_module_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    # The CN script rewraps sys.stdout on win32 (needs .buffer); keep the real
    # stdout during import, then restore so pytest's capture keeps working.
    old_stdout = sys.stdout
    try:
        sys.stdout = sys.__stdout__
        spec.loader.exec_module(mod)
    finally:
        sys.stdout = old_stdout
    return mod


US = _load_us_base_agent()
CN = _load_cn_module()


# ---------------------------------------------------------------------------
# Fake aiohttp session
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status, body, jsondata):
        self.status, self._body, self._json = status, body, jsondata

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def text(self):
        return self._body

    async def json(self):
        return self._json


class FakeSession:
    def __init__(self, status, body="", jsondata=None):
        self.status, self.body, self.jsondata, self.calls = status, body, jsondata, 0

    def post(self, *a, **k):
        self.calls += 1
        return _FakeResp(self.status, self.body, self.jsondata or {})


def _patch_sleep(mod):
    """Replace the module's asyncio.sleep so retry backoff doesn't slow tests."""
    async def fast(_):
        return None
    mod.asyncio.sleep = fast


# ---------------------------------------------------------------------------
# US base_agent._call
# ---------------------------------------------------------------------------

def test_us_terminal_402_raises_and_trips_breaker():
    async def go():
        US.STOP_EVENT = asyncio.Event()
        _patch_sleep(US)
        ag = US.BaseAgent(model_name="deepseek-v4", max_retries=2)
        sess = FakeSession(402, "Insufficient Balance")
        try:
            await ag._call("hi", sess)
        except US.TerminalAPIError:
            return sess.calls, US.STOP_EVENT.is_set()
        raise AssertionError("expected TerminalAPIError")
    calls, stopped = asyncio.run(go())
    assert calls == 1, "terminal error must not retry"
    assert stopped, "terminal error must trip the breaker"


def test_us_429_with_quota_body_is_terminal():
    """The fix the reviewer asked for: a quota error carried on HTTP 429."""
    async def go():
        US.STOP_EVENT = asyncio.Event()
        _patch_sleep(US)
        ag = US.BaseAgent(model_name="deepseek-v4", max_retries=3)
        sess = FakeSession(429, '{"error":{"code":"insufficient_quota","message":"You exceeded your current quota"}}')
        try:
            await ag._call("hi", sess)
        except US.TerminalAPIError:
            return sess.calls, US.STOP_EVENT.is_set()
        raise AssertionError("expected TerminalAPIError for quota-on-429")
    calls, stopped = asyncio.run(go())
    assert calls == 1, "quota 429 must not be retried as rate-limit"
    assert stopped


def test_us_plain_429_is_transient():
    async def go():
        US.STOP_EVENT = asyncio.Event()
        _patch_sleep(US)
        ag = US.BaseAgent(model_name="deepseek-v4", max_retries=2)
        sess = FakeSession(429, "Too Many Requests: slow down")
        try:
            await ag._call("hi", sess)
        except US.CallFailedError:
            return sess.calls, US.STOP_EVENT.is_set()
        raise AssertionError("expected CallFailedError")
    calls, stopped = asyncio.run(go())
    assert calls == 3, "plain rate-limit should retry then fail"
    assert not stopped, "rate-limit must NOT trip the breaker"


def test_us_transient_500_then_callfailed():
    async def go():
        US.STOP_EVENT = asyncio.Event()
        _patch_sleep(US)
        ag = US.BaseAgent(model_name="deepseek-v4", max_retries=2)
        sess = FakeSession(500, "server error")
        try:
            await ag._call("hi", sess)
        except US.CallFailedError:
            return sess.calls, US.STOP_EVENT.is_set()
        raise AssertionError("expected CallFailedError")
    calls, stopped = asyncio.run(go())
    assert calls == 3
    assert not stopped


def test_us_success_and_genuine_empty():
    async def go():
        US.STOP_EVENT = asyncio.Event()
        _patch_sleep(US)
        ag = US.BaseAgent(model_name="deepseek-v4", max_retries=2)
        ok = FakeSession(200, "", {"choices": [{"message": {"content": "hi there"}}],
                                   "usage": {"total_tokens": 10}})
        out = await ag._call("hi", ok, track_phase="single_report")
        # Genuinely empty content is NOT a failure → returns "" without raising.
        empty = FakeSession(200, "", {"choices": [{"message": {"content": ""}}]})
        out2 = await ag._call("hi", empty)
        return out, out2, ag.token_usage
    out, out2, tok = asyncio.run(go())
    assert out == "hi there"
    assert out2 == ""
    assert tok.get("single_report") == 10


# ---------------------------------------------------------------------------
# CN _async_chat
# ---------------------------------------------------------------------------

def test_cn_terminal_and_transient_and_success():
    async def go():
        _patch_sleep(CN)
        results = {}

        CN.STOP_EVENT = asyncio.Event()
        r = CN.CnAgentRunner("deepseek-v4"); r.session = FakeSession(402, "余额不足")
        try:
            await r._async_chat("hi")
        except CN.TerminalAPIError:
            results["terminal"] = (r.session.calls, CN.STOP_EVENT.is_set())

        CN.STOP_EVENT = asyncio.Event()
        r = CN.CnAgentRunner("deepseek-v4"); r.session = FakeSession(503, "unavailable")
        try:
            await r._async_chat("hi")
        except CN.CallFailedError:
            results["transient"] = (r.session.calls, CN.STOP_EVENT.is_set())

        CN.STOP_EVENT = asyncio.Event()
        r = CN.CnAgentRunner("deepseek-v4")
        r.session = FakeSession(200, "", {"choices": [{"message": {"content": "ok"}}],
                                          "usage": {"total_tokens": 7}})
        out = await r._async_chat("hi", track_phase="single_report")
        results["success"] = (out, r.token_usage.get("single_report"))

        # Once tripped, later calls short-circuit without hitting the API.
        CN.STOP_EVENT = asyncio.Event(); CN.STOP_EVENT.set()
        r = CN.CnAgentRunner("deepseek-v4"); r.session = FakeSession(200, "", {"choices": [{"message": {"content": "x"}}]})
        try:
            await r._async_chat("hi")
        except CN.TerminalAPIError:
            results["shortcircuit"] = r.session.calls
        return results

    res = asyncio.run(go())
    assert res["terminal"] == (1, True)
    assert res["transient"][0] == CN.MAX_RETRIES and res["transient"][1] is False
    assert res["success"] == ("ok", 7)
    assert res["shortcircuit"] == 0


# ---------------------------------------------------------------------------
# CN _write_sheet
# ---------------------------------------------------------------------------

def test_cn_write_sheet_preserves_siblings(tmp_path=None):
    import pandas as pd
    base = Path(tmp_path) if tmp_path else REPO_ROOT
    import tempfile
    d = Path(tempfile.mkdtemp())
    xl = d / "out.xlsx"
    CN._write_sheet(xl, "Sheet1", pd.DataFrame([{"CaseId": "1", "X": "a"}]))
    CN._write_sheet(xl, "Eval_Metrics", pd.DataFrame([{"CaseId": "1", "R_I": 0.5}]))
    CN._write_sheet(xl, "Sheet1", pd.DataFrame([{"CaseId": "1", "X": "a"},
                                                {"CaseId": "2", "X": "b"}]))
    sheets = pd.read_excel(xl, sheet_name=None)
    assert set(sheets) == {"Sheet1", "Eval_Metrics"}
    assert list(sheets["Sheet1"]["CaseId"].astype(str)) == ["1", "2"]
    assert float(sheets["Eval_Metrics"]["R_I"].iloc[0]) == 0.5
    # No temp files left behind.
    assert not [p for p in d.iterdir() if p.name.startswith(".cn_tmp_")]


def test_cn_write_sheet_refuses_to_overwrite_unreadable():
    """An existing-but-unreadable workbook must NOT be silently replaced."""
    import pandas as pd
    import tempfile
    d = Path(tempfile.mkdtemp())
    xl = d / "corrupt.xlsx"
    xl.write_bytes(b"this is not a valid xlsx file")  # exists but unreadable
    before = xl.read_bytes()
    raised = False
    try:
        CN._write_sheet(xl, "Sheet1", pd.DataFrame([{"CaseId": "1"}]))
    except Exception:
        raised = True
    assert raised, "must raise instead of overwriting an unreadable file"
    assert xl.read_bytes() == before, "the existing file must be left untouched"
    assert not [p for p in d.iterdir() if p.name.startswith(".cn_tmp_")]


# ---------------------------------------------------------------------------
# Standalone runner (so `python tests/test_robustness.py` works without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
