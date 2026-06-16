from modelbench.matrix import run_matrix
from modelbench.schema import Case, PromptVersion
from tests.conftest import FakeModelClient

PROMPTS = [PromptVersion("v1", "{{topic}}"), PromptVersion("v2", "write about {{topic}}")]
MODELS = ["m1", "m2"]
CASES = [Case("c1", {"topic": "agents"}), Case("c2", {"topic": "evals"})]


async def test_matrix_runs_every_cell():
    client = FakeModelClient()
    results = await run_matrix(PROMPTS, MODELS, CASES, client)
    # 2 prompts x 2 models x 2 cases
    assert len(results) == 8
    assert all(r.ok for r in results)
    assert len(client.calls) == 8
    # every combination present exactly once
    keys = {r.key for r in results}
    assert len(keys) == 8


async def test_matrix_isolates_failure():
    client = FakeModelClient(fail_models={"m2"})
    results = await run_matrix(PROMPTS, MODELS, CASES, client)
    assert len(results) == 8  # batch still complete
    bad = [r for r in results if not r.ok]
    good = [r for r in results if r.ok]
    assert len(bad) == 4 and all(r.model == "m2" for r in bad)
    assert len(good) == 4 and all(r.model == "m1" for r in good)


async def test_matrix_records_metadata():
    client = FakeModelClient()
    results = await run_matrix(PROMPTS, MODELS, CASES, client)
    r = results[0]
    assert r.prompt_tokens == 10 and r.completion_tokens == 5
    assert r.cost_usd == 0.001
    assert r.latency_s is not None and r.latency_s >= 0


async def test_empty_output_is_flagged_not_silently_ok():
    # reasoning-model trap: API "succeeds" but content is blank
    client = FakeModelClient(empty_models={"m2"})
    results = await run_matrix(PROMPTS, MODELS, CASES, client)
    empties = [r for r in results if r.model == "m2"]
    assert len(empties) == 4
    assert all(not r.ok for r in empties)  # surfaced as errors, not dropped
    assert all("max_tokens" in r.error for r in empties)
    assert all(r.model == "m1" for r in results if r.ok)


async def test_matrix_missing_var_becomes_error_not_crash():
    client = FakeModelClient()
    prompts = [PromptVersion("v1", "{{missing}}")]
    results = await run_matrix(prompts, ["m1"], [Case("c1", {})], client)
    assert len(results) == 1
    assert not results[0].ok
    assert "missing" in results[0].error
