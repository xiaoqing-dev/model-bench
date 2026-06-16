from modelbench.funnel import run_funnel
from modelbench.schema import Case, PromptVersion
from tests.conftest import FakeBench


async def test_funnel_promotes_top_k_and_deep_runs_survivors_only():
    models = ["GOODm", "m2", "m3", "m4"]
    prompts = [PromptVersion("v1", "write")]
    cases = [Case("c1", {}), Case("c2", {})]
    client = FakeBench(judge_model="judge", token="GOOD")  # GOODm wins every pair

    out = await run_funnel(
        prompts, models, cases, client, "judge", "rubric", axis="model", top_k=2
    )

    assert out["promoted"][0] == "GOODm"
    assert len(out["promoted"]) == 2

    # round 2 only runs the survivors
    deep_models = {r.model for r in out["round2"]["results"]}
    assert deep_models <= set(out["promoted"])

    # scout used a subset (1 of 2 cases); deep used all
    assert {r.case_id for r in out["round1"]["results"]} == {"c1"}
    assert {r.case_id for r in out["round2"]["results"]} == {"c1", "c2"}


async def test_funnel_prompt_axis_culls_variants():
    # axis="prompt": compare system-prompt variants on one model
    prompts = [
        PromptVersion("good", "reply", system="GOOD warm style"),
        PromptVersion("v2", "reply", system="plain"),
        PromptVersion("v3", "reply", system="plain"),
    ]
    # FakeBench echoes the user prompt only ("reply"), so encode the winner in the
    # user template instead, to drive the judge deterministically.
    prompts = [
        PromptVersion("good", "GOOD reply"),
        PromptVersion("v2", "reply two"),
        PromptVersion("v3", "reply three"),
    ]
    cases = [Case("c1", {}), Case("c2", {})]
    client = FakeBench(judge_model="judge", token="GOOD")

    out = await run_funnel(
        prompts, ["m1"], cases, client, "judge", "rubric", axis="prompt", top_k=2
    )
    assert out["promoted"][0] == "good"
    deep_prompts = {r.prompt_id for r in out["round2"]["results"]}
    assert deep_prompts <= set(out["promoted"])
