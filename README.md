# model-bench

Compare **many models on one prompt**, or **many prompt versions on one model**,
scored by an LLM judge. One subscription (OpenRouter), one config file.

## Why these design choices

- **OpenRouter** as the only call layer — one key reaches every model by slug, no
  per-provider SDKs or subscriptions.
- **One matrix, two uses.** Everything is `(prompt_version, model, case) -> output`.
  Hold an axis fixed and you get model comparison or prompt A/B from the same code.
- **Pairwise judge with a swap test.** LLMs compare more reliably than they score
  1–10. Each pair is judged in both orders; a side only wins if it wins both, so
  position bias is neutralised even when the judge is biased.
- **The judge must be validated** before you trust it (`calibrate`).

## Install

```bash
cd model-bench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ",ui" later for the Streamlit app
export OPENROUTER_API_KEY=sk-or-...
```

## Run

```bash
python -m modelbench run experiments/example.yaml
```

A/B test prompt versions: set `axis: prompt`, list multiple `prompts` and one model.

## Trust it — the testing story

Two layers, because LLM calls aren't deterministic:

**Layer A — logic (offline, no key, repeatable).** Run anytime:

```bash
pytest
```

Covers: matrix runs every cell; one model failing doesn't sink the batch; missing
template vars surface as errors; swap-test combination math; identical inputs →
tie; a position-biased judge is neutralised to a tie; win-rate aggregation.

**Layer B — judge calibration (needs a key, costs a little).** Validate the judge
against pairs you labelled yourself before trusting any leaderboard:

```bash
python -m modelbench calibrate experiments/labelled.example.yaml --judge openai/gpt-5
```

Reports `agreement` (judge vs your labels) and `position_consistency`. If
agreement < ~80%, fix the rubric or switch judge model — a miscalibrated judge is
worse than none.

## Layout

```
modelbench/
  schema.py       # PromptVersion / Case / RunResult
  templating.py   # {{var}} rendering
  client.py       # OpenRouter (OpenAI-compatible) async client + Client protocol
  matrix.py       # concurrent (prompt x model x case) execution, failure-isolated
  judge.py        # pairwise + swap test + round-robin
  aggregate.py    # win-rate leaderboard
  config.py       # load + run an experiment YAML
  calibrate.py    # judge validation against labelled pairs
experiments/      # your configs, rubrics, labelled sets, result archives
tests/            # offline test suite (FakeModelClient, ScriptedJudge)
```

## Reuse from another project (e.g. Writing Agent)

```bash
pip install -e /path/to/model-bench
```

```python
from modelbench import run_matrix, judge_matrix, win_rates, OpenRouterClient
```

## Notes

- Model slugs and pricing change — fetch the live catalog with
  `OpenRouterClient().list_models()` instead of trusting the slugs in `example.yaml`.
- For a real run, the judge model must **not** be one of the candidates.
- Pairwise scoring on subjective writing has length/position bias even after the
  swap test. Treat the leaderboard as a coarse filter; eyeball the outputs too.
