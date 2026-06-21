# skills-bench

Benchmark **(model × Probabl skills)** agents on data-science use cases, across **any LLM
provider**, and track **performance** (ML model quality) against **token cost** (USD spent
on the agent run that produced it).

Reproduces the Probabl "Cost efficiency" curves: performance (%) vs task cost, for many
model/skills combinations.

## The core idea

A "Probabl skill" is an **agent skill** (a `SKILL.md` from
https://github.com/probabl-ai/skills), not a model. A benchmark *cell* is one independent agent run — the only thing held constant across
cells on the same row is the **dataset**:

    churn-prediction  ×  opus-4.8           →  agent writes code  →  pipeline A
    churn-prediction  ×  opus-4.8+skills    →  agent writes code  →  pipeline B
    churn-prediction  ×  opus-4.6           →  agent writes code  →  pipeline C
    ...

`opus-4.8` (no skills) is the reference for `opus-4.8+skills` on the same model — not a
shared skrub fallback. Each config must produce its own fitted estimator via the LLM.

and we measure two independent things about it:

| What | Where it comes from |
|------|---------------------|
| **Performance** (AUC, accuracy, ...) | `skore` report on a held-out split |
| **Token cost** (USD) | `litellm` per call → Hub custom metric `cost_usd` |
| **Tokens burned** | total prompt + completion tokens for the agent run → Hub `tokens_burned` |

The **opening prompt is identical** across all models on a dataset; only `+skills` configs
append SKILL.md. Plot **performance vs `cost_usd`** for cost-efficiency; `tokens_burned` is
the raw token budget each config spent to produce its pipeline.

## Multi-provider by design

The agent loop is provider-agnostic via **LiteLLM** — one OpenAI-format `completion()` call
drives 100+ providers, and `litellm.completion_cost()` prices each call for you (no price
table to maintain). A config's `model` is just a slug:

    anthropic/claude-opus-4-6      openrouter/qwen/qwen-2.5-72b-instruct
    openrouter/deepseek/deepseek-chat       mistral/mistral-large-latest
    deepseek/deepseek-chat         openrouter/qwen/qwen-2.5-72b-instruct
    together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo        ollama/llama3.1

Set only the provider API keys you use (see `.env.example`). **OpenRouter** (`OPENROUTER_API_KEY`)
covers Qwen, DeepSeek, Llama, and many open models with a single key. Anthropic models use
`ANTHROPIC_API_KEY` directly. Local models (Ollama) need no key.

## Layout

```
src/bench/
  datasets.py     # registry of use cases (predictive maintenance, fraud, churn, credit)
  agent.py        # provider-agnostic (model + skills) -> fitted estimator + usage/cost
  scoring.py      # skore evaluate + Project(hub) logging + cost sidecar table
  hub_client.py   # thin REST client over the Skore Hub API (monitoring read path)
  run.py          # the benchmark matrix (datasets x configs)
dashboard/app.py  # Streamlit: performance vs cost, from Skore Hub + cost table
skills/           # `npx skills add github.com/probabl-ai/skills` lands SKILL.md files here
```

## Setup & run

```bash
python -m venv .venv && source .venv/bin/activate   # python >= 3.11
pip install -e .
cp .env.example .env                                # fill in the keys you use
npx skills add github.com/probabl-ai/skills         # pull skills into ./skills

# Quick smoke test (no LLM / Hub required):
python -m bench.run --datasets credit-scoring --configs llama3.1-local \
                    --baseline-only --offline

# Hub layout: workspace/benchmark/churn-prediction, workspace/benchmark/fraud-detection, ...
python -m bench.run --datasets churn-prediction \
  --configs opus-4.6 opus-4.6+skills opus-4.8 opus-4.8+skills \
            qwen-72b qwen-72b+skills deepseek deepseek+skills

# Optional: add skrub-floor for a no-LLM reference column
# --configs skrub-floor opus-4.8 opus-4.8+skills ...

streamlit run dashboard/app.py
```

Failed LLM calls abort the run (no silent fallback). Use `--allow-fallback` only for debugging.
Use `--offline` to log metrics locally (to `runs.db`) instead of Skore Hub.
Use `--baseline-only` to smoke-test the scoring pipeline without calling any LLM.


## The one thing left to build

`agent.py._run_agent_loop` drives a LiteLLM tool-use loop with a Python sandbox. Pipelines must
use **standard sklearn/skrub only** (no custom classes) so reports pickle cleanly to Hub.

Also: the OpenML `data_id`s in `datasets.py` are placeholders — point them at your real
benchmark data — and confirm the Skore Hub REST routes/auth in `hub_client.py` against
https://api.skore.probabl.ai/docs.

Note: skore local mode can fail on SQLite 3.49+ (diskcache compatibility). Use Skore Hub
or `--offline` (metrics are stored in `runs.db` for the dashboard).
