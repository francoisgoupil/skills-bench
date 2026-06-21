"""Run the benchmark matrix: for each (dataset, config), build an estimator with the
agent, score it with skore, log performance to Hub and cost locally.

    python -m bench.run --datasets churn-prediction fraud-detection \
                        --configs opus-4.6 qwen-72b+skills deepseek+skills
"""
from __future__ import annotations

import argparse

from . import datasets
from .agent import Agent, BenchmarkConfig

# The matrix columns: each config = one (model × skills on/off) agent run on the dataset.
# e.g. opus-4.8 and opus-4.8+skills are two independent cells — same data, different agent.
CONFIGS: dict[str, BenchmarkConfig] = {
    "skrub-floor":             BenchmarkConfig("skrub-floor", "skrub-floor"),  # optional reference, no LLM
    "opus-4.6":                BenchmarkConfig("opus-4.6", "anthropic/claude-opus-4-6"),
    "opus-4.6+skills":         BenchmarkConfig("opus-4.6+skills", "anthropic/claude-opus-4-6", use_skills=True),
    "opus-4.8":                BenchmarkConfig("opus-4.8", "anthropic/claude-opus-4-8"),
    "opus-4.8+skills":         BenchmarkConfig("opus-4.8+skills", "anthropic/claude-opus-4-8", use_skills=True),
    "mistral-large":           BenchmarkConfig("mistral-large", "mistral/mistral-large-latest"),
    "mistral-large+skills":    BenchmarkConfig("mistral-large+skills", "mistral/mistral-large-latest", use_skills=True),
    "qwen-72b":                BenchmarkConfig("qwen-72b", "openrouter/qwen/qwen-2.5-72b-instruct",
                                              price_override=(0.35, 0.40)),
    "qwen-72b+skills":         BenchmarkConfig("qwen-72b+skills", "openrouter/qwen/qwen-2.5-72b-instruct",
                                              use_skills=True, price_override=(0.35, 0.40)),
    "deepseek":                BenchmarkConfig("deepseek", "openrouter/deepseek/deepseek-chat",
                                              price_override=(0.14, 0.28)),
    "deepseek+skills":         BenchmarkConfig("deepseek+skills", "openrouter/deepseek/deepseek-chat",
                                              use_skills=True, price_override=(0.14, 0.28)),
    "llama-3.3-70b":           BenchmarkConfig("llama-3.3-70b", "openrouter/meta-llama/llama-3.3-70b-instruct",
                                              price_override=(0.12, 0.30)),
    "llama-3.3-70b+skills":    BenchmarkConfig("llama-3.3-70b+skills", "openrouter/meta-llama/llama-3.3-70b-instruct",
                                              use_skills=True, price_override=(0.12, 0.30)),
    "llama3.1-local":          BenchmarkConfig("llama3.1-local", "ollama/llama3.1", price_override=(0.0, 0.0)),
    "llama3.1-local+skills":   BenchmarkConfig("llama3.1-local+skills", "ollama/llama3.1", use_skills=True, price_override=(0.0, 0.0)),
}


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True, choices=list(datasets.REGISTRY))
    p.add_argument("--configs", nargs="+", required=True, choices=list(CONFIGS))
    p.add_argument("--project-prefix", default=None,
                   help="Hub project prefix; default is one project per dataset named after the dataset")
    p.add_argument("--baseline-only", action="store_true",
                   help="Force every config through the skrub baseline (pipeline smoke test only)")
    p.add_argument("--allow-fallback", action="store_true",
                   help="On LLM failure, fall back to skrub baseline instead of aborting")
    p.add_argument("--offline", action="store_true",
                   help="Log to local skore project instead of Skore Hub")
    args = p.parse_args()

    if not args.offline:
        import skore
        skore.login()  # authenticate to Skore Hub once

    from .scoring import score_and_log

    for ds_name in args.datasets:
        ds = datasets.load(ds_name)
        for cfg_name in args.configs:
            cfg = CONFIGS[cfg_name]
            hub_project = f"{args.project_prefix}-{ds_name}" if args.project_prefix else ds_name
            print(f"-> {ds_name} x {cfg_name} ({cfg.model})  [hub: {hub_project}]")
            agent = Agent(cfg, strict=not args.allow_fallback)
            if args.baseline_only:
                agent._run_agent_loop = lambda ds, system, usage: agent._baseline(ds)  # noqa: SLF001
            run = agent.build_estimator(ds)
            if cfg_name != "skrub-floor" and not args.baseline_only and not args.allow_fallback:
                if run.usage.tokens_burned == 0:
                    raise RuntimeError(
                        f"{cfg_name}: no tokens burned — model call likely failed; "
                        "fix API keys / provider config or pass --allow-fallback to override"
                    )
            r = score_and_log(run, ds, project_prefix=args.project_prefix, hub=not args.offline)
            warn = f"  (skore: {r['skore_warning']})" if r.get("skore_warning") else ""
            note = f"  [{run.notes}]" if run.notes else ""
            print(f"   logged {r['key']}  cost=${r['cost_usd']:.4f}  "
                  f"tokens={r['tokens_burned']}{warn}{note}")


if __name__ == "__main__":
    main()
