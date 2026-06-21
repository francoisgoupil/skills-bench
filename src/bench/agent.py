"""The benchmark 'cell': run a (model + skills) agent on a dataset and return a fitted
scikit-learn estimator, plus the token usage and USD cost it took to produce it.

Provider-agnostic via LiteLLM: one OpenAI-format loop drives Anthropic, Google, Mistral,
DeepSeek/Qwen, open-source (Ollama/vLLM/Together/Fireworks), etc. Model is just a slug:

    anthropic/claude-opus-4-6      openrouter/qwen/qwen-2.5-72b-instruct
    openrouter/deepseek/deepseek-chat       ollama/llama3.1

Cost is computed per call by litellm.completion_cost() against its live price list, so you
do not maintain a price table. For models litellm cannot price (brand-new or local), set
`price_override` on the config (USD per 1M in/out) or leave it None to record 0.
"""
from __future__ import annotations

import glob
import io
import json
import os
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.base import BaseEstimator

from .datasets import Dataset

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code in a sandbox with X, y, pandas, numpy, sklearn, and skrub. "
                "Use only standard sklearn/skrub components — no custom classes. "
                "Assign your fitted estimator to `pipeline`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source to execute"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_pipeline",
            "description": "Submit the fitted `pipeline` when training is complete.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass
class BenchmarkConfig:
    """A column in the benchmark matrix: a model, optionally augmented with skills."""
    key: str                              # e.g. "gemini-2.5-pro+skills0.3"
    model: str                            # litellm slug, e.g. "openrouter/qwen/qwen-2.5-72b-instruct"
    use_skills: bool = False
    skills_dir: str = "skills"
    max_steps: int = 25                   # cap the agent loop
    price_override: tuple[float, float] | None = None  # (in_per_mtok, out_per_mtok)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def tokens_burned(self) -> int:
        """Total tokens spent to produce the pipeline (prompt + completion, all steps)."""
        return self.input_tokens + self.output_tokens

    def add_response(self, response, override: tuple[float, float] | None) -> None:
        u = response.usage
        self.input_tokens += getattr(u, "prompt_tokens", 0) or 0
        self.output_tokens += getattr(u, "completion_tokens", 0) or 0
        self.cost_usd += _response_cost(response, u, override)


def _response_cost(response, usage, override) -> float:
    # OpenRouter returns exact USD cost when usage.include=true (stored in hidden params).
    try:
        from litellm.cost_calculator import get_response_cost_from_hidden_params

        hidden = getattr(response, "_hidden_params", None)
        if hidden:
            provider_cost = get_response_cost_from_hidden_params(hidden)
            if provider_cost is not None:
                return float(provider_cost)
    except Exception:
        pass

    try:
        from litellm import completion_cost

        cost = completion_cost(completion_response=response)
        if cost:
            return float(cost)
    except Exception:
        pass

    # litellm model price table (works for some openrouter/* slugs).
    try:
        from litellm import get_model_info

        model = getattr(response, "model", None)
        if model:
            info = get_model_info(model)
            inp = info.get("input_cost_per_token") or 0.0
            out = info.get("output_cost_per_token") or 0.0
            est = usage.prompt_tokens * inp + usage.completion_tokens * out
            if est:
                return est
    except Exception:
        pass

    if override is not None:
        return (
            usage.prompt_tokens / 1e6 * override[0]
            + usage.completion_tokens / 1e6 * override[1]
        )

    return 0.0


@dataclass
class AgentRun:
    estimator: BaseEstimator              # fitted, ready for skore to score
    usage: Usage
    config: BenchmarkConfig
    notes: str = ""


def _load_skills(skills_dir: str) -> str:
    parts = []
    for path in sorted(glob.glob(os.path.join(skills_dir, "**", "SKILL.md"), recursive=True)):
        parts.append(f"# --- {path} ---\n{Path(path).read_text()}")
    return "\n\n".join(parts)


# Fixed across all configs on a dataset — only +skills appends SKILL.md content.
AGENT_USER_PROMPT = "Build and fit a pipeline for this dataset."

PIPELINE_RULES = """
Pipeline rules (mandatory — pipelines that break these are rejected):
- Use ONLY scikit-learn and skrub components (e.g. skrub.tabular_pipeline, sklearn.pipeline.Pipeline,
  sklearn.preprocessing.*, sklearn.impute.*, sklearn.compose.*, sklearn.ensemble.*, sklearn.linear_model.*).
- Do NOT define custom classes, lambdas, or FunctionTransformer(callable=...).
- Prefer skrub.tabular_pipeline(model) for tabular data; otherwise compose with sklearn.pipeline.Pipeline.
- Assign the fitted estimator to `pipeline`.
"""

_ALLOWED_MODULE_PREFIXES = ("sklearn.", "skrub.")


def _validate_pipeline(estimator) -> str | None:
    """Reject pipelines that cannot be pickled or use non-sklearn/skrub steps."""
    import pickle
    from sklearn.pipeline import Pipeline

    def check_step(step, path: str) -> str | None:
        mod = type(step).__module__
        if not mod.startswith(_ALLOWED_MODULE_PREFIXES):
            return f"{path}: {type(step).__name__} from {mod!r} — use sklearn/skrub only"
        if isinstance(step, Pipeline):
            for i, (_, sub) in enumerate(step.steps):
                if err := check_step(sub, f"{path}[{i}]"):
                    return err
        elif hasattr(step, "steps"):  # FeatureUnion, ColumnTransformer, etc.
            for i, sub in enumerate(step.steps if isinstance(step.steps, list) else []):
                name, trans = sub if isinstance(sub, tuple) else (str(i), sub)
                if err := check_step(trans, f"{path}.{name}"):
                    return err
            if hasattr(step, "transformers"):
                for name, trans, _ in step.transformers:
                    if trans != "drop" and trans != "passthrough":
                        if err := check_step(trans, f"{path}.{name}"):
                            return err
        return None

    if err := check_step(estimator, "pipeline"):
        return err
    try:
        pickle.dumps(estimator)
    except Exception as exc:
        return f"pipeline is not picklable: {exc}"
    return None


def _reject_custom_classes(code: str) -> str | None:
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return f"custom class {node.name!r} is not allowed — use sklearn/skrub only"
    return None


def build_agent_prompt(dataset: Dataset, *, use_skills: bool, skills_dir: str = "skills") -> str:
    """Same base prompt for every model; skills are the only intentional addition."""
    system = (
        "You are a data scientist. Build the best scikit-learn pipeline you can for the "
        "given tabular dataset.\n\n"
        f"Dataset: {dataset.name}\n"
        f"Task: {dataset.task}\n"
        f"Shape: X={dataset.X.shape}, y={dataset.y.shape}\n"
        f"Target dtype: {dataset.y.dtype}\n\n"
        "Use the `run_python` tool to write and execute code. Variables `X` and `y` are "
        "already loaded. Assign your fitted estimator to `pipeline`. When satisfied, call "
        f"`submit_pipeline`.\n{PIPELINE_RULES}"
    )
    if use_skills:
        skills = _load_skills(skills_dir)
        if skills:
            system += "\n\nFollow these skills exactly:\n" + skills
    return system


class _Sandbox:
    """Execute agent Python with dataset bindings; track a fitted `pipeline`."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.pipeline: BaseEstimator | None = None

    def run(self, code: str) -> str:
        import numpy as np
        import pandas as pd
        import skrub
        from sklearn import compose, ensemble, impute, linear_model, metrics, model_selection, pipeline, preprocessing

        namespace = {
            "X": self.dataset.X,
            "y": self.dataset.y,
            "dataset": self.dataset,
            "pd": pd,
            "np": np,
            "skrub": skrub,
            "pipeline": None,
            "Pipeline": pipeline.Pipeline,
            "compose": compose,
            "linear_model": linear_model,
            "ensemble": ensemble,
            "preprocessing": preprocessing,
            "impute": impute,
            "model_selection": model_selection,
            "metrics": metrics,
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            if err := _reject_custom_classes(code):
                return f"error: {err}"
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(code, namespace)  # noqa: S102 — intentional agent sandbox
            candidate = namespace.get("pipeline")
            if candidate is not None and hasattr(candidate, "fit"):
                if err := _validate_pipeline(candidate):
                    return f"error: {err}\nRebuild using skrub.tabular_pipeline or sklearn.pipeline.Pipeline only."
                self.pipeline = candidate
            out = stdout.getvalue().strip()
            err = stderr.getvalue().strip()
            status = "ok"
            if self.pipeline is not None and hasattr(self.pipeline, "predict"):
                status = "pipeline ready"
            parts = [f"status: {status}"]
            if out:
                parts.append(f"stdout:\n{out}")
            if err:
                parts.append(f"stderr:\n{err}")
            if self.pipeline is not None:
                parts.append(f"pipeline: {type(self.pipeline).__name__}")
            return "\n".join(parts)
        except Exception:
            return f"error:\n{traceback.format_exc()}"


class Agent:
    """One executor for every provider. Drives an OpenAI-format tool-use loop via LiteLLM."""

    def __init__(self, config: BenchmarkConfig, *, strict: bool = True):
        self.config = config
        self.strict = strict

    def build_estimator(self, dataset: Dataset) -> AgentRun:
        usage = Usage()
        if self.config.model == "skrub-floor":
            return AgentRun(
                estimator=self._baseline(dataset), usage=usage,
                config=self.config, notes="skrub reference pipeline (no LLM)",
            )

        system = build_agent_prompt(
            dataset, use_skills=self.config.use_skills, skills_dir=self.config.skills_dir,
        )
        estimator = self._run_agent_loop(dataset, system, usage)
        return AgentRun(estimator=estimator, usage=usage, config=self.config)

    def _run_agent_loop(self, dataset: Dataset, system: str, usage: Usage) -> BaseEstimator:
        from litellm import completion

        sandbox = _Sandbox(dataset)
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": AGENT_USER_PROMPT},
        ]

        for _ in range(self.config.max_steps):
            resp = completion(
                model=self.config.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=4096,
            )
            usage.add_response(resp, self.config.price_override)
            msg = resp.choices[0].message
            assistant_msg: dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            if not msg.tool_calls:
                if sandbox.pipeline is not None:
                    if err := _validate_pipeline(sandbox.pipeline):
                        raise RuntimeError(err)
                    return sandbox.pipeline
                continue

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name == "run_python":
                    result = sandbox.run(args.get("code", ""))
                elif name == "submit_pipeline":
                    if sandbox.pipeline is None:
                        result = "error: no fitted pipeline yet — use run_python first"
                    elif err := _validate_pipeline(sandbox.pipeline):
                        result = f"error: {err}"
                    else:
                        return sandbox.pipeline
                else:
                    result = f"error: unknown tool {name!r}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        if sandbox.pipeline is not None:
            if err := _validate_pipeline(sandbox.pipeline):
                raise RuntimeError(err)
            return sandbox.pipeline
        msg = (
            f"Agent stopped after {self.config.max_steps} steps without submitting a pipeline "
            f"for {self.config.model!r}"
        )
        if self.strict:
            raise RuntimeError(msg)
        return self._baseline(dataset)

    @staticmethod
    def _baseline(dataset: Dataset) -> BaseEstimator:
        """skrub one-liner — also a sensible 'no-skills floor' to compare against."""
        import skrub
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingRegressor
        if dataset.task == "binary-classification":
            model = skrub.tabular_pipeline(LogisticRegression(max_iter=1000))
        else:
            model = skrub.tabular_pipeline(HistGradientBoostingRegressor())
        return model.fit(dataset.X, dataset.y)
