"""
pricing.py
----------
Loads the pricing config and computes costs given a token count.
Also runs tokenization across all models for comparison.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass
from llm_cost.tokenizer import count_tokens


CONFIG_PATH = Path(__file__).parent.parent / "config" / "pricing.yaml"


@dataclass
class ModelResult:
    model_name: str
    encoding: str
    token_count: int
    input_cost: float
    output_cost_per_1k: float
    context_window: int
    context_used_pct: float


def load_pricing() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Pricing config not found at {CONFIG_PATH}. "
            "Make sure config/pricing.yaml exists."
        )
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def compute_cost(token_count: int, price_per_million: float) -> float:
    return (token_count / 1_000_000) * price_per_million


def estimate_all_models(text: str, selected_models: list | None = None) -> list:
    config = load_pricing()
    models = config.get("models", {})

    if selected_models:
        unknown = set(selected_models) - set(models.keys())
        if unknown:
            raise ValueError(f"Unknown model(s): {', '.join(unknown)}. "
                             f"Available: {', '.join(models.keys())}")
        models = {k: v for k, v in models.items() if k in selected_models}

    results = []
    for model_name, model_config in models.items():
        encoding     = model_config["encoding"]
        input_price  = model_config["input"]
        output_price = model_config["output"]
        ctx_window   = model_config["context_window"]

        token_count        = count_tokens(text, encoding)
        input_cost         = compute_cost(token_count, input_price)
        output_cost_per_1k = compute_cost(1000, output_price)
        ctx_pct            = (token_count / ctx_window) * 100

        results.append(ModelResult(
            model_name=model_name,
            encoding=encoding,
            token_count=token_count,
            input_cost=input_cost,
            output_cost_per_1k=output_cost_per_1k,
            context_window=ctx_window,
            context_used_pct=ctx_pct,
        ))

    return results


def get_cheapest(results: list) -> ModelResult:
    return min(results, key=lambda r: r.input_cost)