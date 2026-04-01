"""
cli.py
------
The CLI entry point. Uses `typer` to define commands and options.
Wires together tokenizer → pricing → display.
"""

import sys
import typer
from typing import Optional
from llm_cost.pricing import estimate_all_models, get_cheapest
from llm_cost.display import (
    print_results_table,
    print_token_breakdown,
    print_json,
    print_error,
)
from llm_cost.tokenizer import get_token_strings

app = typer.Typer(
    name="llm-cost",
    help="⚡ Estimate LLM token counts and API costs before you spend.",
    add_completion=False,
)


@app.command()
def estimate(
    text: Optional[str] = typer.Argument(
        None,
        help="Text to analyze. If omitted, reads from --file or stdin."
    ),
    file: Optional[str] = typer.Option(
        None, "--file", "-f",
        help="Path to a text file to analyze."
    ),
    models: Optional[str] = typer.Option(
        None, "--models", "-m",
        help="Comma-separated list of models to compare. Default: all."
    ),
    output_format: str = typer.Option(
        "table", "--format",
        help="Output format: 'table' (default) or 'json'."
    ),
    show_tokens: bool = typer.Option(
        False, "--show-tokens",
        help="Show a visual breakdown of how the text is tokenized."
    ),
):
    """
    Estimate token count and cost for a given text across LLM models.

    Examples:\n
      llm-cost estimate "Hello world"\n
      llm-cost estimate --file prompt.txt\n
      cat doc.txt | llm-cost estimate\n
      llm-cost estimate "My prompt" --models gpt-4o,gpt-3.5-turbo\n
      llm-cost estimate "My prompt" --format json
    """

    # ── 1. Get the input text ──────────────────────────────────────────────
    if text:
        input_text = text
    elif file:
        try:
            with open(file, "r", encoding="utf-8") as f:
                input_text = f.read()
        except FileNotFoundError:
            print_error(f"File not found: {file}")
            raise typer.Exit(1)
    elif not sys.stdin.isatty():
        # Text is being piped in via stdin
        input_text = sys.stdin.read()
    else:
        print_error("No input provided. Pass text, use --file, or pipe input.")
        raise typer.Exit(1)

    if not input_text.strip():
        print_error("Input text is empty.")
        raise typer.Exit(1)

    # ── 2. Parse model filter ──────────────────────────────────────────────
    selected_models = [m.strip() for m in models.split(",")] if models else None

    # ── 3. Run estimation ──────────────────────────────────────────────────
    try:
        results = estimate_all_models(input_text, selected_models)
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)
    except FileNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)

    cheapest = get_cheapest(results)

    # ── 4. Output ──────────────────────────────────────────────────────────
    if output_format == "json":
        print_json(results)
    else:
        print_results_table(results, cheapest.model_name)

        # Optional token breakdown (uses cl100k_base as reference)
        if show_tokens:
            token_strings = get_token_strings(input_text, "cl100k_base")
            print_token_breakdown(input_text, token_strings)


@app.command()
def models():
    """List all available models and their pricing."""
    from llm_cost.pricing import load_pricing
    from rich.table import Table
    from rich.console import Console
    from rich import box

    config = load_pricing()
    console = Console()

    table = Table(title="📋 Available Models", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Model",          style="bold white")
    table.add_column("Encoding",       style="dim")
    table.add_column("Input / 1M",     justify="right")
    table.add_column("Output / 1M",    justify="right")
    table.add_column("Context Window", justify="right")

    for name, cfg in config["models"].items():
        table.add_row(
            name,
            cfg["encoding"],
            f"${cfg['input']:.2f}",
            f"${cfg['output']:.2f}",
            f"{cfg['context_window']:,} tokens",
        )

    console.print()
    console.print(table)
    console.print("  [dim]Update pricing in config/pricing.yaml[/dim]\n")


def main():
    app()


if __name__ == "__main__":
    main()