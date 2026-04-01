"""
display.py
----------
All terminal output lives here using the `rich` library.
Keeping display logic separate from business logic keeps code clean.
"""

import json
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text
from llm_cost.pricing import ModelResult


console = Console()


def print_results_table(results: list[ModelResult], cheapest_model: str) -> None:
    """
    Print a rich formatted table of all model results.
    Highlights the cheapest model in green.
    """
    table = Table(
        title="💰 LLM Cost Estimator",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )

    # Define columns
    table.add_column("Model",            style="bold white",  min_width=16)
    table.add_column("Encoding",         style="dim",         min_width=13)
    table.add_column("Tokens",           justify="right",     min_width=8)
    table.add_column("Input Cost",       justify="right",     min_width=12)
    table.add_column("Output / 1k tok",  justify="right",     min_width=15)
    table.add_column("Context Used",     justify="right",     min_width=13)

    for r in results:
        is_cheapest = r.model_name == cheapest_model

        # Format numbers nicely
        token_str   = f"{r.token_count:,}"
        input_str   = f"${r.input_cost:.6f}"
        output_str  = f"${r.output_cost_per_1k:.6f}"
        ctx_str     = f"{r.context_used_pct:.2f}%"

        # Warn if context is getting full
        if r.context_used_pct > 75:
            ctx_str = f"[red]{ctx_str} ⚠[/red]"
        elif r.context_used_pct > 40:
            ctx_str = f"[yellow]{ctx_str}[/yellow]"

        row_style = "bold green" if is_cheapest else ""
        label     = f"✅ {r.model_name}" if is_cheapest else r.model_name

        table.add_row(label, r.encoding, token_str, input_str, output_str, ctx_str,
                      style=row_style)

    console.print()
    console.print(table)
    console.print(f"  💡 [bold green]Cheapest input:[/bold green] {cheapest_model}\n")


def print_token_breakdown(text: str, token_strings: list[str]) -> None:
    """
    Print a visual breakdown of how the text was split into tokens.
    Alternates colors so each token boundary is visible.
    """
    console.print("\n[bold cyan]Token Breakdown:[/bold cyan]")
    colors = ["yellow", "magenta", "cyan", "green", "blue"]
    output = Text()
    for i, tok in enumerate(token_strings):
        color = colors[i % len(colors)]
        output.append(tok, style=color)
    console.print(Panel(output, title=f"{len(token_strings)} tokens", border_style="dim"))


def print_json(results: list[ModelResult]) -> None:
    """
    Output results as clean JSON — useful for piping into other tools.
    """
    data = [
        {
            "model": r.model_name,
            "encoding": r.encoding,
            "token_count": r.token_count,
            "input_cost_usd": round(r.input_cost, 8),
            "output_cost_per_1k_usd": round(r.output_cost_per_1k, 8),
            "context_window": r.context_window,
            "context_used_pct": round(r.context_used_pct, 2),
        }
        for r in results
    ]
    print(json.dumps(data, indent=2))


def print_error(message: str) -> None:
    """Print a styled error message."""
    console.print(f"\n[bold red]Error:[/bold red] {message}\n")