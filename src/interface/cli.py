# src/interface/cli.py

from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich import box
from rich.text import Text

from src.domain.models import SearchResult


console = Console()


def display_welcome_banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]🔍 Semantic Search Module[/bold cyan]\n"
        "[dim]Powered by sentence-transformers + cosine similarity[/dim]",
        box=box.DOUBLE,
        border_style="cyan",
    ))


def display_indexing_status(num_fragments: int) -> None:
    console.print(f"\n[green]✓[/green] Index built — [bold]{num_fragments}[/bold] fragments ready for search.\n")


def prompt_for_query() -> str:
    return Prompt.ask("\n[bold yellow]❓ Your question[/bold yellow]")


def display_results(query: str, results: List[SearchResult]) -> None:
    console.print(f"\n[bold]Results for:[/bold] [italic]\"{query}\"[/italic]\n")

    for rank, result in enumerate(results, start=1):
        score_color = _score_to_color(result.similarity_score)
        score_display = f"[{score_color}]{result.similarity_score:.4f}[/{score_color}]"

        panel_content = Text()
        panel_content.append(f"📄 Source: ", style="dim")
        panel_content.append(result.fragment.source_document, style="bold white")
        panel_content.append(f"\n🎯 Score: ")
        panel_content.append_text(Text.from_markup(score_display))
        panel_content.append(f"\n\n{result.fragment.text}")

        console.print(Panel(
            panel_content,
            title=f"[bold]#{rank}[/bold]",
            border_style=score_color,
            box=box.ROUNDED,
            padding=(1, 2),
        ))


def display_error(message: str) -> None:
    console.print(f"\n[bold red]✗ Error:[/bold red] {message}\n")


def ask_continue() -> bool:
    answer = Prompt.ask(
        "\n[dim]Search again?[/dim]",
        choices=["y", "n"],
        default="y",
    )
    return answer.lower() == "y"


def _score_to_color(score: float) -> str:
    if score >= 0.75:
        return "green"
    elif score >= 0.50:
        return "yellow"
    else:
        return "red"