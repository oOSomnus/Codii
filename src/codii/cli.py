"""CLI tool for codii - code repository indexing and search."""

import sys
import time
import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich import print as rprint

from .utils.config import get_config
from .storage.snapshot import SnapshotManager
from .storage.database import Database
from .indexers.hybrid_search import HybridSearch

app = typer.Typer(
    name="codii",
    help="Local code repository indexing with hybrid BM25 and vector search",
    add_completion=False,
)
console = Console()


def get_path(path: Optional[str]) -> str:
    """Resolve path to absolute path string, defaulting to cwd."""
    if path:
        return str(Path(path).resolve())
    return str(Path.cwd().resolve())


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_index_size(path_hash: str) -> int:
    """Calculate total size of index files for a codebase."""
    config = get_config()
    index_dir = config.indexes_dir / path_hash
    total_size = 0

    if index_dir.exists():
        for file_path in index_dir.iterdir():
            if file_path.is_file():
                total_size += file_path.stat().st_size

    # Include merkle tree file
    merkle_path = config.merkle_dir / f"{path_hash}.json"
    if merkle_path.exists():
        total_size += merkle_path.stat().st_size

    return total_size


@app.command()
def status(
    path: Optional[str] = typer.Argument(
        None,
        help="Path to the codebase (defaults to current directory)",
    ),
):
    """Show indexing status for a codebase."""
    path_str = get_path(path)
    config = get_config()
    snapshot_manager = SnapshotManager(config.snapshot_file)
    status_info = snapshot_manager.get_status(path_str)

    # Color-coded status
    status_colors = {
        "indexed": "green",
        "indexing": "yellow",
        "failed": "red",
        "not_found": "dim",
    }
    color = status_colors.get(status_info.status, "white")

    # Build output
    lines = []
    lines.append(f"[bold]Path:[/bold] {status_info.path}")
    lines.append(f"[bold]Status:[/bold] [{color}]{status_info.status}[/{color}]")

    if status_info.status == "indexed":
        lines.append(f"[bold]Files:[/bold] {status_info.indexed_files}")
        lines.append(f"[bold]Chunks:[/bold] {status_info.total_chunks}")
        if status_info.merkle_root:
            lines.append(f"[bold]Merkle Root:[/bold] {status_info.merkle_root[:16]}...")
        path_hash = snapshot_manager.path_to_hash(path_str)
        index_size = get_index_size(path_hash)
        lines.append(f"[bold]Index Size:[/bold] {format_size(index_size)}")

    elif status_info.status == "indexing":
        lines.append(f"[bold]Progress:[/bold] {status_info.progress}%")
        lines.append(f"[bold]Stage:[/bold] {status_info.current_stage}")
        lines.append(f"[bold]Files:[/bold] {status_info.indexed_files}")
        lines.append(f"[bold]Chunks:[/bold] {status_info.total_chunks}")

    elif status_info.status == "failed":
        lines.append(f"[bold]Error:[/bold] [red]{status_info.error_message}[/red]")

    if status_info.last_updated:
        lines.append(f"[bold]Last Updated:[/bold] {status_info.last_updated}")

    console.print(Panel("\n".join(lines), title="Codebase Status"))


@app.command("list")
def list_codebases():
    """List all indexed codebases."""
    config = get_config()
    snapshot_manager = SnapshotManager(config.snapshot_file)
    codebases = snapshot_manager.get_all_codebases()

    if not codebases:
        console.print("[dim]No codebases indexed.[/dim]")
        return

    table = Table(title="Indexed Codebases")
    table.add_column("Path", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Files", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Size", justify="right")

    status_colors = {
        "indexed": "green",
        "indexing": "yellow",
        "failed": "red",
    }

    for path, status in sorted(codebases.items(), key=lambda x: x[0]):
        path_hash = snapshot_manager.path_to_hash(path)
        index_size = get_index_size(path_hash)
        color = status_colors.get(status.status, "white")

        table.add_row(
            path[:50] + "..." if len(path) > 50 else path,
            f"[{color}]{status.status}[/{color}]",
            str(status.indexed_files),
            str(status.total_chunks),
            format_size(index_size),
        )

    console.print(table)


@app.command()
def inspect(
    query: str = typer.Argument(..., help="Search query"),
    path: Optional[str] = typer.Argument(None, help="Path to codebase (defaults to cwd)"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results"),
    raw: bool = typer.Option(False, "--raw", "-r", help="Show full content without truncation"),
):
    """Search and inspect chunks matching a query."""
    path_str = get_path(path)
    config = get_config()
    snapshot_manager = SnapshotManager(config.snapshot_file)
    status_info = snapshot_manager.get_status(path_str)

    if status_info.status == "not_found":
        console.print(f"[red]Error: Codebase not indexed: {path_str}[/red]")
        console.print("Run [bold]codii build[/bold] to index this codebase.")
        raise typer.Exit(1)

    if status_info.status == "failed":
        console.print(f"[red]Error: Indexing failed: {status_info.error_message}[/red]")
        raise typer.Exit(1)

    if status_info.status == "indexing":
        console.print("[yellow]Warning: Indexing in progress, results may be incomplete.[/yellow]")

    path_hash = snapshot_manager.path_to_hash(path_str)
    index_dir = config.indexes_dir / path_hash
    db_path = index_dir / "chunks.db"
    vector_path = index_dir / "vectors"

    if not db_path.exists():
        console.print(f"[red]Error: Index not found for {path_str}[/red]")
        raise typer.Exit(1)

    try:
        hybrid_search = HybridSearch(db_path, vector_path)
        results = hybrid_search.search(query, limit=min(limit, config.max_search_limit))
        hybrid_search.close()
    except Exception as e:
        console.print(f"[red]Search error: {e}[/red]")
        raise typer.Exit(1)

    if not results:
        console.print(f"[dim]No results found for query: '{query}'[/dim]")
        return

    repo_path = Path(path_str)
    for i, result in enumerate(results, 1):
        # Get relative path
        try:
            rel_path = Path(result.path).relative_to(repo_path)
        except ValueError:
            rel_path = Path(result.path).name

        # Truncate content if needed
        content = result.content
        max_len = None if raw else 1000
        if max_len and len(content) > max_len:
            content = content[:max_len] + "\n... (truncated, use --raw for full content)"

        console.print(f"\n[bold cyan]Result {i}[/bold cyan] (rank: {result.rank:.4f})")
        console.print(f"[bold]File:[/bold] {rel_path}:{result.start_line}-{result.end_line}")
        console.print(f"[bold]Language:[/bold] {result.language} | [bold]Type:[/bold] {result.chunk_type}")
        console.print(f"[dim]{'─' * 60}[/dim]")
        console.print(content)


@app.command()
def clear(
    path: Optional[str] = typer.Argument(None, help="Path to codebase to clear"),
    all: bool = typer.Option(False, "--all", "-a", help="Clear all indexed codebases"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
):
    """Clear index for a codebase or all codebases."""
    config = get_config()
    snapshot_manager = SnapshotManager(config.snapshot_file)

    if all:
        codebases = snapshot_manager.get_all_codebases()
        if not codebases:
            console.print("[dim]No codebases to clear.[/dim]")
            return

        if not force:
            confirm = typer.confirm(f"This will clear {len(codebases)} codebase(s). Continue?")
            if not confirm:
                console.print("[dim]Cancelled.[/dim]")
                return

        for codebase_path in codebases:
            _clear_codebase(codebase_path, snapshot_manager, config)
            console.print(f"[green]Cleared:[/green] {codebase_path}")

        console.print(f"\n[green]Cleared {len(codebases)} codebase(s).[/green]")
        return

    path_str = get_path(path)
    status_info = snapshot_manager.get_status(path_str)

    if status_info.status == "not_found":
        console.print(f"[red]Error: Codebase not found in index: {path_str}[/red]")
        raise typer.Exit(1)

    if status_info.status == "indexing":
        console.print("[red]Error: Cannot clear while indexing is in progress.[/red]")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Clear index for {path_str}?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            return

    _clear_codebase(path_str, snapshot_manager, config)
    console.print(f"[green]Index cleared for {path_str}[/green]")


def _clear_codebase(path_str: str, snapshot_manager: SnapshotManager, config) -> None:
    """Clear the index for a single codebase."""
    path_hash = snapshot_manager.path_to_hash(path_str)
    index_dir = config.indexes_dir / path_hash

    # Delete database
    db_path = index_dir / "chunks.db"
    if db_path.exists():
        db_path.unlink()

    # Delete vector index files
    for f in ["vectors.bin", "vectors.meta.json"]:
        fp = index_dir / f
        if fp.exists():
            fp.unlink()

    # Delete merkle tree
    merkle_path = config.merkle_dir / f"{path_hash}.json"
    if merkle_path.exists():
        merkle_path.unlink()

    # Remove index directory if empty
    try:
        if index_dir.exists() and not any(index_dir.iterdir()):
            index_dir.rmdir()
    except Exception:
        pass

    # Remove from snapshot
    snapshot_manager.remove_codebase(path_str)


@app.command()
def build(
    path: Optional[str] = typer.Argument(None, help="Path to codebase to index"),
    force: bool = typer.Option(False, "--force", "-f", help="Force full re-index"),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run indexing in background"),
):
    """Build or rebuild index for a codebase."""
    path_str = get_path(path)

    # Import here to avoid circular imports and delay loading heavy deps
    from .tools.index_codebase import IndexCodebaseTool

    tool = IndexCodebaseTool()

    # Check path exists
    repo_path = Path(path_str)
    if not repo_path.exists():
        console.print(f"[red]Error: Path does not exist: {path_str}[/red]")
        raise typer.Exit(1)
    if not repo_path.is_dir():
        console.print(f"[red]Error: Path is not a directory: {path_str}[/red]")
        raise typer.Exit(1)

    # Check if already indexing
    if tool.snapshot_manager.is_indexing(path_str):
        console.print("[yellow]Codebase is already being indexed. Use 'codii status' to check progress.[/yellow]")
        raise typer.Exit(1)

    if daemon:
        # Run in background (same as MCP behavior)
        result = tool.run(path_str, force=force)
        if result.get("isError"):
            console.print(f"[red]{result['content'][0]['text']}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]Indexing started in background for {path_str}[/green]")
        console.print("Use [bold]codii status[/bold] to check progress.")
        return

    # Foreground mode with progress bar
    _run_indexing_with_progress(tool, path_str, force)


def _run_indexing_with_progress(tool, path_str: str, force: bool) -> None:
    """Run indexing in foreground with progress bar."""
    config = get_config()
    snapshot_manager = SnapshotManager(config.snapshot_file)

    # Check current status
    status = snapshot_manager.get_status(path_str)

    # Handle already indexed case
    if status.status == "indexed" and not force:
        from .merkle.tree import MerkleTree
        from .utils.file_utils import scan_directory

        path_hash = snapshot_manager.path_to_hash(path_str)
        merkle_path = config.merkle_dir / f"{path_hash}.json"
        old_merkle = MerkleTree.load(merkle_path)

        if old_merkle:
            # Quick check for changes
            files = scan_directory(
                Path(path_str),
                set(config.default_extensions),
                config.default_ignore_patterns,
                [], [], use_gitignore=True,
            )
            new_merkle = MerkleTree()
            for file_path, file_hash in files:
                new_merkle.add_file(str(file_path), file_hash)
            new_merkle.compute_root()

            if new_merkle.root_hash == old_merkle.root_hash:
                console.print("[dim]No changes detected. Use --force to re-index.[/dim]")
                return

    # Clear existing index if force
    if force and status.status == "indexed":
        _clear_codebase(path_str, snapshot_manager, config)

    # Start indexing in background thread
    import threading
    thread = threading.Thread(
        target=tool._index_codebase,
        args=(path_str, "ast", [], [], force),
        daemon=True,
    )

    # Mark as indexing
    snapshot_manager.mark_indexing(path_str)
    thread.start()

    # Progress tracking
    stage_names = {
        "preparing": "Preparing",
        "deleting": "Deleting stale chunks",
        "chunking": "Chunking files",
        "embedding": "Generating embeddings",
        "indexing": "Building index",
        "complete": "Complete",
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Preparing...", total=100)
        last_stage = ""
        last_progress = 0

        try:
            while thread.is_alive():
                status = snapshot_manager.get_status(path_str)
                current_progress = status.progress

                # Update stage description
                if status.current_stage != last_stage:
                    stage_name = stage_names.get(status.current_stage, status.current_stage)
                    progress.update(task, description=f"[cyan]{stage_name}...")
                    last_stage = status.current_stage

                # Update progress bar
                if current_progress != last_progress:
                    progress.update(task, completed=current_progress)
                    last_progress = current_progress

                time.sleep(0.2)

            # Check final status
            final_status = snapshot_manager.get_status(path_str)
            progress.update(task, completed=100)

            if final_status.status == "indexed":
                console.print(f"\n[green]Indexing complete![/green]")
                console.print(f"  Files: {final_status.indexed_files}")
                console.print(f"  Chunks: {final_status.total_chunks}")
            elif final_status.status == "failed":
                console.print(f"\n[red]Indexing failed: {final_status.error_message}[/red]")
                raise typer.Exit(1)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user.[/yellow]")
            snapshot_manager.mark_failed(path_str, "Interrupted by user")
            raise typer.Exit(1)


@app.command()
def stats(
    path: Optional[str] = typer.Argument(None, help="Path to codebase (defaults to cwd)"),
):
    """Show detailed statistics for an indexed codebase."""
    path_str = get_path(path)
    config = get_config()
    snapshot_manager = SnapshotManager(config.snapshot_file)
    status_info = snapshot_manager.get_status(path_str)

    if status_info.status == "not_found":
        console.print(f"[red]Error: Codebase not indexed: {path_str}[/red]")
        console.print("Run [bold]codii build[/bold] to index this codebase.")
        raise typer.Exit(1)

    if status_info.status == "indexing":
        console.print("[yellow]Warning: Indexing in progress, stats may be incomplete.[/yellow]")

    path_hash = snapshot_manager.path_to_hash(path_str)
    index_dir = config.indexes_dir / path_hash
    db_path = index_dir / "chunks.db"

    if not db_path.exists():
        console.print(f"[red]Error: Index database not found for {path_str}[/red]")
        raise typer.Exit(1)

    # Get stats from database
    db = Database(db_path)

    # Basic counts
    chunk_count = db.get_chunk_count()
    file_count = db.get_file_count()

    # Breakdown by language
    cursor = db.conn.execute(
        "SELECT language, COUNT(*) as count FROM chunks GROUP BY language ORDER BY count DESC"
    )
    languages = cursor.fetchall()

    # Breakdown by chunk type
    cursor = db.conn.execute(
        "SELECT chunk_type, COUNT(*) as count FROM chunks GROUP BY chunk_type ORDER BY count DESC"
    )
    chunk_types = cursor.fetchall()

    # Index size
    index_size = get_index_size(path_hash)

    db.close()

    # Display results
    lines = []
    lines.append(f"[bold]Path:[/bold] {status_info.path}")
    lines.append(f"[bold]Status:[/bold] [green]{status_info.status}[/green]")
    lines.append(f"[bold]Merkle Root:[/bold] {status_info.merkle_root or 'N/A'}")
    lines.append("")
    lines.append(f"[bold]Total Files:[/bold] {file_count}")
    lines.append(f"[bold]Total Chunks:[/bold] {chunk_count}")
    lines.append(f"[bold]Index Size:[/bold] {format_size(index_size)}")

    console.print(Panel("\n".join(lines), title="Index Statistics"))

    # Language breakdown table
    if languages:
        lang_table = Table(title="Chunks by Language")
        lang_table.add_column("Language", style="cyan")
        lang_table.add_column("Count", justify="right")
        lang_table.add_column("%", justify="right")

        for row in languages:
            lang = row["language"] or "unknown"
            count = row["count"]
            pct = (count / chunk_count * 100) if chunk_count > 0 else 0
            lang_table.add_row(lang, str(count), f"{pct:.1f}%")

        console.print(lang_table)

    # Chunk type breakdown table
    if chunk_types:
        type_table = Table(title="Chunks by Type")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", justify="right")
        type_table.add_column("%", justify="right")

        for row in chunk_types:
            chunk_type = row["chunk_type"] or "unknown"
            count = row["count"]
            pct = (count / chunk_count * 100) if chunk_count > 0 else 0
            type_table.add_row(chunk_type, str(count), f"{pct:.1f}%")

        console.print(type_table)


@app.command()
def version():
    """Show codii version."""
    console.print("codii version 0.1.0")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()