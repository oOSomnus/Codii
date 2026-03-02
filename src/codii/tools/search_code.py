"""Search code tool for codii."""

from pathlib import Path
from typing import Optional, List

from ..utils.config import get_config
from ..storage.snapshot import SnapshotManager
from ..indexers.hybrid_search import HybridSearch


class SearchCodeTool:
    """Tool to search indexed code."""

    def __init__(self):
        self.config = get_config()
        self.snapshot_manager = SnapshotManager(self.config.snapshot_file)

    def get_input_schema(self) -> dict:
        """Get the input schema for the tool."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the indexed codebase",
                },
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "limit": {
                    "type": "number",
                    "default": 10,
                    "maximum": 50,
                    "description": "Maximum number of results",
                },
                "extensionFilter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Filter by file extensions (e.g., ['.py', '.js'])",
                },
                "rerank": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable cross-encoder re-ranking for improved relevance (default: false)",
                },
            },
            "required": ["path", "query"],
        }

    def run(
        self,
        path: str,
        query: str,
        limit: int = 10,
        extensionFilter: Optional[List[str]] = None,
        rerank: Optional[bool] = None,
    ) -> dict:
        """
        Search indexed code.

        Args:
            path: Absolute path to the indexed codebase
            query: Search query
            limit: Maximum number of results (max 50)
            extensionFilter: Filter by file extensions
            rerank: Enable cross-encoder re-ranking (None uses config default)

        Returns:
            Dict with search results or error
        """
        # Validate path
        repo_path = Path(path).resolve()
        path_str = str(repo_path)

        # Check status
        status = self.snapshot_manager.get_status(path_str)

        if status.status == "not_found":
            return {
                "content": [{
                    "type": "text",
                    "text": f"Codebase not indexed: {path}. Please index it first using the index_codebase tool."
                }],
                "isError": True,
            }

        if status.status == "failed":
            return {
                "content": [{
                    "type": "text",
                    "text": f"Indexing failed for {path}. Error: {status.error_message}. Please retry with index_codebase."
                }],
                "isError": True,
            }

        # Get index paths
        path_hash = self.snapshot_manager.path_to_hash(path_str)
        index_dir = self.config.indexes_dir / path_hash
        db_path = index_dir / "chunks.db"
        vector_path = index_dir / "vectors"

        if not db_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"Index not found for {path}. Please index it first using the index_codebase tool."
                }],
                "isError": True,
            }

        # Limit check
        limit = min(limit, self.config.max_search_limit)

        # Perform search
        try:
            hybrid_search = HybridSearch(db_path, vector_path)
            results = hybrid_search.search(query, limit=limit, rerank=rerank)
            hybrid_search.close()
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Search error: {str(e)}"
                }],
                "isError": True,
            }

        # Format results
        if not results:
            warning = ""
            if status.status == "indexing":
                warning = "\n\nNote: Indexing is still in progress. Results may be incomplete."

            return {
                "content": [{
                    "type": "text",
                    "text": f"No results found for query: '{query}'{warning}\n\nThe codebase may still be indexing or the query returned no matches."
                }],
                "isError": False,
            }

        # Build output
        output_lines = []
        for result in results:
            # Filter by extension if specified
            if extensionFilter:
                ext = Path(result.path).suffix.lower()
                normalized_filters = [f if f.startswith(".") else "." + f for f in extensionFilter]
                if ext not in normalized_filters:
                    continue

            # Truncate content if too long
            content = result.content
            if len(content) > 5000:
                content = content[:5000] + "\n... (truncated)"

            # Get relative path
            try:
                rel_path = Path(result.path).relative_to(repo_path)
            except ValueError:
                rel_path = Path(result.path).name

            # Build score info
            score_info = f"Rank: {result.rank}"
            if result.rerank_score > 0:
                score_info += f" (relevance: {result.rerank_score:.2f})"

            output_lines.append(
                f"Code snippet ({result.language}) [chunk_type: {result.chunk_type}]\n"
                f"Location: {rel_path}:{result.start_line}-{result.end_line}\n"
                f"{score_info}\n"
                f"Context:\n```\n{content}\n```\n"
            )

        # Add warning if indexing
        if status.status == "indexing":
            output_lines.append(
                "\nNote: Indexing in Progress... results may be incomplete."
            )

        return {
            "content": [{
                "type": "text",
                "text": "\n".join(output_lines)
            }],
            "isError": False,
        }