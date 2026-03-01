"""Configuration management for codii."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class CodiiConfig:
    """Configuration for codii indexing."""

    # Storage paths
    base_dir: Path = field(default_factory=lambda: Path.home() / ".codii")

    # Default ignore patterns
    default_ignore_patterns: List[str] = field(default_factory=lambda: [
        ".git/",
        "__pycache__/",
        "node_modules/",
        ".venv/",
        "venv/",
        ".env/",
        "*.pyc",
        "*.pyo",
        "*.so",
        "*.dll",
        "*.dylib",
        "*.exe",
        "*.bin",
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.svg",
        "*.ico",
        "*.pdf",
        "*.zip",
        "*.tar",
        "*.gz",
        "*.rar",
        "*.7z",
        ".DS_Store",
        "Thumbs.db",
        "*.log",
        "*.tmp",
        ".idea/",
        ".vscode/",
        "*.swp",
        "*.swo",
        "dist/",
        "build/",
        "target/",
        ".tox/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        "coverage/",
        "*.egg-info/",
    ])

    # Default file extensions to index
    default_extensions: List[str] = field(default_factory=lambda: [
        ".py", ".js", ".jsx", ".ts", ".tsx",
        ".go", ".rs", ".java", ".c", ".cpp", ".cc", ".cxx",
        ".h", ".hpp", ".hxx",
        ".json", ".yaml", ".yml", ".toml",
        ".md", ".rst", ".txt",
        ".sh", ".bash", ".zsh",
        ".sql", ".proto",
        ".html", ".css", ".scss", ".less",
    ])

    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # HNSW settings
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100  # Increased from 50 for better recall on multi-word queries

    # Chunk settings
    max_chunk_size: int = 1500
    min_chunk_size: int = 100
    chunk_overlap: int = 200

    # Search settings
    default_search_limit: int = 10
    max_search_limit: int = 50
    bm25_weight: float = 0.5
    vector_weight: float = 0.5

    # Re-ranking settings
    rerank_enabled: bool = True  # Enabled by default
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast and effective
    rerank_candidates: int = 50  # BM25/Vector initial candidates for Stage 1
    rrf_limit: int = 20  # RRF results passed to re-ranking (Stage 2)
    rerank_threshold: float = 0.5  # Minimum score threshold (normalized 0-1)

    @property
    def indexes_dir(self) -> Path:
        """Get the indexes directory."""
        path = self.base_dir / "indexes"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def snapshots_dir(self) -> Path:
        """Get the snapshots directory."""
        path = self.base_dir / "snapshots"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def merkle_dir(self) -> Path:
        """Get the merkle tree cache directory."""
        path = self.base_dir / "merkle"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def snapshot_file(self) -> Path:
        """Get the snapshot file path."""
        return self.snapshots_dir / "snapshot.json"

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "CodiiConfig":
        """Load configuration from file or use defaults."""
        config = cls()

        # Try to load from config file
        if config_path is None:
            config_path = Path.cwd() / ".codii.yaml"

        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    data = yaml.safe_load(f) or {}

                # Override base_dir if specified
                if "base_dir" in data:
                    config.base_dir = Path(data["base_dir"])

                # Override ignore patterns if specified
                if "ignore_patterns" in data:
                    config.default_ignore_patterns.extend(data["ignore_patterns"])

                # Override extensions if specified
                if "extensions" in data:
                    config.default_extensions.extend(data["extensions"])

                # Override other settings
                if "embedding_model" in data:
                    config.embedding_model = data["embedding_model"]
                if "embedding_batch_size" in data:
                    config.embedding_batch_size = data["embedding_batch_size"]
                if "max_chunk_size" in data:
                    config.max_chunk_size = data["max_chunk_size"]
                if "min_chunk_size" in data:
                    config.min_chunk_size = data["min_chunk_size"]
                if "chunk_overlap" in data:
                    config.chunk_overlap = data["chunk_overlap"]

            except Exception as e:
                # Log warning but continue with defaults
                import sys
                print(f"Warning: Failed to load config from {config_path}: {e}", file=sys.stderr)

        # Override from environment variables
        if "CODII_BASE_DIR" in os.environ:
            config.base_dir = Path(os.environ["CODII_BASE_DIR"])

        return config

    @staticmethod
    def get_effective_workers(requested: int = 0) -> int:
        """Get chunking workers, auto-detect if not specified.

        Args:
            requested: Requested worker count. 0 means auto-detect.

        Returns:
            Number of workers to use for parallel chunking.
        """
        if requested > 0:
            return requested
        return max(1, (os.cpu_count() or 4) - 1)

    @staticmethod
    def get_effective_hnsw_threads(requested: int = 0) -> int:
        """Get HNSW threads, auto-detect if not specified.

        Args:
            requested: Requested thread count. 0 means auto-detect.

        Returns:
            Number of threads to use for HNSW index construction.
        """
        if requested > 0:
            return requested
        return os.cpu_count() or 4


# Global config instance
_config: Optional[CodiiConfig] = None


def get_config() -> CodiiConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = CodiiConfig.load()
    return _config


def set_config(config: CodiiConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config