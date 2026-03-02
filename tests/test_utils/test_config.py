"""Tests for config module."""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from codii.utils.config import CodiiConfig, get_config, set_config


class TestCodiiConfig:
    """Tests for CodiiConfig class."""

    def test_default_base_dir(self):
        """Test default base directory is set correctly."""
        config = CodiiConfig()
        assert config.base_dir == Path.home() / ".codii"

    def test_custom_base_dir(self):
        """Test custom base directory."""
        config = CodiiConfig()
        config.base_dir = Path("/custom/path")
        assert config.base_dir == Path("/custom/path")

    def test_indexes_dir_created(self, temp_dir):
        """Test indexes directory is created."""
        config = CodiiConfig()
        config.base_dir = temp_dir

        indexes_dir = config.indexes_dir
        assert indexes_dir.exists()
        assert indexes_dir == temp_dir / "indexes"

    def test_snapshots_dir_created(self, temp_dir):
        """Test snapshots directory is created."""
        config = CodiiConfig()
        config.base_dir = temp_dir

        snapshots_dir = config.snapshots_dir
        assert snapshots_dir.exists()
        assert snapshots_dir == temp_dir / "snapshots"

    def test_merkle_dir_created(self, temp_dir):
        """Test merkle directory is created."""
        config = CodiiConfig()
        config.base_dir = temp_dir

        merkle_dir = config.merkle_dir
        assert merkle_dir.exists()
        assert merkle_dir == temp_dir / "merkle"

    def test_snapshot_file_path(self, temp_dir):
        """Test snapshot file path."""
        config = CodiiConfig()
        config.base_dir = temp_dir

        snapshot_file = config.snapshot_file
        assert snapshot_file == temp_dir / "snapshots" / "snapshot.json"

    def test_default_ignore_patterns(self):
        """Test default ignore patterns are set."""
        config = CodiiConfig()
        assert ".git/" in config.default_ignore_patterns
        assert "__pycache__/" in config.default_ignore_patterns
        assert "node_modules/" in config.default_ignore_patterns

    def test_default_extensions(self):
        """Test default extensions are set."""
        config = CodiiConfig()
        assert ".py" in config.default_extensions
        assert ".js" in config.default_extensions
        assert ".ts" in config.default_extensions

    def test_embedding_settings(self):
        """Test embedding settings."""
        config = CodiiConfig()
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.embedding_batch_size == 32

    def test_hnsw_settings(self):
        """Test HNSW settings."""
        config = CodiiConfig()
        assert config.hnsw_m == 16
        assert config.hnsw_ef_construction == 200
        assert config.hnsw_ef_search == 100

    def test_chunk_settings(self):
        """Test chunk settings."""
        config = CodiiConfig()
        assert config.max_chunk_size == 1500
        assert config.min_chunk_size == 100
        assert config.chunk_overlap == 200

    def test_search_settings(self):
        """Test search settings."""
        config = CodiiConfig()
        assert config.default_search_limit == 10
        assert config.max_search_limit == 50
        assert config.bm25_weight == 0.5
        assert config.vector_weight == 0.5

    def test_rerank_settings(self):
        """Test rerank settings."""
        config = CodiiConfig()
        assert config.rerank_enabled is True
        assert config.rerank_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert config.rerank_candidates == 50
        assert config.rrf_limit == 20
        assert config.rerank_threshold == 0.5


class TestCodiiConfigLoad:
    """Tests for CodiiConfig.load method."""

    def test_load_no_config_file(self, temp_dir):
        """Test load with no config file uses defaults."""
        with patch.object(Path, 'cwd', return_value=temp_dir):
            config = CodiiConfig.load()
            assert config.base_dir == Path.home() / ".codii"

    def test_load_from_yaml_file(self, temp_dir):
        """Test load from .codii.yaml file."""
        config_file = temp_dir / ".codii.yaml"
        config_file.write_text("""
base_dir: /custom/storage
ignore_patterns:
  - "*.log"
  - "temp/"
extensions:
  - ".custom"
embedding_model: custom-model
max_chunk_size: 2000
""")

        with patch.object(Path, 'cwd', return_value=temp_dir):
            config = CodiiConfig.load(config_file)
            assert config.base_dir == Path("/custom/storage")
            assert "*.log" in config.default_ignore_patterns
            assert ".custom" in config.default_extensions
            assert config.embedding_model == "custom-model"
            assert config.max_chunk_size == 2000

    def test_load_env_variable_override(self, temp_dir):
        """Test CODII_BASE_DIR environment variable overrides config."""
        config_file = temp_dir / ".codii.yaml"
        config_file.write_text("base_dir: /yaml/path")

        with patch.dict(os.environ, {"CODII_BASE_DIR": "/env/path"}):
            with patch.object(Path, 'cwd', return_value=temp_dir):
                config = CodiiConfig.load(config_file)
                assert config.base_dir == Path("/env/path")

    def test_load_invalid_yaml(self, temp_dir):
        """Test load handles invalid YAML gracefully."""
        config_file = temp_dir / ".codii.yaml"
        config_file.write_text("invalid: [yaml: syntax")

        with patch.object(Path, 'cwd', return_value=temp_dir):
            # Should not raise, uses defaults
            config = CodiiConfig.load(config_file)
            assert config is not None

    def test_load_missing_config_file(self, temp_dir):
        """Test load with non-existent config file."""
        config_path = temp_dir / "nonexistent.yaml"
        config = CodiiConfig.load(config_path)
        assert config.base_dir == Path.home() / ".codii"


class TestGetEffectiveWorkers:
    """Tests for get_effective_workers method."""

    def test_explicit_workers(self):
        """Test explicit worker count."""
        assert CodiiConfig.get_effective_workers(4) == 4

    def test_auto_detect_workers(self):
        """Test auto-detect workers."""
        import os
        expected = max(1, (os.cpu_count() or 4) - 1)
        assert CodiiConfig.get_effective_workers(0) == expected

    def test_minimum_one_worker(self):
        """Test at least 1 worker."""
        with patch('os.cpu_count', return_value=1):
            assert CodiiConfig.get_effective_workers(0) >= 1


class TestGetEffectiveHnswThreads:
    """Tests for get_effective_hnsw_threads method."""

    def test_explicit_threads(self):
        """Test explicit thread count."""
        assert CodiiConfig.get_effective_hnsw_threads(8) == 8

    def test_auto_detect_threads(self):
        """Test auto-detect threads."""
        import os
        expected = os.cpu_count() or 4
        assert CodiiConfig.get_effective_hnsw_threads(0) == expected


class TestGlobalConfig:
    """Tests for global config functions."""

    def test_get_config_singleton(self):
        """Test get_config returns singleton."""
        import codii.utils.config as config_module
        config_module._config = None

        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_set_config(self):
        """Test set_config sets global config."""
        import codii.utils.config as config_module
        config_module._config = None

        custom_config = CodiiConfig()
        custom_config.base_dir = Path("/custom")
        set_config(custom_config)

        assert get_config() is custom_config

    def test_config_reset(self):
        """Test config can be reset."""
        import codii.utils.config as config_module

        config_module._config = None
        c1 = get_config()

        config_module._config = None
        c2 = get_config()

        # Different instances after reset
        assert c1 is not c2


class TestConfigWithYamlContent:
    """Tests for loading various YAML configurations."""

    def test_load_chunk_settings(self, temp_dir):
        """Test loading chunk settings from YAML."""
        config_file = temp_dir / ".codii.yaml"
        config_file.write_text("""
max_chunk_size: 2000
min_chunk_size: 200
chunk_overlap: 300
""")

        config = CodiiConfig.load(config_file)
        assert config.max_chunk_size == 2000
        assert config.min_chunk_size == 200
        assert config.chunk_overlap == 300

    def test_load_embedding_settings(self, temp_dir):
        """Test loading embedding settings from YAML."""
        config_file = temp_dir / ".codii.yaml"
        config_file.write_text("""
embedding_model: sentence-transformers/all-mpnet-base-v2
embedding_batch_size: 64
""")

        config = CodiiConfig.load(config_file)
        assert config.embedding_model == "sentence-transformers/all-mpnet-base-v2"
        assert config.embedding_batch_size == 64

    def test_load_empty_yaml(self, temp_dir):
        """Test loading empty YAML file."""
        config_file = temp_dir / ".codii.yaml"
        config_file.write_text("")

        config = CodiiConfig.load(config_file)
        # Should use defaults
        assert config.embedding_model == "all-MiniLM-L6-v2"

    def test_load_partial_config(self, temp_dir):
        """Test loading partial config keeps other defaults."""
        config_file = temp_dir / ".codii.yaml"
        config_file.write_text("embedding_model: custom-model")

        config = CodiiConfig.load(config_file)
        assert config.embedding_model == "custom-model"
        assert config.max_chunk_size == 1500  # Default