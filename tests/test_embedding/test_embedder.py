"""Tests for Embedder class."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

from codii.embedding.embedder import Embedder, get_embedder


class TestEmbedderSingleton:
    """Tests for the singleton pattern."""

    def test_singleton_returns_same_instance(self):
        """Test that multiple calls return the same instance."""
        # Reset singleton
        Embedder._instance = None

        e1 = Embedder()
        e2 = Embedder()
        assert e1 is e2

    def test_singleton_with_different_model_names(self):
        """Test that singleton ignores different model names after first init."""
        Embedder._instance = None

        e1 = Embedder("model-a")
        e2 = Embedder("model-b")

        # Both should be the same instance
        assert e1 is e2
        # Model name should be from first initialization
        assert e1.model_name == "model-a"

    def test_get_embedder_returns_singleton(self):
        """Test get_embedder returns the singleton instance."""
        Embedder._instance = None

        e1 = get_embedder()
        e2 = get_embedder("different-model")

        assert e1 is e2

    def test_singleton_thread_safety(self):
        """Test singleton works correctly with concurrent access."""
        import threading

        Embedder._instance = None
        instances = []

        def create_instance():
            instances.append(Embedder())

        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same
        assert all(inst is instances[0] for inst in instances)


class TestEmbedderEmbed:
    """Tests for the embed method."""

    def test_embed_empty_list(self):
        """Test embed with empty list returns empty array."""
        Embedder._instance = None

        with patch.object(Embedder, 'model', new_callable=PropertyMock) as mock_model_prop:
            # Don't need to mock model.encode since empty list early returns
            e = Embedder()
            result = e.embed([])

            assert isinstance(result, np.ndarray)
            assert len(result) == 0

    def test_embed_single_text(self):
        """Test embed with a single text."""
        Embedder._instance = None

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        with patch.object(Embedder, 'model', new_callable=PropertyMock) as mock_model_prop:
            mock_model_prop.return_value = mock_model

            e = Embedder()
            result = e.embed(["hello world"])

            assert isinstance(result, np.ndarray)
            assert result.shape == (1, 3)
            mock_model.encode.assert_called_once_with(
                ["hello world"],
                convert_to_numpy=True,
                show_progress_bar=False
            )

    def test_embed_multiple_texts(self):
        """Test embed with multiple texts."""
        Embedder._instance = None

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])

        with patch.object(Embedder, 'model', new_callable=PropertyMock) as mock_model_prop:
            mock_model_prop.return_value = mock_model

            e = Embedder()
            result = e.embed(["text1", "text2"])

            assert result.shape == (2, 3)


class TestEmbedderEmbedSingle:
    """Tests for the embed_single method."""

    def test_embed_single_calls_model_encode(self):
        """Test embed_single calls model.encode with correct args."""
        Embedder._instance = None

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        with patch.object(Embedder, 'model', new_callable=PropertyMock) as mock_model_prop:
            mock_model_prop.return_value = mock_model

            e = Embedder()
            result = e.embed_single("test text")

            assert isinstance(result, np.ndarray)
            mock_model.encode.assert_called_once_with(
                "test text",
                convert_to_numpy=True,
                show_progress_bar=False
            )


class TestEmbedderProperties:
    """Tests for Embedder properties."""

    def test_embedding_dim_returns_cached_value(self):
        """Test embedding_dim returns cached dimension."""
        Embedder._instance = None

        e = Embedder()
        e._embedding_dim = 384  # Set directly to avoid model loading

        assert e.embedding_dim == 384

    def test_embedding_dim_loads_model_if_needed(self):
        """Test embedding_dim loads model if not cached."""
        Embedder._instance = None

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768

        # Create a custom model property that simulates the lazy loading
        def model_property_side_effect():
            e._model = mock_model
            e._embedding_dim = 768
            return mock_model

        with patch.object(Embedder, 'model', new_callable=PropertyMock) as mock_model_prop:
            mock_model_prop.side_effect = model_property_side_effect

            e = Embedder()
            e._embedding_dim = None  # Force model loading

            dim = e.embedding_dim
            assert dim == 768

    def test_model_lazy_loading(self):
        """Test model property lazily loads the model."""
        Embedder._instance = None

        mock_sentence_transformer = MagicMock()
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_instance

        e = Embedder("test-model")
        assert e._model is None  # Not loaded yet

        # Mock the SentenceTransformer import
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock(SentenceTransformer=mock_sentence_transformer)}):
            with patch('builtins.print'):  # Suppress print output
                model = e.model

                # Model should now be loaded
                assert e._model is not None
                mock_sentence_transformer.assert_called_once_with("test-model")

    def test_model_only_loaded_once(self):
        """Test model is only loaded once even with multiple accesses."""
        Embedder._instance = None

        mock_sentence_transformer = MagicMock()
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_instance

        with patch.dict('sys.modules', {'sentence_transformers': MagicMock(SentenceTransformer=mock_sentence_transformer)}):
            with patch('builtins.print'):
                e = Embedder()

                # Access model multiple times
                _ = e.model
                _ = e.model
                _ = e.model

                # Should only be called once
                mock_sentence_transformer.assert_called_once()


class TestEmbedderInitialized:
    """Tests for the initialization flag."""

    def test_init_only_runs_once(self):
        """Test that __init__ only sets attributes on first call."""
        Embedder._instance = None

        e1 = Embedder("model-a")
        assert e1.model_name == "model-a"
        assert e1._initialized is True

        # Create another "instance" (same singleton)
        e2 = Embedder("model-b")
        assert e2.model_name == "model-a"  # Should not change
        assert e2 is e1

    def test_initialized_flag_prevents_reinitialization(self):
        """Test that _initialized flag prevents reinitialization."""
        Embedder._instance = None

        e = Embedder("first-model")
        assert e._initialized is True

        # Manually try to call __init__ again
        e.__init__("second-model")
        assert e.model_name == "first-model"  # Should not change