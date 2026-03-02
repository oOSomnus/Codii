"""Tests for CoIR benchmark integration."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary storage directory for tests."""
    storage = tmp_path / ".codii"
    storage.mkdir(parents=True, exist_ok=True)
    return storage


@pytest.fixture
def benchmark_config(temp_storage_dir):
    """Mock configuration for benchmark tests."""
    from codii.utils.config import CodiiConfig, set_config
    config = CodiiConfig()
    config.base_dir = temp_storage_dir
    set_config(config)
    yield config
    # Reset global config after test
    import codii.utils.config as config_module
    config_module._config = None


@pytest.fixture
def sample_corpus():
    """Create a sample corpus for testing."""
    return {
        "doc1": """
def hello_world():
    '''A simple greeting function.'''
    print("Hello, World!")
    return True
""",
        "doc2": """
class Calculator:
    '''A simple calculator class.'''

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
""",
        "doc3": """
import os
import sys

def main():
    '''Main entry point.'''
    print("Starting application")
    return 0
""",
        "doc4": """
# JavaScript code
function fetchData(url) {
    return fetch(url)
        .then(response => response.json())
        .catch(error => console.error(error));
}
""",
        "doc5": """
// Go code
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
""",
    }


@pytest.fixture
def sample_queries():
    """Create sample queries for testing."""
    return {
        "q1": "greeting function",
        "q2": "calculator class",
        "q3": "main entry point",
        "q4": "fetch data javascript",
        "q5": "print hello world",
    }


@pytest.fixture
def sample_qrels():
    """Create sample relevance judgments."""
    return {
        "q1": {"doc1": 1},
        "q2": {"doc2": 1},
        "q3": {"doc3": 1},
        "q4": {"doc4": 1},
        "q5": {"doc1": 1, "doc5": 1},
    }


class TestCodiiCoIRAdapter:
    """Tests for the CodiiCoIRAdapter class."""

    def test_adapter_initialization(self):
        """Test adapter can be initialized."""
        from codii.evaluation.coir_adapter import CodiiCoIRAdapter

        adapter = CodiiCoIRAdapter(cleanup_index=True)
        assert adapter.cleanup_index is True
        assert adapter.temp_dir is None

    def test_index_corpus(self, sample_corpus, benchmark_config):
        """Test corpus indexing."""
        from codii.evaluation.coir_adapter import CodiiCoIRAdapter

        adapter = CodiiCoIRAdapter(cleanup_index=True, rerank=False)

        try:
            adapter.index_corpus(sample_corpus)

            assert adapter.get_doc_count() == 5
            assert adapter.get_chunk_count() >= 5  # At least one chunk per doc
            assert adapter._hybrid_search is not None
        finally:
            adapter._cleanup()

    def test_search_returns_results(self, sample_corpus, benchmark_config):
        """Test search returns results."""
        from codii.evaluation.coir_adapter import CodiiCoIRAdapter

        adapter = CodiiCoIRAdapter(cleanup_index=True, rerank=False)

        try:
            adapter.index_corpus(sample_corpus)

            results = adapter.search("greeting function", top_k=3)

            assert len(results) > 0
            assert len(results) <= 3
            assert all(hasattr(r, 'doc_id') for r in results)
            assert all(hasattr(r, 'score') for r in results)
        finally:
            adapter._cleanup()

    def test_search_relevance(self, sample_corpus, benchmark_config):
        """Test search returns relevant results."""
        from codii.evaluation.coir_adapter import CodiiCoIRAdapter

        adapter = CodiiCoIRAdapter(cleanup_index=True, rerank=False)

        try:
            adapter.index_corpus(sample_corpus)

            # Search for calculator should return doc2
            results = adapter.search("calculator add subtract", top_k=5)
            doc_ids = [r.doc_id for r in results]

            assert "doc2" in doc_ids
        finally:
            adapter._cleanup()

    def test_get_scores(self, sample_corpus, benchmark_config):
        """Test get_scores returns scores for documents."""
        from codii.evaluation.coir_adapter import CodiiCoIRAdapter

        adapter = CodiiCoIRAdapter(cleanup_index=True, rerank=False)

        try:
            adapter.index_corpus(sample_corpus)

            scores = adapter.get_scores("calculator", ["doc1", "doc2", "doc3"])

            assert len(scores) == 3
            assert all(isinstance(s, float) for s in scores)
        finally:
            adapter._cleanup()

    def test_context_manager(self, sample_corpus, benchmark_config):
        """Test adapter works as context manager."""
        from codii.evaluation.coir_adapter import CodiiCoIRAdapter

        with CodiiCoIRAdapter(cleanup_index=True) as adapter:
            adapter.index_corpus(sample_corpus)
            results = adapter.search("greeting", top_k=3)
            assert len(results) > 0

        # After context, temp dir should be cleaned up
        assert adapter.temp_dir is None

    def test_language_detection(self, benchmark_config):
        """Test language detection from content."""
        from codii.evaluation.coir_adapter import CodiiCoIRAdapter

        adapter = CodiiCoIRAdapter(cleanup_index=True)

        # Python detection
        assert adapter._detect_language("def foo():\n    pass", "") == "python"
        assert adapter._detect_language("", "script.py") == "python"

        # JavaScript detection
        assert adapter._detect_language("function foo() { return 1; }", "") == "javascript"
        assert adapter._detect_language("", "script.js") == "javascript"

        # Go detection
        assert adapter._detect_language("package main\n\nfunc main() { }", "") == "go"
        assert adapter._detect_language("", "main.go") == "go"

    def test_chunk_document(self, benchmark_config):
        """Test document chunking."""
        from codii.evaluation.coir_adapter import CodiiCoIRAdapter

        adapter = CodiiCoIRAdapter(cleanup_index=True)

        # Short document should create one chunk
        content = "def hello():\n    pass"
        chunks = adapter._chunk_document("test.py", content, "python")
        assert len(chunks) == 1
        assert chunks[0].content == content

        # Long document should create multiple chunks
        long_content = "\n".join([f"# Line {i}" * 10 for i in range(200)])
        chunks = adapter._chunk_document("long.py", long_content, "python")
        assert len(chunks) > 1

    def test_search_without_index_raises(self):
        """Test search without indexing raises error."""
        from codii.evaluation.coir_adapter import CodiiCoIRAdapter

        adapter = CodiiCoIRAdapter(cleanup_index=True)

        with pytest.raises(RuntimeError, match="not indexed"):
            adapter.search("query")


class TestCoIRMetrics:
    """Tests for CoIR metric computation."""

    @pytest.fixture
    def pytrec_eval_available(self):
        """Check if pytrec_eval is available."""
        try:
            import pytrec_eval
            return True
        except ImportError:
            return False

    def test_compute_metrics_basic(self, pytrec_eval_available):
        """Test basic metric computation."""
        if not pytrec_eval_available:
            pytest.skip("pytrec_eval not installed")

        from scripts.run_coir_benchmark import compute_metrics

        run_results = {
            "q1": {"doc1": 0.9, "doc2": 0.8, "doc3": 0.7},
            "q2": {"doc4": 0.9, "doc5": 0.8},
        }
        qrels = {
            "q1": {"doc1": 1, "doc2": 0},
            "q2": {"doc4": 1},
        }

        metrics = compute_metrics(run_results, qrels)

        assert "ndcg_at_10" in metrics
        assert "mrr_at_10" in metrics
        assert "recall_at_10" in metrics
        assert "recall_at_100" in metrics
        assert "map_score" in metrics

        # All metrics should be between 0 and 1
        for value in metrics.values():
            assert 0.0 <= value <= 1.0

    def test_compute_metrics_perfect_retrieval(self, pytrec_eval_available):
        """Test metrics with perfect retrieval."""
        if not pytrec_eval_available:
            pytest.skip("pytrec_eval not installed")

        from scripts.run_coir_benchmark import compute_metrics

        run_results = {
            "q1": {"doc1": 1.0, "doc2": 0.5},
        }
        qrels = {
            "q1": {"doc1": 1},
        }

        metrics = compute_metrics(run_results, qrels)

        # With perfect retrieval at position 1, metrics should be high
        assert metrics["ndcg_at_10"] > 0.9
        assert metrics["mrr_at_10"] > 0.9
        assert metrics["recall_at_10"] > 0.9

    def test_compute_metrics_no_relevant_docs(self, pytrec_eval_available):
        """Test metrics when no relevant docs are retrieved."""
        if not pytrec_eval_available:
            pytest.skip("pytrec_eval not installed")

        from scripts.run_coir_benchmark import compute_metrics

        run_results = {
            "q1": {"doc1": 0.9, "doc2": 0.8},
        }
        qrels = {
            "q1": {"doc3": 1},  # Relevant doc not retrieved
        }

        metrics = compute_metrics(run_results, qrels)

        assert metrics["ndcg_at_10"] == 0.0
        assert metrics["mrr_at_10"] == 0.0
        assert metrics["recall_at_10"] == 0.0


class TestCoIRTaskLoading:
    """Tests for CoIR task loading."""

    @pytest.mark.skip(reason="Requires network access to HuggingFace")
    def test_load_coir_task(self):
        """Test loading a CoIR task dataset."""
        pytest.importorskip("datasets")
        from scripts.run_coir_benchmark import load_coir_task

        corpus, queries, qrels = load_coir_task("codetrans-dl", limit=10)

        assert len(corpus) > 0
        assert len(queries) > 0
        assert len(qrels) > 0
        assert len(queries) <= 10  # Limit applied


class TestBenchmarkIntegration:
    """Integration tests for benchmark runner."""

    def test_task_result_dataclass(self):
        """Test TaskResult dataclass."""
        from scripts.run_coir_benchmark import TaskResult

        result = TaskResult(
            task="test-task",
            ndcg_at_10=0.5,
            mrr_at_10=0.4,
            recall_at_10=0.6,
            recall_at_100=0.8,
            map_score=0.45,
            num_queries=100,
            num_documents=1000,
        )

        assert result.task == "test-task"
        assert result.error is None

    def test_benchmark_results_dataclass(self):
        """Test BenchmarkResults dataclass."""
        from scripts.run_coir_benchmark import BenchmarkResults, TaskResult

        results = BenchmarkResults(
            benchmark="CoIR",
            model="codii-hybrid",
            timestamp="2024-01-01T00:00:00",
            config={"rerank_enabled": True},
            results=[
                TaskResult(
                    task="task1",
                    ndcg_at_10=0.5,
                    mrr_at_10=0.4,
                    recall_at_10=0.6,
                    recall_at_100=0.8,
                    map_score=0.45,
                    num_queries=100,
                    num_documents=1000,
                ),
            ],
        )

        assert results.benchmark == "CoIR"
        assert len(results.results) == 1

    def test_run_task_with_mock_data(self, sample_corpus, sample_queries, sample_qrels, benchmark_config):
        """Test run_task with mocked dataset loading."""
        try:
            import pytrec_eval
        except ImportError:
            pytest.skip("pytrec_eval not installed")

        from scripts.run_coir_benchmark import run_task

        with patch('scripts.run_coir_benchmark.load_coir_task') as mock_load:
            mock_load.return_value = (sample_corpus, sample_queries, sample_qrels)

            result = run_task(
                "test-task",
                limit=None,
                rerank=False,
                bm25_weight=0.5,
                vector_weight=0.5,
                keep_index=False,
            )

            assert result.task == "test-task"
            assert result.error is None
            assert result.num_queries == len(sample_queries)
            assert result.num_documents == len(sample_corpus)


# Additional test fixtures for benchmark tests
@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def test_output_json_serialization(temp_output_dir):
    """Test that results can be serialized to JSON."""
    from scripts.run_coir_benchmark import BenchmarkResults, TaskResult

    results = BenchmarkResults(
        benchmark="CoIR",
        model="codii-hybrid",
        timestamp="2024-01-01T00:00:00",
        config={"rerank_enabled": True},
        results=[
            TaskResult(
                task="task1",
                ndcg_at_10=0.5,
                mrr_at_10=0.4,
                recall_at_10=0.6,
                recall_at_100=0.8,
                map_score=0.45,
                num_queries=100,
                num_documents=1000,
            ),
        ],
    )

    output_file = temp_output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results.__dict__, f, indent=2, default=lambda x: x.__dict__)

    assert output_file.exists()

    with open(output_file) as f:
        loaded = json.load(f)

    assert loaded["benchmark"] == "CoIR"
    assert len(loaded["results"]) == 1