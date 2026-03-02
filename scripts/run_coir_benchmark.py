#!/usr/bin/env python3
"""Run CoIR benchmark evaluation for codii.

This script evaluates codii's code search performance using the CoIR benchmark.
It supports running on specific tasks or all tasks, with options for limiting
dataset size and outputting results in JSON format.

Usage:
    # Run all tasks
    python scripts/run_coir_benchmark.py --output results/

    # Run specific tasks
    python scripts/run_coir_benchmark.py --tasks codetrans-dl,stackoverflow-qa

    # Quick test with limited samples
    python scripts/run_coir_benchmark.py --tasks codetrans-dl --limit 100

    # Clean up datasets after run
    python scripts/run_coir_benchmark.py --tasks codetrans-dl --cleanup-datasets
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codii.evaluation.coir_adapter import CodiiCoIRAdapter
from codii.utils.config import CodiiConfig, get_config, set_config


@dataclass
class TaskResult:
    """Results for a single CoIR task."""
    task: str
    ndcg_at_10: float
    mrr_at_10: float
    recall_at_10: float
    recall_at_100: float
    map_score: float
    num_queries: int
    num_documents: int
    error: Optional[str] = None


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    benchmark: str
    model: str
    timestamp: str
    config: dict
    results: List[TaskResult]


# CoIR task names mapping
COIR_TASKS = {
    "codetrans-dl": "CoIRCodeTransDLRetrieval",
    "codetrans-contest": "CoIRCodeTransContestRetrieval",
    "cosqa": "CoIRCosQARetrieval",
    "stackoverflow-qa": "CoIRStackOverflowQARetrieval",
    "apps": "CoIRAppsRetrieval",
    "codefeedback-mt": "CoIRCodeFeedbackMTRetrieval",
    "codefeedback-st": "CoIRCodeFeedbackSTRetrieval",
    "codetranspool": "CoIRCodeTransPoolRetrieval",
    "codesearchnet": "CoIRCodeSearchNetRetrieval",
    "stackoverflow-qa-mr": "CoIRStackOverflowQAMRRetrieval",
}


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    missing = []
    try:
        import mteb
    except ImportError:
        missing.append("mteb")

    try:
        import datasets
    except ImportError:
        missing.append("datasets")

    try:
        import pytrec_eval
    except ImportError:
        missing.append("pytrec-eval-terrier")

    if missing:
        print(f"Error: Missing required packages: {', '.join(missing)}")
        print("Install with: uv pip install -e '.[benchmark]'")
        return False
    return True


def load_coir_task(task_name: str, limit: Optional[int] = None):
    """Load a CoIR task dataset.

    Args:
        task_name: Name of the CoIR task (e.g., "codetrans-dl")
        limit: Optional limit on number of queries to evaluate

    Returns:
        Tuple of (corpus, queries, qrels) dictionaries

    Note:
        corpus: dict of {doc_id: text_content}
        queries: dict of {query_id: query_text}
        qrels: dict of {query_id: {doc_id: relevance}}
    """
    import coir

    # Get the MTEB task name
    mteb_task_name = COIR_TASKS.get(task_name)
    if not mteb_task_name:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(COIR_TASKS.keys())}")

    print(f"Loading dataset: {task_name}...")

    try:
        # Use coir's built-in data loader
        # Returns:
        #   corpus: dict of {doc_id: {'text': str, 'title': str}}
        #   queries: dict of {query_id: query_text}
        #   qrels: dict of {query_id: {doc_id: relevance}}
        corpus_raw, queries, qrels = coir.load_data_from_hf(task_name)
    except Exception as e:
        print(f"Error loading dataset {task_name}: {e}")
        raise

    # Convert corpus from {doc_id: {'text': ...}} to {doc_id: text}
    corpus = {}
    for doc_id, doc_data in corpus_raw.items():
        if isinstance(doc_data, dict):
            corpus[doc_id] = doc_data.get('text', doc_data.get('contents', ''))
        else:
            corpus[doc_id] = str(doc_data)

    # Apply limit if specified
    if limit and limit < len(queries):
        query_ids = list(queries.keys())[:limit]
        queries = {qid: queries[qid] for qid in query_ids}
        qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}

    print(f"Loaded {len(corpus)} documents, {len(queries)} queries, {len(qrels)} relevance judgments")

    return corpus, queries, qrels


def compute_metrics(
    run_results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int] = [10, 100],
) -> Dict[str, float]:
    """Compute retrieval metrics using pytrec_eval.

    Args:
        run_results: Dict of {query_id: {doc_id: score}}
        qrels: Dict of {query_id: {doc_id: relevance}}
        k_values: Values of k for computing metrics

    Returns:
        Dictionary of metric names to values
    """
    import pytrec_eval

    # Create pytrec_eval evaluator
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {
            "ndcg_cut",
            "recip_rank",
            "recall",
            "map",
        },
    )

    # Evaluate
    results = evaluator.evaluate(run_results)

    # Aggregate metrics
    metrics = {}
    for measure in ["ndcg_cut_10", "recip_rank", "recall_10", "recall_100", "map"]:
        values = [r.get(measure, 0.0) for r in results.values()]
        metrics[measure] = sum(values) / len(values) if values else 0.0

    # Rename for clarity
    return {
        "ndcg_at_10": metrics.get("ndcg_cut_10", 0.0),
        "mrr_at_10": metrics.get("recip_rank", 0.0),
        "recall_at_10": metrics.get("recall_10", 0.0),
        "recall_at_100": metrics.get("recall_100", 0.0),
        "map_score": metrics.get("map", 0.0),
    }


def run_task(
    task_name: str,
    limit: Optional[int] = None,
    rerank: bool = False,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    keep_index: bool = False,
    rerank_threshold: float = 0.0,
) -> TaskResult:
    """Run evaluation on a single CoIR task.

    Args:
        task_name: Name of the CoIR task
        limit: Optional limit on queries
        rerank: Enable/disable re-ranking
        bm25_weight: Weight for BM25
        vector_weight: Weight for vector search
        keep_index: Keep temp index after evaluation
        rerank_threshold: Threshold for cross-encoder re-ranking (0 = no filtering)

    Returns:
        TaskResult with metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Running task: {task_name}")
    print(f"{'=' * 60}")

    try:
        # Load dataset
        corpus, queries, qrels = load_coir_task(task_name, limit)

        # Create adapter
        adapter = CodiiCoIRAdapter(
            cleanup_index=not keep_index,
            rerank=rerank,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            rerank_threshold=rerank_threshold,
        )

        # Index corpus
        print(f"Indexing {len(corpus)} documents...")
        adapter.index_corpus(corpus)
        print(f"Created {adapter.get_chunk_count()} chunks from {adapter.get_doc_count()} documents")

        # Run queries
        print(f"Running {len(queries)} queries...")
        run_results: Dict[str, Dict[str, float]] = {}

        for i, (query_id, query_text) in enumerate(queries.items()):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Query {i + 1}/{len(queries)}...")

            # Search with high k to get more candidates for recall@100
            results = adapter.search(query_text, top_k=100)

            # Convert to pytrec_eval format
            run_results[query_id] = {r.doc_id: r.score for r in results}

        # Compute metrics
        print("Computing metrics...")
        metrics = compute_metrics(run_results, qrels)

        return TaskResult(
            task=task_name,
            ndcg_at_10=metrics["ndcg_at_10"],
            mrr_at_10=metrics["mrr_at_10"],
            recall_at_10=metrics["recall_at_10"],
            recall_at_100=metrics["recall_at_100"],
            map_score=metrics["map_score"],
            num_queries=len(queries),
            num_documents=len(corpus),
        )

    except Exception as e:
        import traceback
        print(f"Error running task {task_name}: {e}")
        traceback.print_exc()
        return TaskResult(
            task=task_name,
            ndcg_at_10=0.0,
            mrr_at_10=0.0,
            recall_at_10=0.0,
            recall_at_100=0.0,
            map_score=0.0,
            num_queries=0,
            num_documents=0,
            error=str(e),
        )


def run_benchmark(
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
    output_dir: Optional[str] = None,
    rerank: bool = False,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    keep_index: bool = False,
    cleanup_datasets: bool = False,
    rerank_threshold: float = 0.0,
) -> BenchmarkResults:
    """Run the CoIR benchmark evaluation.

    Args:
        tasks: List of task names to run (None = all tasks)
        limit: Optional limit on queries per task
        output_dir: Directory to save results
        rerank: Enable/disable re-ranking
        bm25_weight: Weight for BM25
        vector_weight: Weight for vector search
        keep_index: Keep temp index after evaluation
        cleanup_datasets: Clean up HuggingFace datasets after run
        rerank_threshold: Threshold for cross-encoder re-ranking (0 = no filtering)

    Returns:
        BenchmarkResults with all task results
    """
    # Default to all tasks if none specified
    if tasks is None:
        tasks = list(COIR_TASKS.keys())

    # Get config
    config = get_config()

    # Create results
    results = BenchmarkResults(
        benchmark="CoIR",
        model="codii-hybrid",
        timestamp=datetime.now().isoformat(),
        config={
            "rerank_enabled": rerank,
            "rerank_threshold": rerank_threshold,
            "bm25_weight": bm25_weight,
            "vector_weight": vector_weight,
            "embedding_model": config.embedding_model,
            "rerank_model": config.rerank_model if rerank else None,
        },
        results=[],
    )

    # Run each task
    for task_name in tasks:
        result = run_task(
            task_name,
            limit=limit,
            rerank=rerank,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            keep_index=keep_index,
            rerank_threshold=rerank_threshold,
        )
        results.results.append(result)

        # Print intermediate results
        if result.error:
            print(f"\n{task_name}: ERROR - {result.error}")
        else:
            print(f"\n{task_name} Results:")
            print(f"  NDCG@10:   {result.ndcg_at_10:.4f}")
            print(f"  MRR@10:    {result.mrr_at_10:.4f}")
            print(f"  Recall@10: {result.recall_at_10:.4f}")
            print(f"  Recall@100:{result.recall_at_100:.4f}")
            print(f"  MAP:       {result.map_score:.4f}")

    # Clean up datasets if requested
    if cleanup_datasets:
        print("\nCleaning up cached datasets...")
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
        if cache_dir.exists():
            import shutil
            for item in cache_dir.iterdir():
                if "coir" in item.name.lower() or "CoIR" in item.name:
                    shutil.rmtree(item, ignore_errors=True)
            print("Datasets cleaned up.")

    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        results_file = output_path / f"coir_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(asdict(results), f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Tasks completed: {len([r for r in results.results if not r.error])}/{len(results.results)}")

    # Compute averages
    valid_results = [r for r in results.results if not r.error]
    if valid_results:
        avg_ndcg = sum(r.ndcg_at_10 for r in valid_results) / len(valid_results)
        avg_mrr = sum(r.mrr_at_10 for r in valid_results) / len(valid_results)
        avg_recall = sum(r.recall_at_10 for r in valid_results) / len(valid_results)
        print(f"\nAverage NDCG@10: {avg_ndcg:.4f}")
        print(f"Average MRR@10:  {avg_mrr:.4f}")
        print(f"Average Recall@10: {avg_recall:.4f}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run CoIR benchmark evaluation for codii",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tasks
  python scripts/run_coir_benchmark.py --output results/

  # Run specific tasks
  python scripts/run_coir_benchmark.py --tasks codetrans-dl,stackoverflow-qa

  # Quick test with limited samples
  python scripts/run_coir_benchmark.py --tasks codetrans-dl --limit 100

Available tasks:
  codetrans-dl, codetrans-contest, cosqa, stackoverflow-qa,
  apps, codefeedback-mt, codefeedback-st, codetranspool,
  codesearchnet, stackoverflow-qa-mr
        """,
    )

    parser.add_argument(
        "--tasks",
        type=str,
        help="Comma-separated list of tasks to run (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of queries per task (for quick testing)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder re-ranking",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.5,
        help="Weight for BM25 scores (default: 0.5)",
    )
    parser.add_argument(
        "--vector-weight",
        type=float,
        default=0.5,
        help="Weight for vector scores (default: 0.5)",
    )
    parser.add_argument(
        "--keep-index",
        action="store_true",
        help="Keep temporary index files after evaluation (for debugging)",
    )
    parser.add_argument(
        "--cleanup-datasets",
        action="store_true",
        help="Clean up HuggingFace dataset cache after run",
    )
    parser.add_argument(
        "--rerank-threshold",
        type=float,
        default=0.0,
        help="Cross-encoder rerank threshold (default: 0.0, no filtering)",
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Parse tasks
    tasks = args.tasks.split(",") if args.tasks else None

    # Run benchmark
    run_benchmark(
        tasks=tasks,
        limit=args.limit,
        output_dir=args.output,
        rerank=not args.no_rerank,
        bm25_weight=args.bm25_weight,
        vector_weight=args.vector_weight,
        keep_index=args.keep_index,
        cleanup_datasets=args.cleanup_datasets,
        rerank_threshold=args.rerank_threshold,
    )


if __name__ == "__main__":
    main()