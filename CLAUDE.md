# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install package
uv pip install -e .

# Install with benchmark dependencies
uv pip install -e ".[benchmark]"

# Run the CLI tool
codii --help
codii status
codii build .
codii list

# Run the MCP server
codii-server

# Or run server directly with uv
uv run python -m codii.server

# Clear all index data (useful during development)
rm -rf ~/.codii/

# Run all tests
pytest

# Run tests with coverage
pytest --cov=codii --cov-report=term-missing

# Run specific test file
pytest tests/test_indexers/test_hybrid_search.py

# Run specific test
pytest tests/test_storage/test_database.py::TestInsertChunk::test_insert_chunk
```

## Entry Points

The package provides two entry points:

1. **`codii`** - CLI tool for direct terminal access
2. **`codii-server`** - MCP server for AI assistant integration

Both are installed when the package is installed.

## Architecture

### Layered Design

The codebase follows a strict layered architecture where dependencies flow downward:

1. **CLI/Tools** (`cli.py`, `tools/`) - User interfaces (CLI commands and MCP tool implementations)
2. **Indexers** (`indexers/`) - Search engines (BM25, Vector, Hybrid)
3. **Chunkers** (`chunkers/`) - Code splitting (AST-based via tree-sitter, text fallback)
4. **Storage** (`storage/`) - SQLite database, snapshot state
5. **Utils** (`utils/`) - Config, file scanning

### Key Patterns

**Singleton Embedder**: The `Embedder` class uses singleton pattern to ensure only one sentence-transformers model is loaded. Access via `get_embedder()`.

**Singleton Cross-Encoder**: The `CrossEncoderWrapper` class uses singleton pattern for re-ranking. Access via `get_cross_encoder()`.

**Background Indexing**: `index_codebase` runs indexing in a background thread. Progress is tracked via `SnapshotManager` in `~/.codii/snapshots/snapshot.json`. Check status with `get_indexing_status`.

**Parallel Indexing**: The indexing pipeline uses parallel processing for CPU-bound operations:
- **Parallel Chunking**: AST parsing uses `ProcessPoolExecutor` with configurable workers. File contents are read in the main process; chunkers are created fresh in each worker to avoid pickling issues with tree-sitter parsers.
- **Multi-threaded HNSW**: Vector index construction uses `hnswlib`'s `num_threads` parameter for parallel insertion.
- **Auto-detection**: Both workers and HNSW threads auto-detect based on CPU count when set to 0 (default).
- Configure via `workers` and `hnswThreads` parameters on the `index_codebase` tool.

**Incremental Indexing**: Re-indexing automatically detects changes via Merkle tree comparison:
- Only processes files that are added, modified, or removed
- DELETE phase: Removes stale chunks from SQLite and vectors from HNSW (soft delete)
- ADD phase: Only chunks and embeds new/modified files
- Early exit if no changes detected
- Use `force=true` to trigger a full re-index

**Hybrid Search**: Uses Reciprocal Rank Fusion (RRF) to combine BM25 and vector search results. See `hybrid_search.py:_reciprocal_rank_fusion()`.

**Multi-Stage Search Pipeline**: The search follows a staged reduction approach:
- Stage 1 (Retrieval): BM25 and Vector each return 50 candidates (`rerank_candidates`)
- Stage 2 (RRF Merge): Reciprocal Rank Fusion merges and reduces to 20 results (`rrf_limit`)
- Stage 3 (Re-ranking): Cross-encoder re-scores candidates for final top-k results

**Cross-Encoder Re-ranking**: A cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the RRF-reduced candidates for improved relevance. **Disabled by default** for faster searches; enable via `rerank=True` parameter or config. Scores are normalized to 0-1 range using sigmoid; results below threshold (0.5) are filtered out.

**Query Preprocessing**: Multi-word queries are preprocessed for better recall:
- OR-based matching: "page table walk" → "page* OR table* OR walk*"
- Wildcard suffixes for partial matching: "kalloc" → "kalloc*"
- Code tokenization: camelCase/snake_case identifiers are split
- Abbreviation expansion: common code abbreviations are expanded
- See `query_processor.py` and `database.py:preprocess_fts_query()`

**Codebase Isolation**: Each codebase gets its own index directory under `~/.codii/indexes/<hash-of-path>/`. The path hash is computed via SHA-256.

**Automatic Gitignore**: The `scan_directory` function automatically reads and applies `.gitignore` patterns from the repository root. Combined with default ignore patterns from config.

### Tree-Sitter Integration

The AST chunker (`chunkers/ast_chunker.py`) uses tree-sitter with language-specific grammars:
- Create `Language` object: `Language(tree_sitter_python.language())`
- Create `Parser` with the language
- Node types for semantic units are defined in `SEMANTIC_NODES` dict

### Database Schema

SQLite with FTS5 virtual table for full-text search:
- `chunks` table stores content with metadata (path, line numbers, language, chunk_type)
- `chunks_fts` is the FTS5 virtual table synced via triggers
- `files` table tracks file hashes for Merkle tree

### Configuration

Config is loaded from (in order):
1. Environment variable `CODII_BASE_DIR`
2. `.codii.yaml` in project root
3. Default `~/.codii/`

Access via `get_config()` from `utils/config.py`.

**Worker/Thread Helpers**: Use `CodiiConfig.get_effective_workers()` and `CodiiConfig.get_effective_hnsw_threads()` to auto-detect parallelism based on CPU cores.

## Adding a New Language

1. Add tree-sitter dependency to `pyproject.toml`
2. Add language to `LANGUAGE_MAP` in `ast_chunker.py`
3. Add semantic node types to `SEMANTIC_NODES` dict
4. Add language to file extension mapping in `file_utils.py:detect_language()`

## Testing

The test suite uses pytest with mocked embeddings. Key patterns:

**Mock Embedder**: Tests use `mock_embedder` fixture in `conftest.py` which returns deterministic vectors. The fixture is auto-applied via `autouse=True`.

**HNSW Requirements**: Vector index tests need 50+ vectors for HNSW reliability. Default `ef_search=100` for better recall on multi-word queries.

**Test Structure**: Tests are organized by component:
- `tests/test_storage/` - Database and snapshot tests
- `tests/test_indexers/` - BM25, vector, hybrid search, and query processor tests
- `tests/test_chunkers/` - AST and text chunker tests (including comprehensive language tests)
- `tests/test_merkle/` - Merkle tree tests
- `tests/test_tools/` - MCP tool tests
- `tests/test_integration/` - End-to-end workflow tests
- `tests/test_benchmark/` - CoIR benchmark evaluation tests

## Benchmark Evaluation

Codii includes a CoIR (Code Information Retrieval) benchmark adapter for evaluating search quality.

```bash
# Install benchmark dependencies
uv pip install -e ".[benchmark]"

# Run all CoIR tasks
python scripts/run_coir_benchmark.py --output results/

# Run specific tasks
python scripts/run_coir_benchmark.py --tasks codetrans-dl,stackoverflow-qa

# Quick test with limited samples
python scripts/run_coir_benchmark.py --tasks codetrans-dl --limit 100

# Enable cross-encoder re-ranking for evaluation
python scripts/run_coir_benchmark.py --tasks codetrans-dl
```

The benchmark script evaluates codii's hybrid search on standard code retrieval tasks and reports NDCG@10, MRR@10, Recall@10, Recall@100, and MAP scores.