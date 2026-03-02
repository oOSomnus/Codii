<p align="center">
  <img src="assets/codii.png" alt="Codii">
</p>

# Codii - Local Code Repository Indexing with Hybrid Search

A local code repository indexing tool with hybrid BM25 and vector search capabilities. Available as both a CLI tool and an MCP (Model Context Protocol) server.

## Architecture Overview

```mermaid
graph TB
    subgraph Clients
        CLI[Terminal / Shell]
        C[Claude Desktop]
    end

    subgraph Codii
        subgraph Entry
            CLIM[CLI Module]
            SVR[MCP Server]
        end

        subgraph Tools
            T1[index_codebase]
            T2[search_code]
            T3[clear_index]
            T4[get_indexing_status]
        end

        subgraph Core
            CH[AST Chunker]
            EM[Embedder]
            CE[Cross-Encoder]
            MK[Merkle Tree]
        end

        subgraph Indexers
            BM25[BM25 Indexer]
            VEC[Vector Indexer]
            HYB[Hybrid Search]
        end

        subgraph Storage
            DB[(SQLite DB)]
            IDX[(HNSW Index)]
            SNAP[(Snapshot)]
            MKC[(Merkle Cache)]
        end
    end

    FS[Code Repository]

    CLI --> CLIM
    C --> SVR
    CLIM --> T1
    CLIM --> T2
    CLIM --> T3
    CLIM --> T4
    SVR --> T1
    SVR --> T2
    SVR --> T3
    SVR --> T4
    T1 --> CH
    CH --> EM
    EM --> BM25
    EM --> VEC
    T1 --> MK
    T1 --> FS
    T2 --> HYB
    HYB --> BM25
    HYB --> VEC
    HYB --> CE
    BM25 --> DB
    VEC --> IDX
    T4 --> SNAP
    MK --> MKC
    T3 --> DB
    T3 --> IDX
    T3 --> MKC
    T3 --> SNAP
```

## Data Flow

```mermaid
sequenceDiagram
    participant C as MCP Client
    participant S as Codii Server
    participant MK as Merkle Tree
    participant CH as AST Chunker
    participant EM as Embedder
    participant BM25 as BM25 Index
    participant VEC as Vector Index
    participant FS as File System

    Note over C,FS: Indexing Flow
    C->>S: index_codebase(path)
    S->>FS: Scan directory
    S->>MK: Build new Merkle tree
    S->>MK: Compare with old tree (if exists)

    alt No changes detected
        S-->>C: Already indexed, no changes
    else Changes detected (incremental)
        S-->>C: Indexing started (async)

        Note over S,VEC: DELETE phase - removed/modified files
        loop For each removed/modified file
            S->>BM25: Delete chunks by path
            S->>VEC: Remove vectors by chunk IDs
        end

        Note over S,VEC: ADD phase - added/modified files
        loop For each added/modified file
            FS->>CH: File content
            CH->>CH: Parse AST
            CH->>CH: Extract chunks
            CH->>BM25: Store chunks
            CH->>EM: Get embeddings
            EM->>VEC: Store vectors
        end

        S->>S: Save snapshot
        S->>MK: Save Merkle tree
    else New codebase (full index)
        S-->>C: Indexing started (async)

        loop For each file
            FS->>CH: File content
            CH->>CH: Parse AST
            CH->>CH: Extract chunks
            CH->>BM25: Store chunks
            CH->>EM: Get embeddings
            EM->>VEC: Store vectors
        end

        S->>S: Save snapshot
        S->>MK: Save Merkle tree
    end

    Note over C,FS: Search Flow
    C->>S: search_code(query)
    S->>BM25: BM25 search (50 candidates)
    S->>VEC: Vector search (50 candidates)
    S->>S: Reciprocal Rank Fusion
    S->>S: RRF reduction (top 20)
    S->>S: Cross-Encoder Re-ranking
    S-->>C: Search results
```

## Features

- **Hybrid Search**: Combines BM25 (SQLite FTS5) and vector search (HNSW) for optimal code retrieval
- **Cross-Encoder Re-ranking**: Results are re-scored with a cross-encoder for improved relevance (disabled by default for faster searches; enable with `rerank=true`)
- **Smart Query Processing**: Multi-word queries are optimized with OR-matching, wildcards, code tokenization, and abbreviation expansion for better recall
- **AST-Aware Chunking**: Uses tree-sitter for semantic code splitting (functions, classes, etc.)
- **Parallel Indexing**: CPU-bound operations use parallel processing - AST chunking via ProcessPoolExecutor and multi-threaded HNSW index construction
- **Incremental Updates**: Merkle tree-based change detection for efficient re-indexing - only processes added, modified, or removed files instead of re-indexing everything
- **Local Embeddings**: CPU-runnable all-MiniLM-L6-v2 model for vector embeddings
- **Multi-Language Support**: Python, JavaScript, TypeScript, Go, Rust, Java, C/C++
- **Gitignore Support**: Automatically respects `.gitignore` patterns when indexing

## Prerequisites

This package depends on `hnswlib` which requires C++ compilation. You need Python development headers installed:

**Ubuntu/Debian:**
```bash
sudo apt install python3-dev build-essential
```

**Fedora/RHEL:**
```bash
sudo dnf install python3-devel gcc-c++
```

**macOS:**
```bash
xcode-select --install
```

**Alpine Linux:**
```bash
apk add python3-dev gcc g++ musl-dev
```

## Installation

### Option 1: pipx or uv tool (Recommended)

Both pipx and uv tool provide isolated environments for CLI tools. Use whichever you prefer.

**Using pipx:**

```bash
# Install directly from GitHub
pipx install git+https://github.com/oOSomnus/Codii.git

# Or install from local clone
git clone https://github.com/oOSomnus/Codii.git
cd codii
pipx install .
```

**Using uv tool:**

```bash
# Install directly from GitHub
uv tool install git+https://github.com/oOSomnus/Codii.git

# Or install from local clone
git clone https://github.com/oOSomnus/Codii.git
cd codii
uv tool install .
```

### Option 2: pip with venv

For users who prefer manual environment management.

```bash
git clone https://github.com/oOSomnus/Codii.git
cd codii
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Option 3: uv pip (Development)

For development or users who already use uv and want an editable install.

```bash
git clone https://github.com/oOSomnus/Codii.git
cd codii
uv pip install -e .
```

## Uninstallation

```bash
# If installed with pipx
pipx uninstall codii

# If installed with uv tool
uv tool uninstall codii

# If installed with pip/uv pip
pip uninstall codii

# Remove Claude Code integration (if added)
claude mcp remove codii

# Optional: Remove all index data
rm -rf ~/.codii/
```

**Note:** The package provides two entry points:
- `codii` - CLI tool for direct terminal access
- `codii-server` - MCP server for AI assistant integration

Both are removed when uninstalling the package.

## Usage

Codii provides two interfaces:

1. **CLI Tool** - Direct terminal access for index management and debugging
2. **MCP Server** - Integration with MCP clients like Claude Code

### CLI Commands

After installation, the `codii` CLI tool is available:

```bash
codii --help                    # Show all commands
codii status [PATH]             # Show indexing status (defaults to cwd)
codii list                      # List all indexed codebases
codii inspect QUERY [PATH]      # Search chunks for debugging
codii build [PATH] [--force] [--daemon]  # Build/rebuild index
codii stats [PATH]              # Show detailed statistics
codii clear [PATH] [--all]      # Clear index for path or all
```

#### `codii status`

Show the indexing status of a codebase with color-coded output:

```bash
codii status                    # Status of current directory
codii status /path/to/repo      # Status of specific path
```

#### `codii list`

List all indexed codebases in a table format:

```bash
codii list
```

Output shows path, status, files, chunks, and index size for each codebase.

#### `codii build`

Build or rebuild an index with optional progress bar:

```bash
codii build .                   # Build index with progress bar (foreground)
codii build . --force           # Force full re-index
codii build . --daemon          # Build in background (like MCP behavior)
```

The foreground mode shows a live progress bar with stages: preparing, deleting, chunking, embedding, indexing.

#### `codii inspect`

Search and inspect chunks for debugging:

```bash
codii inspect "function"        # Search current directory
codii inspect "database" /path/to/repo  # Search specific path
codii inspect "query" --limit 20        # More results
codii inspect "query" --raw             # Show full content (no truncation)
```

#### `codii stats`

Show detailed statistics including breakdown by language and chunk type:

```bash
codii stats                     # Stats for current directory
codii stats /path/to/repo       # Stats for specific path
```

#### `codii clear`

Clear an index with confirmation prompt:

```bash
codii clear .                   # Clear current directory (prompts for confirmation)
codii clear . --force           # Skip confirmation prompt
codii clear --all               # Clear all indexed codebases
```

### Running the MCP Server

The MCP server provides tools for AI assistants:

```bash
codii-server                    # Start MCP server (after installation)
```

If running from the source directory without installing:

```bash
# Using uv
uv run python -m codii.server

# Using standard Python
python -m codii.server
```

### MCP Tools

#### `index_codebase`

Index a codebase for semantic search. Automatically detects file changes and performs incremental updates (only processes added/modified/removed files).

```python
{
    "path": "/path/to/repo",        # Required: Absolute path
    "force": false,                  # Optional: Force full re-index (clears existing index)
    "splitter": "ast",               # Optional: "ast" or "langchain"
    "customExtensions": [".md"],     # Optional: Additional extensions
    "ignorePatterns": ["tests/"],    # Optional: Additional ignore patterns
    "workers": 0,                    # Optional: Parallel chunking workers (0 = auto-detect)
    "hnswThreads": 0                 # Optional: HNSW index threads (0 = auto-detect)
}
```

**Behavior:**
- New codebase → Full index
- Already indexed + no file changes → Returns "No changes detected"
- Already indexed + file changes detected → Incremental update (only processes changed files)
- `force=true` → Clears existing index and performs full re-index

**Parallelism:**
- `workers`: Number of parallel processes for AST chunking. Default 0 auto-detects based on CPU count.
- `hnswThreads`: Number of threads for HNSW vector index construction. Default 0 auto-detects.

Use `force=true` only for recovery from corrupted indexes or when you want to reset the index completely.

#### `search_code`

Search indexed code.

```python
{
    "path": "/path/to/repo",    # Required: Absolute path
    "query": "function to sort", # Required: Search query
    "limit": 10,                 # Optional: Max results (default 10, max 50)
    "extensionFilter": [".py"],  # Optional: Filter by extension
    "rerank": false              # Optional: Enable cross-encoder re-ranking (default: false)
}
```

#### `get_indexing_status`

Check indexing progress.

```python
{
    "path": "/path/to/repo"  # Required: Absolute path
}
```

#### `clear_index`

Clear an indexed codebase.

```python
{
    "path": "/path/to/repo"  # Required: Absolute path
}
```

## MCP Client Integration

### Claude Code

After installing the package (via pipx or pip), add the MCP server to Claude Code:

```bash
# Simple method - works after pipx install or pip install
claude mcp add --transport stdio codii -- codii-server
```

For manual configuration, edit `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "codii": {
      "command": "codii-server"
    }
  }
}
```

**Note:** Use `codii-server` (not `codii`) for MCP integration. The `codii` command is the CLI tool.

**Development Setup** (running from source without installing):

```bash
# Add using uv to run from source directory
claude mcp add --transport stdio codii -- uv run --directory /path/to/codii python -m codii.server
```

Or manually:

```json
{
  "mcpServers": {
    "codii": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/codii", "python", "-m", "codii.server"]
    }
  }
}
```

### Custom Storage Location

To use a custom storage location, set the `CODII_BASE_DIR` environment variable:

```json
{
  "mcpServers": {
    "codii": {
      "command": "codii",
      "env": {
        "CODII_BASE_DIR": "/custom/storage/path"
      }
    }
  }
}
```

### First Run Note

On first run, the embedding model (`all-MiniLM-L6-v2`) will be downloaded, which may take a few minutes.

## Storage

All index data is stored in `~/.codii/`:

```
~/.codii/
├── indexes/                    # SQLite databases per codebase
│   └── <hash-of-path>/
│       ├── chunks.db           # SQLite with FTS5
│       └── vectors.bin         # HNSW index
├── snapshots/
│   └── snapshot.json           # Index state tracking
└── merkle/
    └── <hash-of-path>.json     # Merkle tree cache per codebase
```

For detailed information about database schemas, dataclasses, and file formats, see [docs/schemas.md](docs/schemas.md).


## Configuration

Create a `.codii.yaml` file in your project root:

```yaml
# Custom ignore patterns
ignore_patterns:
  - "dist/"
  - "*.generated.*"

# Custom file extensions
extensions:
  - ".kt"
  - ".scala"

# Embedding settings
embedding_model: "all-MiniLM-L6-v2"
embedding_batch_size: 32

# Chunk settings
max_chunk_size: 1500
min_chunk_size: 100
```

## Environment Variables

- `CODII_BASE_DIR`: Override the default storage directory

## Supported Languages

| Language | AST Chunking |
|----------|-------------|
| Python   | ✅          |
| JavaScript | ✅        |
| TypeScript | ✅        |
| Go       | ✅          |
| Rust     | ✅          |
| Java     | ✅          |
| C        | ✅          |
| C++      | ✅          |
| Others   | Text-based fallback |

## Project Structure

```
codii/
├── src/codii/
│   ├── cli.py                 # CLI entry point
│   ├── server.py              # MCP server entry point
│   ├── tools/                 # MCP tool implementations
│   │   ├── index_codebase.py
│   │   ├── search_code.py
│   │   ├── clear_index.py
│   │   └── status.py
│   ├── indexers/              # Search indexers
│   │   ├── bm25_indexer.py    # SQLite FTS5
│   │   ├── vector_indexer.py  # HNSW
│   │   ├── hybrid_search.py   # RRF combination
│   │   └── query_processor.py # Query preprocessing
│   ├── chunkers/              # Code chunking
│   │   ├── ast_chunker.py     # tree-sitter based
│   │   └── text_chunker.py    # Fallback
│   ├── embedding/             # Embedding utilities
│   │   ├── embedder.py
│   │   └── cross_encoder.py   # Re-ranking model
│   ├── evaluation/            # Benchmark evaluation
│   │   └── coir_adapter.py    # CoIR benchmark adapter
│   ├── merkle/                # Change detection
│   │   └── tree.py
│   ├── storage/               # Persistence
│   │   ├── database.py
│   │   └── snapshot.py
│   └── utils/                 # Utilities
│       ├── config.py
│       └── file_utils.py
├── scripts/
│   └── run_coir_benchmark.py  # Benchmark evaluation script
└── pyproject.toml
```

## Benchmark Evaluation

Codii includes a CoIR (Code Information Retrieval) benchmark adapter for evaluating search quality on standard code retrieval tasks.

### Installation

```bash
# Install with benchmark dependencies
uv pip install -e ".[benchmark]"

# Or with pip
pip install -e ".[benchmark]"
```

### Running Benchmarks

```bash
# Run all CoIR tasks
python scripts/run_coir_benchmark.py --output results/

# Run specific tasks
python scripts/run_coir_benchmark.py --tasks codetrans-dl,stackoverflow-qa

# Quick test with limited samples
python scripts/run_coir_benchmark.py --tasks codetrans-dl --limit 100

# Enable re-ranking for evaluation
python scripts/run_coir_benchmark.py --tasks codetrans-dl

# Clean up datasets after run
python scripts/run_coir_benchmark.py --tasks codetrans-dl --cleanup-datasets
```

### Available Tasks

- `codetrans-dl` - Code translation (deep learning)
- `codetrans-contest` - Code translation (contest)
- `cosqa` - Code search QA
- `stackoverflow-qa` - StackOverflow QA
- `apps` - APPS dataset
- `codefeedback-mt` - Code feedback (multi-turn)
- `codefeedback-st` - Code feedback (single-turn)
- `codetranspool` - Code translation pool
- `codesearchnet` - CodeSearchNet
- `stackoverflow-qa-mr` - StackOverflow QA (multi-round)

### Metrics Reported

- **NDCG@10** - Normalized Discounted Cumulative Gain at 10
- **MRR@10** - Mean Reciprocal Rank at 10
- **Recall@10** - Recall at 10
- **Recall@100** - Recall at 100
- **MAP** - Mean Average Precision

## Development

```bash
# Install dev dependencies (using uv)
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT