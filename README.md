<p align="center">
  <img src="assets/codii.png" alt="Codii" width="200">
</p>

# Codii - Local Code Repository Indexing MCP Server

A local code repository indexing MCP (Model Context Protocol) server with hybrid BM25 and vector search capabilities.

## Architecture Overview

```mermaid
graph TB
    subgraph "MCP Client"
        C[Claude Desktop / MCP Inspector]
    end

    subgraph "Codii MCP Server"
        S[FastMCP Server]

        subgraph "Tools"
            T1[index_codebase]
            T2[search_code]
            T3[clear_index]
            T4[get_indexing_status]
        end

        subgraph "Core Components"
            CH[AST Chunker<br/>tree-sitter]
            EM[Embedder<br/>all-MiniLM-L6-v2]
            MK[Merkle Tree<br/>Change Detection]
        end

        subgraph "Indexers"
            BM25[BM25 Indexer<br/>SQLite FTS5]
            VEC[Vector Indexer<br/>HNSW]
            HYB[Hybrid Search<br/>RRF]
        end

        subgraph "Storage"
            DB[(SQLite DB)]
            IDX[(HNSW Index)]
            SNAP[(Snapshot JSON)]
            MKC[(Merkle Cache)]
        end
    end

    subgraph "File System"
        FS[Code Repository]
    end

    C -->|MCP Protocol| S
    S --> T1 & T2 & T3 & T4
    T1 --> CH --> EM --> BM25 & VEC
    T1 --> MK
    T2 --> HYB
    HYB --> BM25 & VEC
    BM25 --> DB
    VEC --> IDX
    T4 --> SNAP
    MK --> MKC
    T1 --> FS
    T3 --> DB & IDX & MKC & SNAP
```

## Data Flow

```mermaid
sequenceDiagram
    participant C as MCP Client
    participant S as Codii Server
    participant CH as AST Chunker
    participant EM as Embedder
    participant BM25 as BM25 Index
    participant VEC as Vector Index
    participant FS as File System

    Note over C,FS: Indexing Flow
    C->>S: index_codebase(path)
    S->>FS: Scan directory
    S->>S: Build Merkle tree
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
    S->>S: Save Merkle tree

    Note over C,FS: Search Flow
    C->>S: search_code(query)
    S->>BM25: BM25 search
    S->>VEC: Vector search
    S->>S: Reciprocal Rank Fusion
    S-->>C: Search results
```

## Features

- **Hybrid Search**: Combines BM25 (SQLite FTS5) and vector search (HNSW) for optimal code retrieval
- **AST-Aware Chunking**: Uses tree-sitter for semantic code splitting (functions, classes, etc.)
- **Incremental Updates**: Merkle tree-based change detection for efficient re-indexing
- **Local Embeddings**: CPU-runnable all-MiniLM-L6-v2 model for vector embeddings
- **Multi-Language Support**: Python, JavaScript, TypeScript, Go, Rust, Java, C/C++
- **Gitignore Support**: Automatically respects `.gitignore` patterns when indexing

## Installation

```bash
# Clone and install
git clone https://github.com/oOSomnus/Codii.git
cd codii
uv pip install -e .
```

## Uninstallation

```bash
# Remove the package
uv pip uninstall codii

# Remove Claude Code integration
claude mcp remove codii

# Optional: Remove all index data
rm -rf ~/.codii/
```

## Usage

### Running the MCP Server

```bash
# Using uv
uv run python -m codii.server

# Or directly
codii
```

### MCP Tools

#### `index_codebase`

Index a codebase for semantic search.

```python
{
    "path": "/path/to/repo",        # Required: Absolute path
    "force": false,                  # Optional: Force re-index
    "splitter": "ast",               # Optional: "ast" or "langchain"
    "customExtensions": [".md"],     # Optional: Additional extensions
    "ignorePatterns": ["tests/"]     # Optional: Additional ignore patterns
}
```

#### `search_code`

Search indexed code.

```python
{
    "path": "/path/to/repo",    # Required: Absolute path
    "query": "function to sort", # Required: Search query
    "limit": 10,                 # Optional: Max results (default 10, max 50)
    "extensionFilter": [".py"]   # Optional: Filter by extension
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

After installing the package locally, add it to Claude Code:

```bash
# Add the MCP server to Claude Code
claude mcp add --transport stdio codii -- uv run --directory /path/to/codii python -m codii.server
```

Or if you've installed the package:

```bash
# After running: uv pip install -e .
claude mcp add --transport stdio codii -- codii
```

For manual configuration, edit `~/.claude/settings.json`:

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
      "command": "uv",
      "args": ["run", "--directory", "/path/to/codii", "python", "-m", "codii.server"],
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
│   ├── server.py              # MCP server entry point
│   ├── tools/                 # MCP tool implementations
│   │   ├── index_codebase.py
│   │   ├── search_code.py
│   │   ├── clear_index.py
│   │   └── status.py
│   ├── indexers/              # Search indexers
│   │   ├── bm25_indexer.py    # SQLite FTS5
│   │   ├── vector_indexer.py  # HNSW
│   │   └── hybrid_search.py   # RRF combination
│   ├── chunkers/              # Code chunking
│   │   ├── ast_chunker.py     # tree-sitter based
│   │   └── text_chunker.py    # Fallback
│   ├── embedding/             # Embedding utilities
│   │   └── embedder.py
│   ├── merkle/                # Change detection
│   │   └── tree.py
│   ├── storage/               # Persistence
│   │   ├── database.py
│   │   └── snapshot.py
│   └── utils/                 # Utilities
│       ├── config.py
│       └── file_utils.py
└── pyproject.toml
```

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT