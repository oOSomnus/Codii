"""SQLite database management for codii."""

import re
import sqlite3
from pathlib import Path
from typing import Optional, List
import threading


def preprocess_fts_query(query: str, use_or: bool = True, add_wildcards: bool = True) -> str:
    """
    Preprocess a query for FTS5 full-text search.

    Converts multi-word queries to use OR matching for better recall,
    and adds wildcard suffixes for partial matching.

    Args:
        query: The search query
        use_or: If True, convert multi-word queries to OR-based matching
        add_wildcards: If True, add wildcard suffixes to terms

    Returns:
        Preprocessed FTS5 query string

    Examples:
        >>> preprocess_fts_query("page table walk")
        'page* OR table* OR walk*'
        >>> preprocess_fts_query("kalloc")
        'kalloc*'
    """
    if not query or not query.strip():
        return ""

    # Remove ALL non-alphanumeric characters except spaces and underscores
    # This ensures FTS5 syntax errors don't occur
    cleaned = re.sub(r'[^a-zA-Z0-9\s_]', ' ', query)

    # Split into terms
    terms = cleaned.split()

    if not terms:
        return ""

    processed_terms = []
    for term in terms:
        term = term.strip()
        if not term:
            continue

        # Skip very short terms that are just numbers or single chars
        if len(term) <= 1:
            continue

        # Skip pure numbers (they don't help with semantic search)
        if term.isdigit():
            continue

        # Add wildcard suffix for partial matching
        if add_wildcards and not term.endswith('*'):
            term = f"{term}*"

        processed_terms.append(term)

    if not processed_terms:
        return ""

    # Join with OR for better recall
    if use_or and len(processed_terms) > 1:
        return ' OR '.join(processed_terms)
    else:
        return processed_terms[0] if len(processed_terms) == 1 else ' OR '.join(processed_terms)


class Database:
    """Thread-safe SQLite database manager."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self.conn

        # Create chunks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                path TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                language TEXT,
                chunk_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create FTS5 virtual table for BM25 search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                path,
                language,
                content='chunks',
                content_rowid='id'
            )
        """)

        # Create triggers to keep FTS in sync
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content, path, language)
                VALUES (new.id, new.content, new.path, new.language);
            END
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, path, language)
                VALUES('delete', old.id, old.content, old.path, old.language);
            END
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF content, path, language ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, path, language)
                VALUES('delete', old.id, old.content, old.path, old.language);
                INSERT INTO chunks_fts(rowid, content, path, language)
                VALUES (new.id, new.content, new.path, new.language);
            END
        """)

        # Create files table for Merkle tree tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index on path for faster lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)
        """)

        conn.commit()

    def insert_chunk(
        self,
        content: str,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> int:
        """Insert a chunk and return its ID."""
        cursor = self.conn.execute(
            """
            INSERT INTO chunks (content, path, start_line, end_line, language, chunk_type)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (content, path, start_line, end_line, language, chunk_type),
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_chunks_batch(self, chunks: list) -> None:
        """Insert multiple chunks in a batch."""
        self.conn.executemany(
            """
            INSERT INTO chunks (content, path, start_line, end_line, language, chunk_type)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            chunks,
        )
        self.conn.commit()

    def search_bm25(self, query: str, limit: int = 10, path_filter: Optional[str] = None) -> list:
        """Search using BM25 via FTS5.

        Uses query preprocessing to convert multi-word queries to OR-based
        matching with wildcards for better recall.
        """
        # Preprocess the query for better FTS5 matching
        fts_query = preprocess_fts_query(query, use_or=True, add_wildcards=True)

        # Escape single quotes for SQL
        escaped_query = fts_query.replace("'", "''")

        if path_filter:
            sql = f"""
                SELECT
                    c.id, c.content, c.path, c.start_line, c.end_line, c.language, c.chunk_type,
                    bm25(chunks_fts) as score
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.rowid = c.id
                WHERE chunks_fts MATCH ?
                AND c.path LIKE ?
                ORDER BY score
                LIMIT ?
            """
            cursor = self.conn.execute(sql, (escaped_query, f"%{path_filter}%", limit))
        else:
            sql = f"""
                SELECT
                    c.id, c.content, c.path, c.start_line, c.end_line, c.language, c.chunk_type,
                    bm25(chunks_fts) as score
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.rowid = c.id
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """
            cursor = self.conn.execute(sql, (escaped_query, limit))

        return cursor.fetchall()

    def get_chunk_by_id(self, chunk_id: int) -> Optional[dict]:
        """Get a chunk by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM chunks WHERE id = ?",
            (chunk_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_chunk_ids(self) -> list:
        """Get all chunk IDs."""
        cursor = self.conn.execute("SELECT id FROM chunks")
        return [row["id"] for row in cursor.fetchall()]

    def get_chunk_ids_by_path(self, path: str) -> List[int]:
        """Get all chunk IDs for a given file path."""
        cursor = self.conn.execute(
            "SELECT id FROM chunks WHERE path = ?",
            (path,),
        )
        return [row["id"] for row in cursor.fetchall()]

    def delete_chunks_by_path(self, path: str) -> int:
        """Delete all chunks for a given path."""
        cursor = self.conn.execute(
            "DELETE FROM chunks WHERE path = ?",
            (path,),
        )
        self.conn.commit()
        return cursor.rowcount

    def clear_all_chunks(self) -> int:
        """Delete all chunks."""
        cursor = self.conn.execute("DELETE FROM chunks")
        self.conn.commit()
        return cursor.rowcount

    def upsert_file_hash(self, path: str, file_hash: str) -> None:
        """Insert or update file hash."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO files (path, hash, last_modified)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (path, file_hash),
        )
        self.conn.commit()

    def get_file_hash(self, path: str) -> Optional[str]:
        """Get stored hash for a file."""
        cursor = self.conn.execute(
            "SELECT hash FROM files WHERE path = ?",
            (path,),
        )
        row = cursor.fetchone()
        return row["hash"] if row else None

    def get_all_file_hashes(self) -> dict:
        """Get all file hashes."""
        cursor = self.conn.execute("SELECT path, hash FROM files")
        return {row["path"]: row["hash"] for row in cursor.fetchall()}

    def delete_file_hash(self, path: str) -> None:
        """Delete file hash entry."""
        self.conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self.conn.commit()

    def clear_all_file_hashes(self) -> None:
        """Delete all file hashes."""
        self.conn.execute("DELETE FROM files")
        self.conn.commit()

    def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM chunks")
        return cursor.fetchone()["count"]

    def get_file_count(self) -> int:
        """Get number of indexed files."""
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM files")
        return cursor.fetchone()["count"]

    def close(self) -> None:
        """Close the connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None