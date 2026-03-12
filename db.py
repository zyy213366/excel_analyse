"""
SQLite 持久化层
- uploads  表：记录上传文件的 file_id ↔ 磁盘路径映射
- analyses 表：记录每次分析的元数据（历史记录）
"""
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from config import BASE_DIR

DB_PATH = BASE_DIR / "data.db"

# 线程本地连接
_local = threading.local()


def get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return _local.conn


def init_db() -> None:
    """建表（幂等，已存在则忽略）"""
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS uploads (
            file_id          TEXT PRIMARY KEY,
            original_name    TEXT NOT NULL,
            save_path        TEXT NOT NULL,
            row_count        INTEGER DEFAULT 0,
            numeric_cols     TEXT DEFAULT '',
            created_at       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS analyses (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id          TEXT NOT NULL,
            original_name    TEXT NOT NULL,
            instruction      TEXT DEFAULT '',
            mode             TEXT DEFAULT '',
            report_filename  TEXT DEFAULT '',
            created_at       TEXT NOT NULL
        );
    """)
    conn.commit()


# ── uploads 表操作 ──────────────────────────────

def save_upload(file_id: str, original_name: str,
                save_path: str, row_count: int,
                numeric_cols: list[str]) -> None:
    import json
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO uploads
           (file_id, original_name, save_path, row_count, numeric_cols, created_at)
           VALUES (?,?,?,?,?,?)""",
        (file_id, original_name, save_path,
         row_count, json.dumps(numeric_cols, ensure_ascii=False),
         datetime.now().isoformat()),
    )
    conn.commit()


def get_upload(file_id: str) -> sqlite3.Row | None:
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM uploads WHERE file_id = ?", (file_id,)
    ).fetchone()


def load_all_uploads() -> dict[str, Path]:
    """启动时重建内存 dict：file_id → Path（只保留磁盘上仍存在的文件）"""
    conn = get_conn()
    rows = conn.execute("SELECT file_id, save_path FROM uploads").fetchall()
    result = {}
    for row in rows:
        p = Path(row["save_path"])
        if p.exists():
            result[row["file_id"]] = p
    return result


# ── analyses 表操作 ─────────────────────────────

def save_analysis(file_id: str, original_name: str,
                  instruction: str, mode: str,
                  report_filename: str) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT INTO analyses
           (file_id, original_name, instruction, mode, report_filename, created_at)
           VALUES (?,?,?,?,?,?)""",
        (file_id, original_name, instruction, mode,
         report_filename, datetime.now().isoformat()),
    )
    conn.commit()


def get_analyses(limit: int = 50) -> list[sqlite3.Row]:
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
