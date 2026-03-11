"""临时文件管理与输出路径工具"""
import os
from datetime import datetime
from pathlib import Path
from config import OUTPUTS_DIR, MAX_REPORTS_KEEP


def get_output_path(original_filename: str, suffix: str = "分析报告") -> Path:
    """
    生成带时间戳的输出路径，防止文件覆盖。
    例：sales_20240311_143022_分析报告.xlsx
    """
    base = Path(original_filename).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base}_{timestamp}_{suffix}.xlsx"
    return OUTPUTS_DIR / filename


def cleanup_old_reports() -> int:
    """
    清理旧报告，保留最近 MAX_REPORTS_KEEP 个 .xlsx 文件。
    返回删除的文件数量。
    """
    files = sorted(
        OUTPUTS_DIR.glob("*.xlsx"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    to_delete = files[MAX_REPORTS_KEEP:]
    for f in to_delete:
        try:
            f.unlink()
        except OSError:
            pass
    return len(to_delete)


def check_file_writable(path: Path) -> bool:
    """检查文件是否可写（未被其他进程占用）"""
    if not path.exists():
        return True
    try:
        with open(path, "a"):
            return True
    except PermissionError:
        return False
