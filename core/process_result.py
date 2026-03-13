"""
ProcessResult — Excel处理 / 数据运算操作的统一返回值数据类
"""
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class ProcessResult:
    """
    Excel处理和数据运算操作的结果载体。
    与 AnalysisResult 分离，语义更清晰。
    """
    # 操作类型标识
    operation: str          # clean / lookup / count / merge / split /
                            # calc_sum / calc_mean / calc_extremes /
                            # calc_logic / calc_aggregate

    # 核心文本摘要（Markdown，展示在聊天气泡中）
    summary_text: str = ""

    # 结果表格（供前端渲染表格 / 生成 Excel）
    result_df: Optional[pd.DataFrame] = None

    # 多张结果表（拆分操作会生成多个分组）
    result_sheets: dict = field(default_factory=dict)  # {sheet_name: DataFrame}

    # 可下载的 Excel 文件名（已写入 outputs/）
    report_filename: Optional[str] = None

    # 处理前后行数
    raw_row_count: int = 0
    valid_row_count: int = 0

    # 操作附加信息（操作描述、清洗统计等）
    extra: dict = field(default_factory=dict)
