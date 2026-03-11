"""Excel 数据读取与清洗模块"""
import pandas as pd
import numpy as np
from pathlib import Path
from config import MISSING_RATE_THRESHOLD


def load_excel(file_path: str) -> tuple[pd.DataFrame, list[str]]:
    """
    读取 Excel 文件，返回 (DataFrame, 所有列名列表)
    对列名做基础清洗（去首尾空格）
    """
    pd.set_option("future.no_silent_downcasting", True)
    df = pd.read_excel(file_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df, list(df.columns)


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """
    返回可参与分析的数值列列表：
    - 能被 pd.to_numeric 转换
    - 缺失率 < 50%
    - 有方差（非常数列）
    """
    numeric_cols = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().mean() >= MISSING_RATE_THRESHOLD:
            continue
        if series.nunique() <= 1:
            continue
        numeric_cols.append(col)
    return numeric_cols


def preprocess_for_analysis(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, int, int]:
    """
    针对指定列做数据清洗：
    - 替换字符串 null 值
    - 强制转换数值
    - 筛选有效列（缺失率 < 阈值，有方差）
    - dropna

    Returns:
        (clean_df, raw_row_count, valid_row_count)
    """
    raw_row_count = len(df)

    # 替换常见 null 字符串
    df = df.copy()
    df.replace(["(null)", "null", "NULL", "None", "N/A", "na", "NA"], np.nan, inplace=True)
    df = df.infer_objects(copy=False)

    # 只保留目标列
    subset = df[columns].copy()

    # 转换数值
    for col in subset.columns:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")

    # 过滤缺失率过高或常数列
    valid_cols = []
    for col in subset.columns:
        if subset[col].isna().mean() < MISSING_RATE_THRESHOLD and subset[col].nunique() > 1:
            valid_cols.append(col)

    clean_df = subset[valid_cols].dropna()
    valid_row_count = len(clean_df)

    return clean_df, raw_row_count, valid_row_count
