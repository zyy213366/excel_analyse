"""
XlsxWriter 多 Sheet 报告生成器
支持三种分析模式的图表和报告输出
"""
from pathlib import Path
import pandas as pd
import xlsxwriter

from core.analysis_engine import AnalysisResult


def build_report(result: AnalysisResult, output_path: str | Path) -> Path:
    """
    根据 AnalysisResult 生成 Excel 报告，返回文件路径。
    自动根据 result.mode 选择对应的报告模板。
    """
    output_path = Path(output_path)
    writer = pd.ExcelWriter(str(output_path), engine="xlsxwriter")
    workbook = writer.book

    # ── 通用格式 ──
    fmt_title = workbook.add_format({"bold": True, "font_size": 16, "font_color": "#2F5496"})
    fmt_header = workbook.add_format({"bold": True, "align": "center", "bg_color": "#D9E1F2", "border": 1})
    fmt_center = workbook.add_format({"align": "center"})
    fmt_num4 = workbook.add_format({"num_format": "0.0000", "align": "center"})
    fmt_pct = workbook.add_format({"num_format": "0.00%", "align": "center"})

    if result.mode == "y_vs_all":
        _build_y_vs_all(writer, workbook, result, fmt_title, fmt_header, fmt_center, fmt_num4, fmt_pct)
    elif result.mode == "two_column":
        _build_two_column(writer, workbook, result, fmt_title, fmt_header, fmt_center, fmt_num4, fmt_pct)
    elif result.mode == "multi_x_vs_y":
        _build_multi_x_vs_y(writer, workbook, result, fmt_title, fmt_header, fmt_center, fmt_num4, fmt_pct)

    writer.close()
    return output_path


# ──────────────────────────────────────────────────────────
# 模式1：Y vs All
# ──────────────────────────────────────────────────────────

def _build_y_vs_all(writer, workbook, result: AnalysisResult,
                    fmt_title, fmt_header, fmt_center, fmt_num4, fmt_pct):
    target_y = result.target_y
    merged = result.feature_importance_df
    corr_df = result.correlation_df
    reg_df = result.regression_df
    plot_df = result.plot_df
    stats_df = result.stats_df

    # ── 数据 Sheet（隐藏）──
    imp_sheet = "权重数据"
    plot_sheet = "图表数据源"
    merged.to_excel(writer, sheet_name=imp_sheet, index=False)
    plot_df.to_excel(writer, sheet_name=plot_sheet, index=False)
    writer.sheets[imp_sheet].hide()
    writer.sheets[plot_sheet].hide()

    # ── Sheet 1: 综合分析报告 ──
    sht = workbook.add_worksheet("综合分析报告")
    sht.set_column("A:A", 40)
    sht.write("A1", f"📊 全维数据深度分析报告 — {target_y}", fmt_title)

    for i, line in enumerate(result.summary_text.split("\n")):
        clean = line.lstrip("#").lstrip()
        sht.write(i + 2, 0, clean)

    # 因子权重柱状图
    count = len(merged)
    chart_bar = workbook.add_chart({"type": "bar", "subtype": "stacked"})
    chart_bar.add_series({
        "name": "正相关因素",
        "categories": [imp_sheet, 1, 0, count, 0],
        "values":     [imp_sheet, 1, 3, count, 3],
        "fill":       {"color": "#70AD47"},
        "data_labels": {"value": True, "num_format": "0%"},
    })
    chart_bar.add_series({
        "name": "负相关因素",
        "categories": [imp_sheet, 1, 0, count, 0],
        "values":     [imp_sheet, 1, 4, count, 4],
        "fill":       {"color": "#C00000"},
        "data_labels": {"value": True, "num_format": "0%"},
    })
    chart_bar.set_title({"name": "全局因子贡献度排名（颜色区分正负相关）"})
    chart_bar.set_x_axis({"name": "贡献权重", "major_gridlines": {"visible": True}})
    chart_bar.set_y_axis({"name": "影响因子", "reverse": True})
    chart_bar.set_legend({"position": "top"})
    dynamic_h = max(400, count * 28)
    chart_bar.set_size({"width": 650, "height": dynamic_h})
    sht.insert_chart("E2", chart_bar)

    # 趋势折线图（目标变量波动）
    data_len = len(plot_df)
    chart_line = workbook.add_chart({"type": "line"})
    chart_line.add_series({
        "name": target_y,
        "values": [plot_sheet, 1, 0, data_len, 0],
        "line": {"color": "#5B9BD5", "width": 2},
    })
    chart_line.set_title({"name": f"{target_y} 数据波动趋势"})
    chart_line.set_style(12)
    chart_line.set_size({"width": 900, "height": 280})
    bar_rows = int(dynamic_h / 18) + 4
    sht.insert_chart(f"A{bar_rows + 2}", chart_line)

    # 6 散点图矩阵
    top_features = [c for c in plot_df.columns if c != target_y]
    start_scatter = bar_rows + 18
    for i, feat in enumerate(top_features[:6]):
        chart_sc = workbook.add_chart({"type": "scatter"})
        x_col_idx = list(plot_df.columns).index(feat)
        chart_sc.add_series({
            "name": "观测点",
            "categories": [plot_sheet, 1, x_col_idx, data_len, x_col_idx],
            "values":     [plot_sheet, 1, 0, data_len, 0],
            "marker":     {"type": "circle", "size": 4, "fill": {"color": "#ED7D31"}},
            "trendline":  {"type": "linear", "line": {"color": "red", "width": 1.5, "dash_type": "dash"}},
        })
        chart_sc.set_title({"name": f"{i+1}. {feat} vs {target_y}"})
        chart_sc.set_x_axis({"name": feat})
        chart_sc.set_y_axis({"name": target_y})
        chart_sc.set_style(10)
        chart_sc.set_size({"width": 450, "height": 300})
        col_letter = "A" if i % 2 == 0 else "G"
        row_pos = start_scatter + (i // 2) * 18
        sht.insert_chart(f"{col_letter}{row_pos}", chart_sc)

    # ── Sheet 2: 单因子相关性 ──
    corr_df[["Feature", "Pearson", "Spearman"]].to_excel(writer, sheet_name="单因子相关性", index=False)
    _format_sheet_headers(writer.sheets["单因子相关性"], workbook, fmt_header)

    # ── Sheet 3: 线性拟合回归 ──
    reg_df.to_excel(writer, sheet_name="线性拟合回归", index=False)
    _format_sheet_headers(writer.sheets["线性拟合回归"], workbook, fmt_header)

    # ── Sheet 4: 数据特征统计 ──
    stats_df.to_excel(writer, sheet_name="数据特征统计")
    _format_sheet_headers(writer.sheets["数据特征统计"], workbook, fmt_header)


# ──────────────────────────────────────────────────────────
# 模式2：Two Column
# ──────────────────────────────────────────────────────────

def _build_two_column(writer, workbook, result: AnalysisResult,
                      fmt_title, fmt_header, fmt_center, fmt_num4, fmt_pct):
    col_a, col_b = result.col_a, result.col_b
    plot_df = result.plot_df
    stats_df = result.stats_df

    # ── 数据 Sheet（隐藏）──
    plot_sheet = "图表数据源"
    plot_df.to_excel(writer, sheet_name=plot_sheet, index=False)
    writer.sheets[plot_sheet].hide()

    # ── Sheet 1: 综合报告 ──
    sht = workbook.add_worksheet("双列关系分析")
    sht.set_column("A:A", 50)
    sht.write("A1", f"📊 双列关系分析报告 — {col_a} ↔ {col_b}", fmt_title)
    for i, line in enumerate(result.summary_text.split("\n")):
        sht.write(i + 2, 0, line.lstrip("#").lstrip())

    data_len = len(plot_df)
    a_col = 0
    b_col = 1

    # 散点图 A vs B（带趋势线）
    chart_sc = workbook.add_chart({"type": "scatter"})
    chart_sc.add_series({
        "name": f"{col_a} vs {col_b}",
        "categories": [plot_sheet, 1, a_col, data_len, a_col],
        "values":     [plot_sheet, 1, b_col, data_len, b_col],
        "marker":     {"type": "circle", "size": 5, "fill": {"color": "#5B9BD5"}},
        "trendline":  {"type": "linear",
                       "display_equation": True,
                       "display_r_squared": True,
                       "line": {"color": "red", "width": 1.5, "dash_type": "dash"}},
    })
    chart_sc.set_title({"name": f"{col_a} vs {col_b} 散点图"})
    chart_sc.set_x_axis({"name": col_a})
    chart_sc.set_y_axis({"name": col_b})
    chart_sc.set_style(10)
    chart_sc.set_size({"width": 500, "height": 350})
    sht.insert_chart("E2", chart_sc)

    # 分布直方图 A
    chart_hist_a = workbook.add_chart({"type": "column"})
    chart_hist_a.add_series({
        "name": col_a,
        "values": [plot_sheet, 1, a_col, data_len, a_col],
        "fill":   {"color": "#70AD47"},
    })
    chart_hist_a.set_title({"name": f"{col_a} 数据分布趋势"})
    chart_hist_a.set_size({"width": 400, "height": 250})
    sht.insert_chart("E22", chart_hist_a)

    # 分布直方图 B
    chart_hist_b = workbook.add_chart({"type": "column"})
    chart_hist_b.add_series({
        "name": col_b,
        "values": [plot_sheet, 1, b_col, data_len, b_col],
        "fill":   {"color": "#ED7D31"},
    })
    chart_hist_b.set_title({"name": f"{col_b} 数据分布趋势"})
    chart_hist_b.set_size({"width": 400, "height": 250})
    sht.insert_chart("N22", chart_hist_b)

    # ── Sheet 2: 回归统计表 ──
    reg_data = {
        "方向": [f"{col_a} → {col_b}", f"{col_b} → {col_a}"],
        "R²": [result.reg_a_to_b["r2"], result.reg_b_to_a["r2"]],
        "斜率": [result.reg_a_to_b["slope"], result.reg_b_to_a["slope"]],
        "截距": [result.reg_a_to_b["intercept"], result.reg_b_to_a["intercept"]],
    }
    pd.DataFrame(reg_data).to_excel(writer, sheet_name="回归统计", index=False)

    # ── Sheet 3: 描述性统计 ──
    stats_df.to_excel(writer, sheet_name="描述性统计")


# ──────────────────────────────────────────────────────────
# 模式3：Multi X vs Y
# ──────────────────────────────────────────────────────────

def _build_multi_x_vs_y(writer, workbook, result: AnalysisResult,
                         fmt_title, fmt_header, fmt_center, fmt_num4, fmt_pct):
    target_y = result.target_y
    x_columns = result.x_columns
    merged = result.feature_importance_df
    coef_df = result.multi_reg_coef_df
    x_corr = result.x_corr_matrix
    plot_df = result.plot_df
    stats_df = result.stats_df

    # ── 数据 Sheet（隐藏）──
    imp_sheet = "权重数据"
    plot_sheet = "图表数据源"
    merged.to_excel(writer, sheet_name=imp_sheet, index=False)
    plot_df.to_excel(writer, sheet_name=plot_sheet, index=False)
    writer.sheets[imp_sheet].hide()
    writer.sheets[plot_sheet].hide()

    # ── Sheet 1: 综合报告 ──
    sht = workbook.add_worksheet("多因素分析报告")
    sht.set_column("A:A", 50)
    sht.write("A1", f"📊 多因素分析报告 — {target_y}", fmt_title)
    for i, line in enumerate(result.summary_text.split("\n")):
        sht.write(i + 2, 0, line.lstrip("#").lstrip())

    # 特征重要性柱状图（同模式1）
    count = len(merged)
    chart_bar = workbook.add_chart({"type": "bar", "subtype": "stacked"})
    chart_bar.add_series({
        "name": "正相关因素",
        "categories": [imp_sheet, 1, 0, count, 0],
        "values":     [imp_sheet, 1, 3, count, 3],
        "fill":       {"color": "#70AD47"},
        "data_labels": {"value": True, "num_format": "0%"},
    })
    chart_bar.add_series({
        "name": "负相关因素",
        "categories": [imp_sheet, 1, 0, count, 0],
        "values":     [imp_sheet, 1, 4, count, 4],
        "fill":       {"color": "#C00000"},
        "data_labels": {"value": True, "num_format": "0%"},
    })
    chart_bar.set_title({"name": "多因素贡献度排名"})
    chart_bar.set_y_axis({"reverse": True})
    chart_bar.set_legend({"position": "top"})
    dynamic_h = max(350, count * 30)
    chart_bar.set_size({"width": 600, "height": dynamic_h})
    sht.insert_chart("E2", chart_bar)

    # 散点图矩阵（X vs Y）
    data_len = len(plot_df)
    col_names = list(plot_df.columns)
    start_sc = int(dynamic_h / 18) + 6
    for i, feat in enumerate(x_columns[:6]):
        x_idx = col_names.index(feat)
        chart_sc = workbook.add_chart({"type": "scatter"})
        chart_sc.add_series({
            "name": f"{feat} vs {target_y}",
            "categories": [plot_sheet, 1, x_idx, data_len, x_idx],
            "values":     [plot_sheet, 1, 0, data_len, 0],
            "marker":     {"type": "circle", "size": 4, "fill": {"color": "#5B9BD5"}},
            "trendline":  {"type": "linear", "line": {"color": "red", "width": 1.5, "dash_type": "dash"}},
        })
        chart_sc.set_title({"name": f"{feat} vs {target_y}"})
        chart_sc.set_style(10)
        chart_sc.set_size({"width": 420, "height": 280})
        col_letter = "A" if i % 2 == 0 else "G"
        row_pos = start_sc + (i // 2) * 17
        sht.insert_chart(f"{col_letter}{row_pos}", chart_sc)

    # ── Sheet 2: 多元回归系数 ──
    coef_df.to_excel(writer, sheet_name="回归系数", index=False)
    sht2 = writer.sheets["回归系数"]
    sht2.set_column("A:A", 20)
    sht2.set_column("B:D", 14)

    # 回归系数柱状图
    k = len(coef_df)
    chart_coef = workbook.add_chart({"type": "bar"})
    chart_coef.add_series({
        "name": "回归系数",
        "categories": ["回归系数", 1, 0, k, 0],
        "values":     ["回归系数", 1, 1, k, 1],
        "fill":       {"color": "#4472C4"},
    })
    chart_coef.set_title({"name": "多元回归系数"})
    chart_coef.set_y_axis({"reverse": True})
    chart_coef.set_size({"width": 500, "height": max(300, k * 30)})
    sht2.insert_chart("F2", chart_coef)

    # ── Sheet 3: X 间相关矩阵 ──
    if x_corr is not None:
        x_corr.to_excel(writer, sheet_name="X相关矩阵")
        sht3 = writer.sheets["X相关矩阵"]
        # 标注共线性警告
        if result.collinearity_warnings:
            row_start = len(x_corr) + 3
            sht3.write(row_start, 0, "⚠ 共线性警告：")
            for j, w in enumerate(result.collinearity_warnings):
                sht3.write(row_start + j + 1, 0, w)

    # ── Sheet 4: 描述性统计 ──
    if stats_df is not None:
        stats_df.to_excel(writer, sheet_name="描述性统计")


# ──────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────

def _format_sheet_headers(worksheet, workbook, fmt_header):
    """为 Sheet 首行添加样式（已通过 to_excel 写入，只调整列宽）"""
    worksheet.set_column("A:A", 25)
    worksheet.set_column("B:Z", 14)
