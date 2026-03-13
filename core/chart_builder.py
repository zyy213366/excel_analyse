"""
ECharts JSON 生成器
返回可直接传给前端 echarts.setOption() 的 dict
支持：柱状图、饼图、折线图、面积图、组合图（折+柱）
"""
from typing import Optional
import pandas as pd


# ──────────────────────────────────────────────────────────
# 颜色主题
# ──────────────────────────────────────────────────────────

_PALETTE = [
    "#5470C6", "#91CC75", "#FAC858", "#EE6666",
    "#73C0DE", "#3BA272", "#FC8452", "#9A60B4",
]


def _base_option(title: str) -> dict:
    return {
        "title": {"text": title, "left": "center", "textStyle": {"fontSize": 15}},
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0},
        "grid": {"left": "10%", "right": "8%", "bottom": "15%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "color": _PALETTE,
    }


# ──────────────────────────────────────────────────────────
# 柱状图
# ──────────────────────────────────────────────────────────

def build_bar(df: pd.DataFrame,
              x_col: str,
              y_cols: list[str],
              title: str = "柱状图",
              horizontal: bool = False) -> dict:
    """
    柱状图（支持分组多系列）。
    x_col: X 轴类别列名
    y_cols: 一列或多列数值（多列=分组柱）
    horizontal=True: 条形图
    """
    categories = df[x_col].astype(str).tolist()
    series = []
    for col in y_cols:
        series.append({
            "name": col,
            "type": "bar",
            "data": df[col].round(4).tolist(),
            "label": {"show": len(df) <= 15, "position": "top"},
        })

    opt = _base_option(title)
    if horizontal:
        opt["xAxis"] = {"type": "value"}
        opt["yAxis"] = {"type": "category", "data": categories}
        for s in series:
            s["label"]["position"] = "right"
    else:
        opt["xAxis"] = {"type": "category", "data": categories,
                        "axisLabel": {"rotate": 30 if len(categories) > 8 else 0}}
        opt["yAxis"] = {"type": "value"}
    opt["series"] = series
    return opt


# ──────────────────────────────────────────────────────────
# 饼图
# ──────────────────────────────────────────────────────────

def build_pie(df: pd.DataFrame,
              name_col: str,
              value_col: str,
              title: str = "饼图") -> dict:
    """饼图（玫瑰图模式可选）"""
    data = [
        {"name": str(row[name_col]), "value": round(row[value_col], 4)}
        for _, row in df.iterrows()
    ]
    opt = _base_option(title)
    opt["tooltip"]["trigger"] = "item"
    opt["series"] = [{
        "name": value_col,
        "type": "pie",
        "radius": ["30%", "65%"],
        "data": data,
        "label": {"formatter": "{b}: {d}%"},
        "emphasis": {"itemStyle": {"shadowBlur": 10}},
    }]
    return opt


# ──────────────────────────────────────────────────────────
# 折线图
# ──────────────────────────────────────────────────────────

def build_line(df: pd.DataFrame,
               x_col: str,
               y_cols: list[str],
               title: str = "折线图",
               smooth: bool = True) -> dict:
    """折线图（支持多系列）"""
    categories = df[x_col].astype(str).tolist()
    series = []
    for col in y_cols:
        series.append({
            "name": col,
            "type": "line",
            "data": df[col].round(4).tolist(),
            "smooth": smooth,
            "symbol": "circle",
            "symbolSize": 5,
        })

    opt = _base_option(title)
    opt["xAxis"] = {"type": "category", "data": categories,
                    "boundaryGap": False,
                    "axisLabel": {"rotate": 30 if len(categories) > 10 else 0}}
    opt["yAxis"] = {"type": "value"}
    opt["series"] = series
    return opt


# ──────────────────────────────────────────────────────────
# 面积图
# ──────────────────────────────────────────────────────────

def build_area(df: pd.DataFrame,
               x_col: str,
               y_cols: list[str],
               title: str = "面积图",
               stack: bool = False) -> dict:
    """面积图（支持堆叠）"""
    opt = build_line(df, x_col, y_cols, title, smooth=True)
    for s in opt["series"]:
        s["areaStyle"] = {"opacity": 0.3}
        if stack:
            s["stack"] = "total"
    return opt


# ──────────────────────────────────────────────────────────
# 组合图（折线 + 柱状）
# ──────────────────────────────────────────────────────────

def build_combo(df: pd.DataFrame,
                x_col: str,
                bar_cols: list[str],
                line_cols: list[str],
                title: str = "组合图") -> dict:
    """折柱组合图，bar_cols 为柱，line_cols 为折线（使用右轴）"""
    categories = df[x_col].astype(str).tolist()
    series = []
    for col in bar_cols:
        series.append({
            "name": col,
            "type": "bar",
            "yAxisIndex": 0,
            "data": df[col].round(4).tolist(),
        })
    for col in line_cols:
        series.append({
            "name": col,
            "type": "line",
            "yAxisIndex": 1,
            "data": df[col].round(4).tolist(),
            "smooth": True,
            "symbol": "circle",
            "symbolSize": 5,
        })

    opt = _base_option(title)
    opt["xAxis"] = {"type": "category", "data": categories}
    opt["yAxis"] = [{"type": "value", "name": "柱"},
                    {"type": "value", "name": "折", "splitLine": {"show": False}}]
    opt["series"] = series
    return opt


# ──────────────────────────────────────────────────────────
# 散点图
# ──────────────────────────────────────────────────────────

def build_scatter(df: pd.DataFrame,
                  x_col: str,
                  y_col: str,
                  title: str = "散点图") -> dict:
    """散点图"""
    data = df[[x_col, y_col]].dropna().round(4).values.tolist()
    opt = _base_option(title)
    opt["tooltip"]["formatter"] = f"{{b}}<br/>{x_col}: {{c[0]}}<br/>{y_col}: {{c[1]}}"
    opt["xAxis"] = {"type": "value", "name": x_col}
    opt["yAxis"] = {"type": "value", "name": y_col}
    opt["series"] = [{
        "name": f"{x_col} vs {y_col}",
        "type": "scatter",
        "data": data,
        "symbolSize": 6,
    }]
    return opt


# ──────────────────────────────────────────────────────────
# 统一入口：根据类型分派
# ──────────────────────────────────────────────────────────

def build_chart(chart_type: str,
                df: pd.DataFrame,
                x_col: str,
                y_cols: list[str],
                title: str = "",
                extra: Optional[dict] = None) -> dict:
    """
    统一图表生成入口。
    chart_type: bar / pie / line / area / combo / scatter
    y_cols:
      - pie: [name_col, value_col]（第0个=类别，第1个=数值）
      - combo: bar_cols 放前面，line_cols 通过 extra["line_cols"] 传入
      - scatter: [y_col]（x 由 x_col 提供）
    """
    extra = extra or {}
    title = title or f"{chart_type.upper()} — {', '.join(y_cols)}"

    if chart_type == "bar":
        return build_bar(df, x_col, y_cols, title,
                         horizontal=extra.get("horizontal", False))
    elif chart_type == "pie":
        # 饼图语义：x_col = 类别/标签列，y_cols[0] = 数值列
        # 若 y_cols 为空则报明确错误，避免 IndexError
        if not y_cols:
            raise ValueError("饼图需要在 y 中指定数值列，例如 y=['count'] 或 y=['Y_sum']")
        name_col = x_col          # 类别来自 x
        val_col  = y_cols[0]      # 数值来自 y[0]
        return build_pie(df, name_col, val_col, title)
    elif chart_type == "line":
        return build_line(df, x_col, y_cols, title,
                          smooth=extra.get("smooth", True))
    elif chart_type == "area":
        return build_area(df, x_col, y_cols, title,
                          stack=extra.get("stack", False))
    elif chart_type == "combo":
        line_cols = extra.get("line_cols", [])
        return build_combo(df, x_col, y_cols, line_cols, title)
    elif chart_type == "scatter":
        return build_scatter(df, x_col, y_cols[0], title)
    else:
        raise ValueError(f"未知图表类型：{chart_type}，"
                         f"支持：bar/pie/line/area/combo/scatter")
