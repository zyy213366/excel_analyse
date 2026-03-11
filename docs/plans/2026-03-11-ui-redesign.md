# UI 重设计实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 Gradio 界面从几乎无样式升级为现代浅色商务风，新增关键统计数据表格展示。

**Architecture:** 仅修改 `app.py`——替换主题为 `gr.themes.Soft()`，重写 CSS，新增 `_build_table_df()` 辅助函数从现有 `AnalysisResult` 提取展示用 DataFrame，`on_analyze` 新增第三个返回值，UI 新增 `gr.Dataframe` 组件。`analysis_engine.py` 无需改动。

**Tech Stack:** Python 3.10, Gradio, pandas

---

## 改动总览

- **修改**：`app.py`（主题、CSS、布局、回调返回值）
- **不改动**：`core/analysis_engine.py`、`core/nlp_parser.py`、`utils/`

---

## Task 1：替换主题 + 重写 CSS

**Files:**
- Modify: `app.py`

**Step 1：替换 `build_app()` 中的 `gr.Blocks` 调用，加入主题**

将：
```python
with gr.Blocks(css=CSS, title="Excel 智能分析助手") as demo:
```
改为：
```python
with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Excel 智能分析助手") as demo:
```

**Step 2：将文件顶部的 `CSS` 常量替换为以下内容**

```python
CSS = """
/* ── 整体容器 ── */
.gradio-container {
    max-width: 1200px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

/* ── Header 卡片：白底 + 左侧蓝色竖线 ── */
.title-card {
    background: #ffffff;
    border-left: 5px solid #3b82f6;
    border-radius: 8px;
    padding: 18px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.title-card h1 {
    margin: 0 0 4px 0;
    font-size: 1.5rem;
    font-weight: 700;
    color: #1e293b;
}
.title-card p {
    margin: 0;
    color: #64748b;
    font-size: 0.9rem;
}

/* ── Section 标题 ── */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin: 16px 0 6px 0;
}

/* ── 上传区：蓝色虚线卡片 ── */
.upload-card .wrap {
    border: 2px dashed #93c5fd !important;
    border-radius: 10px !important;
    background: #f0f9ff !important;
    transition: border-color 0.2s, background 0.2s;
}
.upload-card .wrap:hover {
    border-color: #3b82f6 !important;
    background: #e0f2fe !important;
}

/* ── 示例指令 Chip 按钮 ── */
.chip-btn button {
    background: #f1f5f9 !important;
    color: #475569 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 20px !important;
    font-size: 0.78rem !important;
    padding: 4px 12px !important;
    transition: all 0.15s;
}
.chip-btn button:hover {
    background: #e2e8f0 !important;
    color: #1e293b !important;
    border-color: #cbd5e1 !important;
}

/* ── 主分析按钮 ── */
.analyze-btn button {
    background: #3b82f6 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 12px 0 !important;
    letter-spacing: 0.02em;
    transition: background 0.2s;
}
.analyze-btn button:hover {
    background: #2563eb !important;
}

/* ── 结果卡片 ── */
.result-card {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    padding: 20px;
    min-height: 200px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ── 数据表格区 ── */
.table-card {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    margin-top: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    overflow: hidden;
}

/* ── 状态栏 ── */
.status-bar p {
    font-size: 0.82rem;
    color: #64748b;
    margin: 4px 0 0 0;
}

/* ── 下载按钮区 ── */
.download-area {
    margin-top: 12px;
}
"""
```

**Step 3：手动启动应用，确认无语法错误、主题生效（页面背景变白，元素更圆润）**

```bash
cd C:/Users/29571/PycharmProjects/excel-analyzer
python app.py
```

访问 `http://127.0.0.1:7860`，预期看到白色底色的 Soft 主题。

**Step 4：Commit**

```bash
git add app.py
git commit -m "style: apply Soft theme and rewrite CSS for business UI"
```

---

## Task 2：重写 Header HTML + 为控件加 CSS 类

**Files:**
- Modify: `app.py` — `build_app()` 函数内的 UI 构建部分

**Step 1：替换 `gr.HTML` Header 块**

将：
```python
gr.HTML("""
<div class="title-box">
    <h1>🎀 Excel 智能分析助手</h1>
    <p>上传 Excel → 用自然语言描述分析需求 → 获取专业分析报告</p>
</div>
""")
```
改为：
```python
gr.HTML("""
<div class="title-card">
    <h1>📊 Excel 智能分析助手</h1>
    <p>上传 Excel 文件 &nbsp;→&nbsp; 用自然语言描述分析需求 &nbsp;→&nbsp; 获取专业分析报告</p>
</div>
""")
```

**Step 2：为上传组件加 `elem_classes`**

将：
```python
file_input = gr.File(
    label="📂 上传 Excel 文件",
    file_types=[".xlsx", ".xls"],
    type="filepath",
)
```
改为：
```python
file_input = gr.File(
    label="📂 上传 Excel 文件",
    file_types=[".xlsx", ".xls"],
    type="filepath",
    elem_classes=["upload-card"],
)
```

**Step 3：为示例按钮加 `elem_classes`，并加 section 标签**

找到示例按钮部分，改为：
```python
gr.HTML('<p class="section-label">快捷示例</p>')
with gr.Row():
    for ex in EXAMPLE_INSTRUCTIONS[:2]:
        gr.Button(ex, size="sm", elem_classes=["chip-btn"]).click(
            fn=lambda t=ex: t,
            outputs=instruction_input,
        )
with gr.Row():
    for ex in EXAMPLE_INSTRUCTIONS[2:]:
        gr.Button(ex, size="sm", elem_classes=["chip-btn"]).click(
            fn=lambda t=ex: t,
            outputs=instruction_input,
        )
```

**Step 4：为主按钮和状态栏加 `elem_classes`**

```python
analyze_btn = gr.Button("▶ 开始分析", variant="primary", size="lg", elem_classes=["analyze-btn"])
status_label = gr.Markdown("*状态：就绪*", elem_classes=["status-bar"])
```

**Step 5：启动验证**

访问 `http://127.0.0.1:7860`，确认：
- Header 是白底蓝左边框样式
- 上传区有蓝色虚线
- 示例按钮是圆角 chip 样式
- 主按钮是蓝色

**Step 6：Commit**

```bash
git add app.py
git commit -m "style: apply CSS classes to UI components"
```

---

## Task 3：新增 `_build_table_df()` 辅助函数 + 更新 `on_analyze` 返回值

**Files:**
- Modify: `app.py`

**Step 1：在 `on_analyze` 函数定义之前，添加辅助函数**

```python
def _build_table_df(analysis) -> pd.DataFrame | None:
    """
    从 AnalysisResult 中提取用于界面展示的关键统计 DataFrame。
    - y_vs_all:     特征重要性表（Top 10）
    - two_column:   描述性统计对比表
    - multi_x_vs_y: 多元回归系数表
    """
    mode = analysis.mode

    if mode == "y_vs_all" and analysis.feature_importance_df is not None:
        df = analysis.feature_importance_df[["Feature", "Importance", "Pearson"]].head(10).copy()
        df["Importance"] = df["Importance"].map(lambda x: f"{x:.1%}")
        df["Pearson"] = df["Pearson"].map(lambda x: f"{x:+.3f}")
        df.columns = ["特征", "重要性", "Pearson相关"]
        return df

    if mode == "two_column" and analysis.stats_df is not None:
        return analysis.stats_df.reset_index().rename(columns={"index": "列名"})

    if mode == "multi_x_vs_y" and analysis.multi_reg_coef_df is not None:
        df = analysis.multi_reg_coef_df.copy()
        df["Coefficient"] = df["Coefficient"].map(lambda x: f"{x:.4f}")
        df["p_value"] = df["p_value"].map(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
        df.columns = ["特征", "回归系数", "P值", "显著性"]
        return df

    return None
```

**Step 2：修改 `on_analyze` 的所有 `return` 语句，新增第三个返回值**

所有提前返回的错误路径（返回 `(str, None)` 的地方），全部改为 `(str, None, None)`。

最后的成功返回：
```python
# 原来：
return summary_md, str(output_path)

# 改为：
table_df = _build_table_df(analysis)
return summary_md, str(output_path), table_df
```

**Step 3：启动验证（无 UI 改动，仅确认无报错）**

```bash
python app.py
```

**Step 4：Commit**

```bash
git add app.py
git commit -m "feat: extract key stats DataFrame from AnalysisResult for table display"
```

---

## Task 4：UI 新增 `gr.Dataframe` 组件并绑定

**Files:**
- Modify: `app.py` — 右列 UI 部分 + 事件绑定部分

**Step 1：在右列中新增 Dataframe 组件**

找到右列区域：
```python
with gr.Column(scale=6):
    result_md = gr.Markdown(...)
    report_file = gr.File(...)
```

改为：
```python
with gr.Column(scale=6):
    gr.HTML('<p class="section-label">分析结论</p>')
    result_md = gr.Markdown(
        "上传文件并点击「开始分析」按钮。",
        elem_classes=["result-card"],
    )
    gr.HTML('<p class="section-label">关键统计数据</p>')
    result_table = gr.Dataframe(
        label=None,
        interactive=False,
        wrap=True,
        elem_classes=["table-card"],
    )
    with gr.Row(elem_classes=["download-area"]):
        report_file = gr.File(label="📥 下载 Excel 分析报告", visible=True)
```

**Step 2：更新事件绑定，加入 `result_table` 输出**

找到 `.then(fn=on_analyze, ...)` 那段，改为：
```python
.then(
    fn=on_analyze,
    inputs=[file_input, instruction_input, manual_mode, manual_y, manual_x, use_nlp_toggle],
    outputs=[result_md, report_file, result_table],
)
```

**Step 3：完整端到端测试**

1. 启动 `python app.py`，访问 `http://127.0.0.1:7860`
2. 上传 Excel 文件，输入自然语言指令，点击「开始分析」
3. 确认：右侧出现 Markdown 结论 + 下方数据表格 + 下载按钮

**Step 4：Commit**

```bash
git add app.py
git commit -m "feat: add Dataframe component to display key stats table in results"
```

---

## Task 5：推送到 GitHub

```bash
git push origin master
```

确认 https://github.com/zyy213366/excel_analyse 已更新。
