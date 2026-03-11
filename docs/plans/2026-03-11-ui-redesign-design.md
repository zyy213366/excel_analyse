# UI 重设计方案 — 现代浅色商务风

**日期**：2026-03-11
**状态**：已批准

## 目标

将当前几乎无样式的 Gradio 界面升级为现代浅色商务风，提升视觉专业度和使用体验。

## 决策

- **风格**：现代浅色商务风（白底 + 蓝灰配色）
- **布局**：保持左右分栏（左操作 / 右结果）
- **结果展示**：Markdown 结论 + 新增 `gr.Dataframe` 关键统计表

## 技术方案：Gradio Theme + 精准 CSS

### 主题基础

使用 `gr.themes.Soft()` 作为底层主题，在此基础上叠加自定义 CSS。

### 视觉改动

| 元素 | 当前 | 改后 |
|------|------|------|
| 整体主题 | 默认 | `gr.themes.Soft()` |
| Header | 渐变紫色背景 | 白底 + 左侧蓝色竖线 + 深色标题 |
| 上传区 | 默认控件 | 虚线蓝色边框卡片 |
| 示例按钮 | 默认小按钮 | 浅灰 chip 样式 |
| 主按钮 | 默认 primary | 蓝色实心，加宽 |
| 结果区 | 单一 Markdown | Markdown + Dataframe 表格 |
| 分割线 | `---` 硬线 | 卡片/section 间距替代 |

### 数据层改动

`on_analyze` 新增返回值：`pd.DataFrame`，包含关键统计列：

- `y_vs_all` 模式：列名、相关系数、P值、显著性
- `two_column` 模式：指标名、数值
- `multi_x_vs_y` 模式：列名、系数、贡献度

新增 `gr.Dataframe` 组件接收该返回值。

## 文件改动范围

- `app.py`：主题、CSS、布局、新增 Dataframe 组件、`on_analyze` 返回值
- `core/analysis_engine.py`：确认各分析结果对象包含可转 DataFrame 的数据
