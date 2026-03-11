"""
Excel 智能分析助手 - Gradio Web 界面
支持自然语言指令 + 三模式分析 + Excel 报告生成
"""
import sys
import os
# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import pandas as pd

from utils.data_loader import load_excel, get_numeric_columns, preprocess_for_analysis
from utils.file_manager import get_output_path, cleanup_old_reports, check_file_writable
from core.analysis_engine import analyze_y_vs_all, analyze_two_column, analyze_multi_x_vs_y
from core.report_builder import build_report

# 懒加载 NLP 解析器（避免启动时因 API Key 缺失报错）
_intent_parser = None

def _get_parser():
    global _intent_parser
    if _intent_parser is None:
        try:
            from core.nlp_parser import IntentParser
            _intent_parser = IntentParser()
        except Exception as e:
            return None, str(e)
    return _intent_parser, None


# ──────────────────────────────────────────────────────────
# 核心回调函数
# ──────────────────────────────────────────────────────────

def on_file_upload(file_obj):
    """文件上传后，自动检测列名并更新下拉框"""
    if file_obj is None:
        return gr.update(value=""), gr.update(choices=[], value=None), gr.update(choices=[], value=[])

    try:
        df, all_cols = load_excel(file_obj)
        numeric_cols = get_numeric_columns(df)

        if not numeric_cols:
            return (
                "⚠ 未检测到可分析的数值列，请检查文件格式。",
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=[]),
            )

        cols_preview = "  ".join([f"`{c}`" for c in numeric_cols[:15]])
        if len(numeric_cols) > 15:
            cols_preview += f"  *...共 {len(numeric_cols)} 列*"

        status = f"✅ 文件加载成功！检测到 **{len(numeric_cols)}** 个可分析列：\n\n{cols_preview}"
        return (
            status,
            gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None),
            gr.update(choices=numeric_cols, value=[]),
        )
    except Exception as e:
        return f"❌ 文件读取失败：{str(e)}", gr.update(choices=[]), gr.update(choices=[])


def on_analyze(file_obj, instruction, manual_mode, manual_y, manual_x_cols, use_nlp):
    """
    主分析回调：
    - use_nlp=True: 调用 DeepSeek 解析意图
    - use_nlp=False: 使用手动选择的参数
    """
    if file_obj is None:
        return "❌ 请先上传 Excel 文件！", None

    try:
        df_raw, all_cols = load_excel(file_obj)
        numeric_cols = get_numeric_columns(df_raw)
    except Exception as e:
        return f"❌ 文件读取失败：{str(e)}", None

    # ── 确定分析参数 ──
    intent_info = ""
    if use_nlp and instruction.strip():
        parser, err = _get_parser()
        if err:
            return f"❌ AI 解析器初始化失败：{err}\n\n请检查 .env 中的 API Key 配置。", None

        result = parser.parse(instruction, numeric_cols)

        if result.get("error"):
            return (
                f"❌ AI 解析失败：{result['error']}\n\n请切换到手动模式或检查 API 配置。",
                None,
            )

        mode = result["analysis_mode"]
        target_y = result["target_y"]
        x_cols = result["x_columns"]
        confidence = result.get("confidence", 0.5)
        hint = result.get("analysis_hint", "")

        conf_icon = "✅" if confidence >= 0.7 else ("⚠" if confidence >= 0.5 else "❌")
        intent_info = (
            f"### 🤖 AI 意图解析结果\n"
            f"- **理解**：{hint}\n"
            f"- **模式**：`{mode}`\n"
            f"- **目标变量**：`{target_y}`\n"
            f"- **自变量**：{', '.join([f'`{c}`' for c in x_cols]) if x_cols else '（全部）'}\n"
            f"- **置信度**：{conf_icon} {confidence:.0%}\n\n"
        )

        if confidence < 0.5:
            intent_info += "⚠ **置信度较低**，建议切换到手动模式确认参数。\n\n"

    else:
        # 手动模式
        mode = manual_mode
        target_y = manual_y
        x_cols = [c for c in manual_x_cols if c != target_y] if manual_x_cols else []
        intent_info = f"### 🔧 手动模式\n- **模式**：`{mode}`，**目标**：`{target_y}`\n\n"

    # ── 参数校验 ──
    if not target_y or target_y not in numeric_cols:
        return intent_info + f"❌ 目标变量 `{target_y}` 不存在或非数值列。可用列：{', '.join(numeric_cols[:5])}", None

    # ── 数据预处理 ──
    try:
        if mode == "y_vs_all":
            cols_needed = numeric_cols
        elif mode == "two_column":
            col_b = x_cols[0] if x_cols else None
            if not col_b:
                return intent_info + "❌ two_column 模式需要指定第二列。", None
            cols_needed = [target_y, col_b]
        else:  # multi_x_vs_y
            if not x_cols:
                return intent_info + "❌ multi_x_vs_y 模式需要指定自变量列。", None
            cols_needed = [target_y] + x_cols

        clean_df, raw_count, valid_count = preprocess_for_analysis(df_raw, cols_needed)

        if valid_count < 10:
            return intent_info + f"❌ 有效数据不足（{valid_count} 行），无法进行可靠分析。原始数据 {raw_count} 行。", None

        data_info = f"> 📊 数据：原始 **{raw_count}** 行 → 有效 **{valid_count}** 行\n\n"

    except Exception as e:
        return intent_info + f"❌ 数据预处理失败：{str(e)}", None

    # ── 执行分析 ──
    try:
        if mode == "y_vs_all":
            analysis = analyze_y_vs_all(clean_df, target_y)
        elif mode == "two_column":
            col_b = x_cols[0]
            if col_b not in clean_df.columns:
                return intent_info + f"❌ 列 `{col_b}` 在清洗后数据中不存在。", None
            analysis = analyze_two_column(clean_df, col_b, target_y)
        else:
            valid_x = [c for c in x_cols if c in clean_df.columns]
            if not valid_x:
                return intent_info + "❌ 所有自变量列在清洗后均不存在。", None
            analysis = analyze_multi_x_vs_y(clean_df, target_y, valid_x)

        analysis.raw_row_count = raw_count
        analysis.valid_row_count = valid_count

    except Exception as e:
        import traceback
        return intent_info + f"❌ 分析计算失败：{str(e)}\n```\n{traceback.format_exc()}\n```", None

    # ── 生成报告 ──
    try:
        original_name = os.path.basename(file_obj) if isinstance(file_obj, str) else "上传文件"
        output_path = get_output_path(original_name, "分析报告")
        build_report(analysis, output_path)
        cleanup_old_reports()
    except Exception as e:
        return intent_info + data_info + analysis.summary_text + f"\n\n⚠ 报告生成失败：{str(e)}", None

    summary_md = intent_info + data_info + analysis.summary_text
    return summary_md, str(output_path)


# ──────────────────────────────────────────────────────────
# Gradio 界面
# ──────────────────────────────────────────────────────────

EXAMPLE_INSTRUCTIONS = [
    "分析所有因素对目标变量的影响",
    "找出对销售额影响最大的因素",
    "分析温度和湿度之间的关系",
    "分析温度、湿度、时间对销量的综合影响",
]

CSS = """
.gradio-container { max-width: 1200px !important; }
.title-box { text-align: center; padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
             border-radius: 10px; color: white !important; margin-bottom: 15px; }
"""

def build_app():
    with gr.Blocks(css=CSS, title="Excel 智能分析助手") as demo:
        gr.HTML("""
        <div class="title-box">
            <h1>🎀 Excel 智能分析助手</h1>
            <p>上传 Excel → 用自然语言描述分析需求 → 获取专业分析报告</p>
        </div>
        """)

        with gr.Row():
            # ── 左列（40%）──
            with gr.Column(scale=4):
                file_input = gr.File(
                    label="📂 上传 Excel 文件",
                    file_types=[".xlsx", ".xls"],
                    type="filepath",
                )
                file_status = gr.Markdown("*等待文件上传...*")

                gr.Markdown("---")
                gr.Markdown("### 💬 分析指令")

                use_nlp_toggle = gr.Checkbox(
                    label="使用 AI 自然语言解析（需要 API Key）",
                    value=True,
                )
                instruction_input = gr.Textbox(
                    label="自然语言指令",
                    placeholder="例：帮我分析哪些因素对销售额影响最大",
                    lines=2,
                )
                with gr.Row():
                    for ex in EXAMPLE_INSTRUCTIONS[:2]:
                        gr.Button(ex, size="sm").click(
                            fn=lambda t=ex: t,
                            outputs=instruction_input,
                        )
                with gr.Row():
                    for ex in EXAMPLE_INSTRUCTIONS[2:]:
                        gr.Button(ex, size="sm").click(
                            fn=lambda t=ex: t,
                            outputs=instruction_input,
                        )

                gr.Markdown("---")
                with gr.Accordion("🔧 手动模式（AI 解析失败时使用）", open=False):
                    manual_mode = gr.Dropdown(
                        label="分析模式",
                        choices=["y_vs_all", "two_column", "multi_x_vs_y"],
                        value="y_vs_all",
                    )
                    manual_y = gr.Dropdown(label="目标变量 (Y)", choices=[], interactive=True)
                    manual_x = gr.CheckboxGroup(label="自变量 (X)（multi_x_vs_y 模式使用）", choices=[])

                analyze_btn = gr.Button("▶ 开始分析", variant="primary", size="lg")
                status_label = gr.Markdown("*状态：就绪*")

            # ── 右列（60%）──
            with gr.Column(scale=6):
                result_md = gr.Markdown(
                    "# 分析结果将在这里显示\n\n上传文件并点击「开始分析」按钮。",
                    label="分析结论",
                )
                report_file = gr.File(label="📥 下载 Excel 分析报告", visible=True)

        # ── 事件绑定 ──
        file_input.change(
            fn=on_file_upload,
            inputs=[file_input],
            outputs=[file_status, manual_y, manual_x],
        )

        analyze_btn.click(
            fn=lambda: gr.update(value="*状态：分析中...*"),
            outputs=[status_label],
        ).then(
            fn=on_analyze,
            inputs=[file_input, instruction_input, manual_mode, manual_y, manual_x, use_nlp_toggle],
            outputs=[result_md, report_file],
        ).then(
            fn=lambda: gr.update(value="*状态：✅ 完成*"),
            outputs=[status_label],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
