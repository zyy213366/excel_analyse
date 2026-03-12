"""
FastAPI 路由 — 页面路由 + REST API
"""
import uuid
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_excel, get_numeric_columns, preprocess_for_analysis
from utils.file_manager import get_output_path, cleanup_old_reports
from core.analysis_engine import analyze_y_vs_all, analyze_two_column, analyze_multi_x_vs_y
from core.report_builder import build_report
from core.result_formatter import format_result
from config import OUTPUTS_DIR

router = APIRouter()
templates = Jinja2Templates(directory="templates")

from db import init_db, save_upload, get_upload, load_all_uploads, save_analysis

UPLOADS_DIR = OUTPUTS_DIR.parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# 启动时从 SQLite 重建（重启不丢失已上传文件的路径映射）
init_db()
_uploaded_files: dict[str, Path] = load_all_uploads()


# ──────────────────────────────────────────────────────────
# ModelScope / Gradio 兼容接口
# ModelScope 平台用 modelscope-gradio JS 加载所有 Docker 应用，
# 会轮询 /config、/info、/queue/status，404 会导致永久卡在加载中
# ──────────────────────────────────────────────────────────

@router.get("/config")
async def ms_config():
    return {"status": "NORMAL", "version": "2.0", "mode": "docker",
            "dev_mode": False, "analytics_enabled": False,
            "components": [], "title": "Excel 智能分析助手",
            "is_space": True, "enable_queue": False}


@router.get("/info")
async def ms_info():
    return {"named_endpoints": {}, "unnamed_endpoints": {}}


@router.get("/queue/status")
async def ms_queue_status():
    return {"queue_size": 0, "status": "COMPLETE", "success": True}


# ──────────────────────────────────────────────────────────
# 页面路由
# ──────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {
        "request": request,
        "active_page": "home",
    })


@router.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request, instruction: str = ""):
    return templates.TemplateResponse("analyze.html", {
        "request": request,
        "active_page": "analyze",
        "prefill_instruction": instruction,
    })


@router.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    return templates.TemplateResponse("history.html", {
        "request": request,
        "active_page": "history",
    })


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "active_page": "settings",
    })


# ──────────────────────────────────────────────────────────
# API：文件上传
# ──────────────────────────────────────────────────────────

@router.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """接收 Excel 文件，返回 file_id 和列名列表"""
    if not (file.filename or "").lower().endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(400, "仅支持 .xlsx、.xls 或 .csv 格式")

    file_id = str(uuid.uuid4())
    save_path = UPLOADS_DIR / f"{file_id}_{file.filename}"

    content = await file.read()
    save_path.write_bytes(content)
    _uploaded_files[file_id] = save_path

    try:
        df, all_cols = load_excel(str(save_path))
        numeric_cols = get_numeric_columns(df)
        row_count = len(df)
        save_upload(file_id, file.filename, str(save_path), row_count, numeric_cols)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, f"文件读取失败：{str(e)}")

    return {
        "file_id": file_id,
        "filename": file.filename,
        "row_count": row_count,
        "all_columns": all_cols,
        "numeric_columns": numeric_cols,
    }


# ──────────────────────────────────────────────────────────
# API：执行分析
# ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    file_id: str
    instruction: str = ""
    use_ai: bool = True
    manual_mode: str = "y_vs_all"
    manual_y: Optional[str] = None
    manual_x_cols: list[str] = []


def _build_table_data(analysis) -> list[dict]:
    """从 AnalysisResult 提取界面展示用的关键统计列表"""
    rows = []
    if analysis.mode == "y_vs_all" and analysis.feature_importance_df is not None:
        df = analysis.feature_importance_df[["Feature", "Importance", "Pearson"]].head(10)
        for _, r in df.iterrows():
            rows.append({
                "特征": r["Feature"],
                "重要性": f"{r['Importance']:.1%}",
                "Pearson": f"{r['Pearson']:+.3f}",
            })
    elif analysis.mode == "two_column":
        rows = [
            {
                "指标": "Pearson r",
                "值": f"{analysis.pearson_r:.4f}",
                "显著性": "p<0.001" if analysis.pearson_p < 0.001 else f"p={analysis.pearson_p:.3f}",
            },
            {
                "指标": "Spearman ρ",
                "值": f"{analysis.spearman_r:.4f}",
                "显著性": "p<0.001" if analysis.spearman_p < 0.001 else f"p={analysis.spearman_p:.3f}",
            },
        ]
    elif analysis.mode == "multi_x_vs_y" and analysis.multi_reg_coef_df is not None:
        for _, r in analysis.multi_reg_coef_df.iterrows():
            rows.append({
                "特征": r["Feature"],
                "系数": f"{r['Coefficient']:.4f}",
                "P值": f"{r['p_value']:.4f}" if r['p_value'] == r['p_value'] else "N/A",
                "显著性": r.get("Significant", ""),
            })
    return rows


@router.post("/api/analyze")
async def api_analyze(req: AnalyzeRequest):
    """执行分析，返回 JSON 结果"""
    if req.file_id not in _uploaded_files:
        raise HTTPException(400, "文件不存在，请重新上传")

    file_path = _uploaded_files[req.file_id]
    try:
        df_raw, _ = load_excel(str(file_path))
        numeric_cols = get_numeric_columns(df_raw)
    except Exception as e:
        raise HTTPException(400, f"文件读取失败：{str(e)}")

    # 确定分析参数
    intent_info = None
    mode = req.manual_mode
    target_y = req.manual_y
    x_cols = [c for c in req.manual_x_cols if c != req.manual_y]

    if req.use_ai and req.instruction.strip():
        try:
            from core.nlp_parser import IntentParser
            parser = IntentParser()
            result = parser.parse(req.instruction, numeric_cols)
            if result.get("error"):
                return JSONResponse({"success": False, "error": result["error"]})
            mode = result["analysis_mode"]
            target_y = result["target_y"]
            x_cols = result["x_columns"]
            # 关键词覆盖：若 NLP 返回 y_vs_all 但指令明显是多因素回归，强制切换
            inst_lower = req.instruction.lower()
            _multi_kw = ["系数", "回归系数", "p值", "p-value", "多元", "多因素", "共线性", "多重共线", "特征重要性"]
            if mode == "y_vs_all" and any(k in inst_lower for k in _multi_kw):
                mode = "multi_x_vs_y"
                # x_cols 留空，后续自动填充除 Y 外所有列
                x_cols = []

            intent_info = {
                "mode": mode,
                "target_y": target_y,
                "x_cols": x_cols,
                "hint": result.get("analysis_hint", ""),
                "confidence": result.get("confidence", 0.5),
            }
        except Exception as e:
            return JSONResponse({"success": False, "error": f"AI 解析失败：{str(e)}"})

    if not target_y or target_y not in numeric_cols:
        # target_y 未指定时，对 multi_x_vs_y 默认取第一个数值列
        if mode == "multi_x_vs_y" and numeric_cols:
            target_y = numeric_cols[0]
        else:
            return JSONResponse({"success": False, "error": f"目标变量 `{target_y}` 不存在，可用列：{numeric_cols[:5]}"})

    # multi_x_vs_y 模式：若 X 列为空，自动使用除 Y 外所有数值列
    if mode == "multi_x_vs_y" and not x_cols:
        x_cols = [c for c in numeric_cols if c != target_y]

    # 数据预处理
    try:
        if mode == "y_vs_all":
            cols_needed = numeric_cols
        elif mode == "two_column":
            if not x_cols:
                return JSONResponse({"success": False, "error": "two_column 模式需要指定第二列"})
            cols_needed = [target_y, x_cols[0]]
        else:
            cols_needed = [target_y] + x_cols

        clean_df, raw_count, valid_count = preprocess_for_analysis(df_raw, cols_needed)
        if valid_count < 10:
            return JSONResponse({"success": False, "error": f"有效数据不足（{valid_count} 行），无法进行可靠分析"})
    except Exception as e:
        return JSONResponse({"success": False, "error": f"数据预处理失败：{str(e)}"})

    # 执行分析
    try:
        if mode == "y_vs_all":
            analysis = analyze_y_vs_all(clean_df, target_y)
        elif mode == "two_column":
            analysis = analyze_two_column(clean_df, x_cols[0], target_y)
        else:
            valid_x = [c for c in x_cols if c in clean_df.columns]
            if not valid_x:
                return JSONResponse({"success": False, "error": "所有自变量列在清洗后均不存在"})
            analysis = analyze_multi_x_vs_y(clean_df, target_y, valid_x)
        analysis.raw_row_count = raw_count
        analysis.valid_row_count = valid_count
    except Exception as e:
        import traceback
        return JSONResponse({"success": False, "error": f"分析失败：{str(e)}\n{traceback.format_exc()}"})

    # 生成报告
    report_filename = None
    try:
        out_path = get_output_path(file_path.name, "分析报告")
        build_report(analysis, out_path)
        cleanup_old_reports()
        report_filename = out_path.name
    except Exception:
        pass  # 报告生成失败不影响结果返回

    # 记录分析历史到 SQLite
    try:
        _orig_name = file_path.name.split("_", 1)[-1] if "_" in file_path.name else file_path.name
        save_analysis(req.file_id, _orig_name, req.instruction, mode, report_filename or "")
    except Exception:
        pass

    # 根据用户指令意图，选择专项格式化器输出差异化结果
    # 优先用 instruction 文字（不论 AI 模式是否开启），无指令时用 manual_mode 作为后备
    format_instruction = req.instruction.strip() if req.instruction.strip() else req.manual_mode
    formatted_text = format_result(analysis, format_instruction)

    return {
        "success": True,
        "summary_text": formatted_text,
        "table_data": _build_table_data(analysis),
        "report_filename": report_filename,
        "intent": intent_info,
        "data_info": {"raw": raw_count, "valid": valid_count},
    }


# ──────────────────────────────────────────────────────────
# API：下载报告
# ──────────────────────────────────────────────────────────

@router.get("/api/download/{filename}")
async def api_download(filename: str):
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "文件不存在")
    return FileResponse(
        str(file_path),
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ──────────────────────────────────────────────────────────
# API：历史记录
# ──────────────────────────────────────────────────────────

@router.get("/api/history")
async def api_history():
    from db import get_analyses
    rows = get_analyses(limit=50)
    result = []
    for r in rows:
        fname = r["report_filename"]
        fpath = OUTPUTS_DIR / fname if fname else None
        result.append({
            "filename":     fname,
            "display_name": r["original_name"],
            "instruction":  r["instruction"],
            "mode":         r["mode"],
            "size_kb":      round(fpath.stat().st_size / 1024, 1) if fpath and fpath.exists() else 0,
            "modified":     r["created_at"],
        })
    return result


# ──────────────────────────────────────────────────────────
# API：设置（读写 .env）
# ──────────────────────────────────────────────────────────

from dotenv import dotenv_values, set_key

ENV_PATH = Path(__file__).parent.parent / ".env"


@router.get("/api/settings")
async def api_get_settings():
    vals = dotenv_values(str(ENV_PATH))
    return {
        "api_key": vals.get("DEEPSEEK_API_KEY", ""),
        "base_url": vals.get("DEEPSEEK_BASE_URL", ""),
        "model": vals.get("DEEPSEEK_MODEL", ""),
    }


class SettingsRequest(BaseModel):
    api_key: str
    base_url: str
    model: str


@router.post("/api/settings")
async def api_save_settings(req: SettingsRequest):
    set_key(str(ENV_PATH), "DEEPSEEK_API_KEY", req.api_key)
    set_key(str(ENV_PATH), "DEEPSEEK_BASE_URL", req.base_url)
    set_key(str(ENV_PATH), "DEEPSEEK_MODEL", req.model)
    return {"success": True}
