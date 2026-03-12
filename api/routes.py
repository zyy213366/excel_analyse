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
from core.analysis_engine import analyze_y_vs_all, analyze_two_column, analyze_multi_x_vs_y, analyze_model_comparison
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
    manual_mode: str = "y_vs_all"  # y_vs_all/two_column/multi_x_vs_y/time_series/pca/anova/logistic/cluster/neural_reg/ridge_lasso
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
    elif analysis.mode == "pca" and analysis.pca_loadings_df is not None:
        for i, (pc, ratio) in enumerate(zip(
            [f"PC{j+1}" for j in range(analysis.pca_n_components)],
            analysis.pca_explained_ratio
        )):
            rows.append({"主成分": pc, "方差贡献率": f"{ratio:.1%}", "累计贡献率": f"{analysis.pca_cumulative_ratio[i]:.1%}"})
    elif analysis.mode == "anova" and analysis.anova_group_stats_df is not None:
        for _, r in analysis.anova_group_stats_df.iterrows():
            rows.append({"组别": str(r["组别"]), "样本量": r["样本量"], "均值": r["均值"]})
    elif analysis.mode == "logistic" and analysis.logistic_coef_df is not None:
        for _, r in analysis.logistic_coef_df.iterrows():
            rows.append({"特征": r["Feature"], "系数": f"{r['Coefficient']:+.4f}", "Odds Ratio": f"{r['OddsRatio']:.4f}"})
    elif analysis.mode == "cluster" and analysis.cluster_stats_df is not None:
        for _, r in analysis.cluster_stats_df.iterrows():
            rows.append(dict(r))
    elif analysis.mode in ("neural_reg", "ridge_lasso", "time_series"):
        pass  # summary_text 已包含核心数据
    elif analysis.mode == "model_comparison" and analysis.mc_comparison_df is not None:
        for _, r in analysis.mc_comparison_df.iterrows():
            rows.append({
                "模型": r["模型"],
                "CV均值R²": f"{r['CV均值R²']:.4f}",
                "RMSE": f"{r['RMSE']:.4f}",
                "MAE": f"{r['MAE']:.4f}",
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

            _ts_kw = ["时间趋势", "时序", "走势", "预测未来", "季节", "arima", "trend"]
            if mode == "y_vs_all" and any(k in inst_lower for k in _ts_kw):
                mode = "time_series"

            _pca_kw = ["pca", "降维", "主成分", "factor"]
            if mode == "y_vs_all" and any(k in inst_lower for k in _pca_kw):
                mode = "pca"
                x_cols = []  # 自动取所有数值列

            _anova_kw = ["方差分析", "anova", "组间", "各组差异", "组别比较"]
            if mode == "y_vs_all" and any(k in inst_lower for k in _anova_kw):
                mode = "anova"

            _cluster_kw = ["聚类", "k-means", "kmeans", "分群", "分组发现"]
            if mode == "y_vs_all" and any(k in inst_lower for k in _cluster_kw):
                mode = "cluster"
                x_cols = []

            _neural_kw = ["神经网络", "mlp", "深度学习", "neural", "非线性回归"]
            if mode == "y_vs_all" and any(k in inst_lower for k in _neural_kw):
                mode = "neural_reg"

            _ridge_kw = ["岭回归", "套索", "lasso", "ridge", "正则化"]
            if mode == "y_vs_all" and any(k in inst_lower for k in _ridge_kw):
                mode = "ridge_lasso"

            _mc_kw = ["模型对比", "model comparison", "automl", "多模型", "哪个模型最好", "最优模型", "对比模型"]
            if any(k in inst_lower for k in _mc_kw):
                mode = "model_comparison"
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
        elif mode == "multi_x_vs_y":
            valid_x = [c for c in x_cols if c in clean_df.columns]
            if not valid_x:
                return JSONResponse({"success": False, "error": "所有自变量列在清洗后均不存在"})
            analysis = analyze_multi_x_vs_y(clean_df, target_y, valid_x)
        elif mode == "time_series":
            time_c = x_cols[0] if x_cols else None
            if not time_c:
                return JSONResponse({"success": False, "error": "时序分析需要指定时间列（x_columns）"})
            from core.analysis_engine import analyze_time_series
            analysis = analyze_time_series(clean_df, time_c, target_y)
        elif mode == "pca":
            cols_for_pca = x_cols if x_cols else [c for c in numeric_cols if c != target_y]
            if len(cols_for_pca) < 2:
                return JSONResponse({"success": False, "error": "PCA 至少需要 2 个变量"})
            from core.analysis_engine import analyze_pca
            analysis = analyze_pca(clean_df, cols_for_pca)
        elif mode == "anova":
            group_c = x_cols[0] if x_cols else None
            if not group_c:
                return JSONResponse({"success": False, "error": "ANOVA 需要指定分组列（x_columns）"})
            from core.analysis_engine import analyze_anova
            analysis = analyze_anova(clean_df, target_y, group_c)
        elif mode == "logistic":
            valid_x = [c for c in x_cols if c in clean_df.columns]
            if not valid_x:
                valid_x = [c for c in numeric_cols if c != target_y]
            from core.analysis_engine import analyze_logistic
            analysis = analyze_logistic(clean_df, target_y, valid_x)
        elif mode == "cluster":
            cols_for_cluster = x_cols if x_cols else [c for c in numeric_cols if c != target_y]
            if len(cols_for_cluster) < 2:
                return JSONResponse({"success": False, "error": "聚类至少需要 2 个变量"})
            from core.analysis_engine import analyze_cluster
            analysis = analyze_cluster(clean_df, cols_for_cluster)
        elif mode == "neural_reg":
            valid_x = [c for c in x_cols if c in clean_df.columns]
            if not valid_x:
                valid_x = [c for c in numeric_cols if c != target_y]
            from core.analysis_engine import analyze_neural_reg
            analysis = analyze_neural_reg(clean_df, target_y, valid_x)
        elif mode == "ridge_lasso":
            valid_x = [c for c in x_cols if c in clean_df.columns]
            if not valid_x:
                valid_x = [c for c in numeric_cols if c != target_y]
            from core.analysis_engine import analyze_ridge_lasso
            analysis = analyze_ridge_lasso(clean_df, target_y, valid_x)
        elif mode == "model_comparison":
            valid_x = [c for c in x_cols if c in clean_df.columns]
            if not valid_x:
                valid_x = [c for c in numeric_cols if c != target_y]
            if len(valid_x) < 1:
                return JSONResponse({"success": False, "error": "模型对比至少需要 1 个自变量"})
            analysis = analyze_model_comparison(clean_df, target_y, valid_x)
        else:
            return JSONResponse({"success": False, "error": f"未知分析模式：{mode}"})
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
