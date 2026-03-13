"""
FastAPI 路由 — 页面路由 + REST API
"""
import re
import uuid
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_excel, get_numeric_columns, preprocess_for_analysis
from utils.file_manager import get_output_path, cleanup_old_reports
from core.analysis_engine import (
    analyze_y_vs_all, analyze_two_column, analyze_multi_x_vs_y,
    analyze_model_comparison, analyze_compare, analyze_crosstab,
)
from core.report_builder import build_report
from core.result_formatter import format_result
from core.excel_processor import ExcelProcessor
from core.chart_builder import build_chart
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


_CN_NUM = {
    "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
}


def _cn_num(s: str) -> int:
    """中文或阿拉伯数字字符串 → int，解析失败返回 5"""
    s = s.strip()
    if s.isdigit():
        return int(s)
    return _CN_NUM.get(s, 5)


def _extract_col(inst: str, all_cols: list) -> Optional[str]:
    """从指令文本中提取列名（优先最长匹配，大小写不敏感）"""
    cols_by_len = sorted(all_cols, key=len, reverse=True)
    for col in cols_by_len:
        if col in inst:
            return col
    inst_l = inst.lower()
    for col in cols_by_len:
        if col.lower() in inst_l:
            return col
    return None


def _clean_lookup_condition(inst: str) -> str:
    """
    从查找类指令中提取条件文本：
    1. 移除动词前缀（查找/筛选/过滤/找出/找到/搜索/显示）
    2. 移除常见结尾（的行/的记录/的数据/的/行/条）
    3. 去除首尾标点和空格
    """
    _lookup_verbs = ["查找", "筛选", "过滤", "找出", "找到", "搜索", "显示", "列出"]
    condition = inst
    for v in _lookup_verbs:
        condition = condition.replace(v, "")
    # 从后向前去掉多余尾缀
    _tails = ["的行", "的记录", "的数据", "的条目", "的数据行", "行", "记录", "条", "的"]
    changed = True
    while changed:
        changed = False
        for tail in _tails:
            if condition.endswith(tail):
                condition = condition[: -len(tail)]
                changed = True
                break
    return condition.strip("，,。.、 \t")


def _parse_process_instruction(inst: str, all_cols: list) -> Optional[dict]:
    """
    从自然语言指令检测 Excel 处理/运算操作，返回含 'op' 字段的参数字典，或 None。
    优先于 NLP 分析器执行，避免清洗/查找等操作被误判为统计分析。

    识别范围：
      clean         数据清洗（去重/填充/删除空列/标准化）
      lookup        查找/筛选（支持各种条件）
      count         统计行数/分组计数
      calc_sum      求和
      calc_mean     求均值
      calc_extremes 求极值（最大/最小/极差）
      split         按列拆分为多Sheet
    """
    # ── 【最高优先级】分析意图豁免：含统计/建模词汇时直接交给 NLP 分析器 ──────
    # 防止"找出影响Y的因素"被误判为数据查找，"建立模型"被误判为其他操作
    _analysis_exempt_kw = [
        # 统计分析
        "影响因素", "影响y", "影响Y", "相关性", "相关分析",
        "回归", "回归分析", "线性回归", "逻辑回归",
        # 机器学习
        "随机森林", "神经网络", "决策树", "支持向量", "梯度提升", "XGBoost", "LightGBM",
        # 建模意图
        "建立模型", "建模", "训练模型", "拟合", "预测模型",
        "特征重要性", "特征选择", "变量筛选",
        # 统计检验
        "方差分析", "ANOVA", "anova", "假设检验", "显著性",
        # 降维聚类
        "主成分", "PCA", "pca", "聚类", "cluster",
        # 时序
        "时间序列", "时序分析", "趋势预测", "ARIMA",
        # 多模型比较
        "比较模型", "模型比较", "效果最好", "最优模型", "多个模型",
    ]
    if any(k in inst for k in _analysis_exempt_kw):
        return None  # 交给 NLP 分析器 / AI 规划器处理

    # ── 数据清洗 ─────────────────────────────────────────────────────────
    # 关键词分组：只要命中其中任何一组，就确认是清洗操作
    _clean_kw = [
        "清洗", "预处理", "数据预处理",           # 通用清洗
        "去重", "去除重复", "重复行",              # 去重
        "填充缺失", "填充空值", "补全缺失",        # 填充
        "去除空列", "删除空列", "删除全空",        # 删除空列
        "去除全是空", "删除全是空", "空列",        # 删除空列（自然语言）
        "标准化",                                  # 标准化
    ]
    if any(k in inst for k in _clean_kw):
        drop_dup = any(k in inst for k in ["去重", "去除重复", "重复行", "重复"])
        fill_m: Optional[str] = None
        if any(k in inst for k in ["填充缺失", "填充空值", "补全缺失", "填充", "补全", "补缺"]):
            if "中位数" in inst:
                fill_m = "median"
            elif "众数" in inst:
                fill_m = "mode"
            elif any(k in inst for k in ["用0", "填0", "零填充", "填零"]):
                fill_m = "zero"
            else:
                fill_m = "mean"
        normalize = "标准化" in inst
        # 判断是否需要删除空列
        drop_empty = any(k in inst for k in [
            "去除空列", "删除空列", "删除全空", "去除全是空", "删除全是空", "空列",
            "全是空", "全空", "空的列",
        ])
        # 若只说"清洗/预处理"，默认去重+填充均值
        if not drop_dup and fill_m is None and not normalize and not drop_empty:
            drop_dup, fill_m = True, "mean"
        return {
            "op": "clean",
            "drop_duplicates": drop_dup,
            "fill_missing": fill_m or "mean",
            "normalize": normalize,
            "drop_empty_cols": drop_empty,
        }

    # ── 查找/筛选（必须在极值检测之前）─────────────────────────────────
    _lookup_kw = ["查找", "筛选", "过滤", "找出", "找到", "搜索", "显示", "列出"]
    if any(k in inst for k in _lookup_kw):
        condition = _clean_lookup_condition(inst)
        return {"op": "lookup", "condition": condition}

    # ── 计数/统计数量 ──────────────────────────────────────────────────
    _count_kw = ["计数", "统计数量", "有多少行", "有多少条", "统计行数", "共多少行"]
    if any(k in inst for k in _count_kw):
        return {"op": "count", "group_col": None, "condition": ""}

    # ── 各列总和（全列聚合，优先于单列求和检测）────────────────────
    _all_col_sum_kw = ["各列总和", "各列求和", "每列总和", "每列求和", "所有列求和",
                       "全列总和", "全部列求和", "所有列的总和", "各列之和"]
    if any(k in inst for k in _all_col_sum_kw):
        return {"op": "calc_all_col_sum"}

    # ── 各列均值（全列聚合）───────────────────────────────────────────
    _all_col_mean_kw = ["各列均值", "各列平均", "每列均值", "每列平均", "所有列均值",
                        "全列均值", "各列平均值"]
    if any(k in inst for k in _all_col_mean_kw):
        return {"op": "calc_all_col_mean"}

    # ── 求和 ─────────────────────────────────────────────────────────
    if any(k in inst for k in ["求和", "总和", "合计", "求总和", "加总"]):
        vc = _extract_col(inst, all_cols)
        return {"op": "calc_sum", "value_col": vc}

    # ── 均值/平均 ────────────────────────────────────────────────────
    if any(k in inst for k in ["均值", "平均值", "平均数", "求平均"]):
        vc = _extract_col(inst, all_cols)
        return {"op": "calc_mean", "value_col": vc}

    # ── 最大最小/极差（在查找之后，避免"查找最大"走这里）───────────
    if any(k in inst for k in ["最大值", "最小值", "极差", "极值", "求极值"]):
        vc = _extract_col(inst, all_cols)
        return {"op": "calc_extremes", "value_col": vc}

    # ── 按列数拆分（每 N 列一组）优先于按值拆分 ─────────────────────
    # 关键词匹配：覆盖"五列一组"、"按5列拆分"、"每N列"等各种表达
    _col_count_split_kw = ["列一组", "列为一组", "列分组", "按列数", "每列拆", "列拆分为多"]
    # 额外模式：指令同时含"N列"和拆分意图
    _m_n_col = re.search(r"([一两二三四五六七八九十\d]+)列", inst)
    _has_split_intent = any(k in inst for k in ["拆分", "分表", "多张表", "多sheet", "多Sheet"])
    if any(k in inst for k in _col_count_split_kw) or (_m_n_col and _has_split_intent):
        m_num = _m_n_col or re.search(r"([一两二三四五六七八九十\d]+)列", inst)
        n = _cn_num(m_num.group(1)) if m_num else 5
        return {"op": "split_col_count", "n": n}

    # ── 拆分（按列值分Sheet）────────────────────────────────────────
    if any(k in inst for k in ["拆分", "分组导出", "按列拆分", "分sheet", "分Sheet", "按列分"]):
        sc = _extract_col(inst, all_cols)
        return {"op": "split", "split_col": sc}

    return None


def _build_chart_from_analysis(analysis) -> Optional[dict]:
    """
    根据分析结果生成适合前端渲染的 ECharts option。
    各模式选择最有意义的图表类型。
    """
    try:
        from core.chart_builder import build_chart as _bc

        if analysis.mode == "y_vs_all" and analysis.feature_importance_df is not None:
            df = analysis.feature_importance_df[["Feature", "Importance"]].head(10).copy()
            df.columns = ["特征", "重要性"]
            return _bc("bar", df, "特征", ["重要性"],
                       title=f"全因子重要性排名 — {analysis.target_y}")

        elif analysis.mode == "multi_x_vs_y" and analysis.feature_importance_df is not None:
            df = analysis.feature_importance_df[["Feature", "Importance"]].head(10).copy()
            df.columns = ["特征", "重要性"]
            return _bc("bar", df, "特征", ["重要性"],
                       title=f"特征重要性 — {analysis.target_y}")

        elif analysis.mode == "compare" and analysis.compare_group_stats_df is not None:
            gs = analysis.compare_group_stats_df[["组别", "均值"]].copy()
            return _bc("bar", gs, "组别", ["均值"],
                       title=f"{analysis.compare_value_col} 各组均值对比")

        elif analysis.mode == "crosstab" and analysis.crosstab_df is not None:
            # 将透视表转成 bar，取第一列数值
            ct = analysis.crosstab_df.copy().reset_index()
            num_cols = [c for c in ct.columns if ct[c].dtype.kind in "iuf" and c != ct.columns[0]]
            if num_cols:
                return _bc("bar", ct, ct.columns[0], num_cols[:4],
                           title="交叉分析结果")

        elif analysis.mode == "anova" and analysis.anova_group_stats_df is not None:
            gs = analysis.anova_group_stats_df[["组别", "均值"]].copy()
            return _bc("bar", gs, "组别", ["均值"],
                       title=f"ANOVA — {analysis.target_y} 各组均值")

        elif analysis.mode == "model_comparison" and analysis.mc_comparison_df is not None:
            mc = analysis.mc_comparison_df[["模型", "CV均值R²"]].copy()
            return _bc("bar", mc, "模型", ["CV均值R²"],
                       title="多模型交叉验证 R² 对比")

        elif analysis.mode == "pca" and analysis.pca_explained_ratio:
            n = len(analysis.pca_explained_ratio)
            df = pd.DataFrame({
                "主成分": [f"PC{i+1}" for i in range(n)],
                "方差贡献率": analysis.pca_explained_ratio,
                "累计贡献率": analysis.pca_cumulative_ratio,
            })
            return _bc("bar", df, "主成分", ["方差贡献率"],
                       title="PCA 方差贡献率")

        elif analysis.mode == "cluster" and analysis.cluster_stats_df is not None:
            cs = analysis.cluster_stats_df.copy()
            num_cols = [c for c in cs.columns if cs[c].dtype.kind in "iuf"]
            cat_cols = [c for c in cs.columns if c not in num_cols]
            if cat_cols and num_cols:
                return _bc("bar", cs, cat_cols[0], num_cols[:2],
                           title="各簇统计分布")

    except Exception:
        pass
    return None


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

    # ── 处理/运算类指令前置检测（优先于 NLP，直接调 ExcelProcessor）──────────
    if req.use_ai and req.instruction.strip():
        _proc_params = _parse_process_instruction(
            req.instruction.strip(), list(df_raw.columns)
        )
        if _proc_params:
            op = _proc_params.pop("op")
            try:
                if op == "clean":
                    proc_result = ExcelProcessor.process_clean(df_raw, **_proc_params)
                elif op == "lookup":
                    proc_result = ExcelProcessor.process_lookup(
                        df_raw, _proc_params.get("condition", "")
                    )
                elif op == "count":
                    proc_result = ExcelProcessor.process_count(
                        df_raw,
                        _proc_params.get("group_col"),
                        _proc_params.get("condition", ""),
                    )
                elif op == "calc_sum":
                    vc = _proc_params.get("value_col")
                    if not vc:
                        return JSONResponse({
                            "success": False,
                            "error": f"请指定求和的列名，可用列：{list(df_raw.columns)[:10]}",
                        })
                    proc_result = ExcelProcessor.calc_sum(df_raw, vc)
                elif op == "calc_mean":
                    vc = _proc_params.get("value_col")
                    if not vc:
                        return JSONResponse({
                            "success": False,
                            "error": f"请指定均值的列名，可用列：{list(df_raw.columns)[:10]}",
                        })
                    proc_result = ExcelProcessor.calc_mean(df_raw, vc)
                elif op == "calc_extremes":
                    vc = _proc_params.get("value_col")
                    if not vc:
                        return JSONResponse({
                            "success": False,
                            "error": f"请指定列名，可用列：{list(df_raw.columns)[:10]}",
                        })
                    proc_result = ExcelProcessor.calc_extremes(df_raw, vc)
                elif op == "split":
                    sc = _proc_params.get("split_col")
                    if not sc:
                        return JSONResponse({
                            "success": False,
                            "error": f"请指定拆分列，可用列：{list(df_raw.columns)[:10]}",
                        })
                    proc_result = ExcelProcessor.process_split(df_raw, sc)
                elif op == "split_col_count":
                    n = _proc_params.get("n", 5)
                    proc_result = ExcelProcessor.split_by_col_count(df_raw, n)
                elif op == "calc_all_col_sum":
                    num_cols = df_raw.select_dtypes(include="number").columns.tolist()
                    if not num_cols:
                        return JSONResponse({
                            "success": False,
                            "error": "数据中未找到数值列，无法求和",
                        })
                    sums = df_raw[num_cols].sum()
                    result_df = pd.DataFrame({"列名": sums.index, "总和": sums.values})
                    from core.process_result import ProcessResult as _PR
                    proc_result = _PR(
                        operation="calc_all_col_sum",
                        summary_text=(
                            f"## ➕ 各列总和\n"
                            + "\n".join(f"- `{c}`：**{v:,.4g}**"
                                        for c, v in zip(sums.index, sums.values))
                        ),
                        result_df=result_df,
                        raw_row_count=len(df_raw),
                        valid_row_count=len(df_raw),
                    )
                elif op == "calc_all_col_mean":
                    num_cols = df_raw.select_dtypes(include="number").columns.tolist()
                    if not num_cols:
                        return JSONResponse({
                            "success": False,
                            "error": "数据中未找到数值列，无法求均值",
                        })
                    means = df_raw[num_cols].mean()
                    result_df = pd.DataFrame({"列名": means.index, "均值": means.values})
                    from core.process_result import ProcessResult as _PR
                    proc_result = _PR(
                        operation="calc_all_col_mean",
                        summary_text=(
                            f"## 📊 各列均值\n"
                            + "\n".join(f"- `{c}`：**{v:,.4g}**"
                                        for c, v in zip(means.index, means.values))
                        ),
                        result_df=result_df,
                        raw_row_count=len(df_raw),
                        valid_row_count=len(df_raw),
                    )
                else:
                    proc_result = None

                if proc_result is not None:
                    report_filename = None
                    try:
                        out_path = get_output_path(file_path.name, op)
                        _write_process_excel(proc_result, out_path)
                        cleanup_old_reports()
                        report_filename = out_path.name
                    except Exception:
                        pass
                    table_rows: list = []
                    if proc_result.result_df is not None and not proc_result.result_df.empty:
                        table_rows = proc_result.result_df.head(100).to_dict("records")
                    elif proc_result.result_sheets:
                        # 多 Sheet 结果：取第一张表的前 100 行作内联预览
                        first_sheet = next(iter(proc_result.result_sheets.values()))
                        table_rows = first_sheet.head(100).to_dict("records")
                    return JSONResponse({
                        "success": True,
                        "summary_text": proc_result.summary_text,
                        "table_data": table_rows,
                        "report_filename": report_filename,
                        "data_info": {
                            "raw": proc_result.raw_row_count,
                            "valid": proc_result.valid_row_count,
                        },
                    })
            except Exception as e:
                import traceback as _tb
                return JSONResponse({
                    "success": False,
                    "error": f"处理失败：{str(e)}\n{_tb.format_exc()}",
                })

    # ── DS-first：AI 规划器作为唯一大脑 ──────────────────────────────────────
    # 当用户有自然语言指令时，始终先让 DeepSeek 规划（包括分析类指令）。
    # NLP Parser 仅作为 DS 不可用时的 fallback。
    if req.use_ai and req.instruction.strip():
        ds_plan = None
        ds_error = None
        try:
            from core.ai_planner import AIPLanner
            planner = AIPLanner()
            ds_plan = planner.plan(df_raw, req.instruction.strip())
        except Exception as e:
            ds_error = str(e)

        if ds_plan:
            from core.skill_executor import execute_plan
            exec_result = execute_plan(df_raw, ds_plan)

            if not exec_result.success:
                # DS 规划执行失败 → 降级到 NLP Parser
                ds_error = exec_result.error
            elif exec_result.analysis_result is not None:
                # DS 调用了 run_analysis skill，走分析报告路径
                analysis = exec_result.analysis_result
                analysis.raw_row_count = len(df_raw)
                analysis.valid_row_count = len(exec_result.df)
                intent_info = {
                    "mode": analysis.mode,
                    "target_y": analysis.target_y,
                    "x_cols": [],
                    "hint": "DeepSeek 规划",
                    "confidence": 1.0,
                }
                # 走统一分析输出路径（跳过 NLP 分支）
                mode = analysis.mode
                raw_count = analysis.raw_row_count
                valid_count = analysis.valid_row_count
                # fall-through 到下方报告生成逻辑
            else:
                # DS 只做了数据处理（无 analysis_result），直接返回
                steps_summary = [s for s in exec_result.steps_summary if s]
                # 多表结果（split_columns）
                if exec_result.result_sheets:
                    first_sheet = next(iter(exec_result.result_sheets.values()))
                    table_rows = first_sheet.head(200).fillna("").to_dict("records")
                elif not exec_result.df.empty:
                    table_rows = exec_result.df.head(200).fillna("").to_dict("records")
                else:
                    table_rows = []
                report_filename = None
                try:
                    from core.process_result import ProcessResult as _PR
                    _pr = _PR(
                        operation="ai_plan",
                        summary_text=exec_result.full_summary,
                        result_df=exec_result.df if not exec_result.result_sheets else None,
                        result_sheets=exec_result.result_sheets,
                        raw_row_count=len(df_raw),
                        valid_row_count=len(exec_result.df),
                    )
                    out_path = get_output_path(file_path.name, "处理结果")
                    _write_process_excel(_pr, out_path)
                    cleanup_old_reports()
                    report_filename = out_path.name
                except Exception:
                    pass
                return JSONResponse({
                    "success": True,
                    "summary_text": exec_result.full_summary or "✅ 操作完成",
                    "table_data": table_rows,
                    "report_filename": report_filename,
                    "steps": steps_summary,
                    "chart_option": exec_result.chart_option,
                    "data_info": {"raw": len(df_raw), "valid": len(exec_result.df)},
                })

        if ds_error and not (exec_result.analysis_result if ds_plan and not exec_result.success else False):
            # DS 完全失败 → fallback 到 NLP Parser
            try:
                from core.nlp_parser import IntentParser
                parser = IntentParser()
                nlp_result = parser.parse(req.instruction, numeric_cols)
                if nlp_result.get("error"):
                    return JSONResponse({"success": False,
                                        "error": f"AI 规划失败：{ds_error}\nNLP 解析也失败：{nlp_result['error']}"})
                mode = nlp_result["analysis_mode"]
                target_y = nlp_result["target_y"]
                x_cols = nlp_result["x_columns"]
                intent_info = {
                    "mode": mode, "target_y": target_y, "x_cols": x_cols,
                    "hint": "NLP fallback", "confidence": nlp_result.get("confidence", 0.5),
                }
            except Exception as e2:
                return JSONResponse({"success": False,
                                     "error": f"AI 规划失败：{ds_error}\nNLP fallback 也失败：{e2}"})

    # ── 分析执行（DS run_analysis 或 NLP fallback 都走这里）──────────────────
    # 若已经通过 DS run_analysis 拿到了 analysis，直接跳到报告生成
    if 'analysis' not in dir():
        # NLP fallback 路径：手动执行分析
        _no_y_modes = {"pca", "cluster", "compare", "crosstab"}
        if not target_y or target_y not in numeric_cols:
            if mode == "multi_x_vs_y" and numeric_cols:
                target_y = numeric_cols[0]
            elif mode in _no_y_modes:
                pass
            else:
                return JSONResponse({"success": False,
                                     "error": f"目标变量 `{target_y}` 不存在，可用列：{numeric_cols[:5]}"})

        if mode == "multi_x_vs_y" and not x_cols:
            x_cols = [c for c in numeric_cols if c != target_y]

        try:
            from core.skill_executor import _run_analysis_skill
            _sr = _run_analysis_skill(df_raw, mode=mode, target_y=target_y, x_cols=x_cols)
            if _sr.error:
                return JSONResponse({"success": False, "error": f"分析失败：{_sr.error}"})
            analysis = _sr._analysis_result
            raw_count = len(df_raw)
            valid_count = len(_sr.df)
            analysis.raw_row_count = raw_count
            analysis.valid_row_count = valid_count
        except Exception as e:
            import traceback
            return JSONResponse({"success": False,
                                 "error": f"分析失败：{str(e)}\n{traceback.format_exc()}"})

    # 统一兜底：确保 raw_count/valid_count/mode/intent_info 始终有值
    raw_count = getattr(analysis, "raw_row_count", len(df_raw))
    valid_count = getattr(analysis, "valid_row_count", raw_count)
    mode = getattr(analysis, "mode", mode if 'mode' in dir() else "unknown")
    if 'intent_info' not in dir():
        intent_info = {"mode": mode, "target_y": getattr(analysis, "target_y", ""),
                       "x_cols": [], "hint": "", "confidence": 1.0}

    # 生成通俗报告（DeepSeek 翻译）
    plain_report = ""
    try:
        from core.plain_reporter import generate_plain_report
        plain_report = generate_plain_report(analysis)
    except Exception:
        pass

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
    format_instruction = req.instruction.strip() if req.instruction.strip() else req.manual_mode
    formatted_text = format_result(analysis, format_instruction)

    import math

    def _sanitize_val(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize_val(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_val(v) for v in obj]
        return obj

    chart_option = _sanitize_val(_build_chart_from_analysis(analysis))

    return _sanitize_val({
        "success": True,
        "summary_text": formatted_text,
        "plain_report": plain_report,
        "table_data": _build_table_data(analysis),
        "report_filename": report_filename,
        "intent": intent_info,
        "data_info": {"raw": raw_count, "valid": valid_count},
        "chart_option": chart_option,
    })


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


# ──────────────────────────────────────────────────────────
# API：Excel 处理 / 数据运算
# ──────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    file_id: str
    operation: str          # clean/lookup/count/merge/split/calc_sum/calc_mean/calc_extremes/calc_logic/calc_aggregate
    value_col: Optional[str] = None
    group_col: Optional[str] = None
    condition: str = ""
    split_col: Optional[str] = None
    agg_funcs: list[str] = []
    value_cols: list[str] = []
    # 清洗参数
    drop_duplicates: bool = True
    fill_missing: str = "mean"
    normalize: bool = False
    # 逻辑计算参数
    true_label: str = "是"
    false_label: str = "否"
    new_col: str = "逻辑结果"


def _write_process_excel(proc_result, out_path) -> None:
    """将 ProcessResult 写入 Excel（支持多 Sheet 拆分）"""
    import openpyxl
    wb = openpyxl.Workbook()
    if proc_result.result_sheets:
        # 多 Sheet（拆分操作）
        for i, (name, df) in enumerate(proc_result.result_sheets.items()):
            ws = wb.create_sheet(title=str(name)[:31]) if i > 0 else wb.active
            if i == 0:
                ws.title = str(name)[:31]
            for row in [df.columns.tolist()] + df.values.tolist():
                ws.append([str(v) for v in row])
    elif proc_result.result_df is not None:
        ws = wb.active
        ws.title = "结果"
        df = proc_result.result_df
        for row in [df.columns.tolist()] + df.values.tolist():
            ws.append([str(v) for v in row])
    wb.save(str(out_path))


@router.post("/api/process")
async def api_process(req: ProcessRequest):
    """Excel 处理 / 数据运算接口"""
    if req.file_id not in _uploaded_files:
        raise HTTPException(400, "文件不存在，请重新上传")

    file_path = _uploaded_files[req.file_id]
    try:
        df, _ = load_excel(str(file_path))
    except Exception as e:
        raise HTTPException(400, f"文件读取失败：{str(e)}")

    proc = ExcelProcessor()
    op = req.operation

    try:
        if op == "clean":
            result = proc.process_clean(df,
                drop_duplicates=req.drop_duplicates,
                fill_missing=req.fill_missing or "mean",
                normalize=req.normalize)
        elif op == "lookup":
            result = proc.process_lookup(df, req.condition)
        elif op == "count":
            result = proc.process_count(df, req.group_col, req.condition)
        elif op == "split":
            if not req.split_col:
                return JSONResponse({"success": False, "error": "split 操作需要指定 split_col"})
            result = proc.process_split(df, req.split_col)
        elif op == "merge":
            # 单文件合并模式：按 sheet 合并
            sheets = pd.read_excel(str(file_path), sheet_name=None)
            result = proc.process_merge(list(sheets.values()))
        elif op == "calc_sum":
            if not req.value_col:
                return JSONResponse({"success": False, "error": "calc_sum 需要 value_col"})
            result = proc.calc_sum(df, req.value_col, req.group_col, req.condition)
        elif op == "calc_mean":
            if not req.value_col:
                return JSONResponse({"success": False, "error": "calc_mean 需要 value_col"})
            result = proc.calc_mean(df, req.value_col, req.group_col, req.condition)
        elif op == "calc_extremes":
            if not req.value_col:
                return JSONResponse({"success": False, "error": "calc_extremes 需要 value_col"})
            result = proc.calc_extremes(df, req.value_col, req.group_col)
        elif op == "calc_logic":
            if not req.value_col or not req.condition:
                return JSONResponse({"success": False,
                                     "error": "calc_logic 需要 value_col 和 condition"})
            result = proc.calc_logic(df, req.value_col, req.condition,
                                     req.true_label, req.false_label, req.new_col)
        elif op == "calc_aggregate":
            cols = req.value_cols or [c for c in df.select_dtypes(include="number").columns]
            result = proc.calc_aggregate(df, cols, req.group_col,
                                         req.agg_funcs or None)
        else:
            return JSONResponse({"success": False, "error": f"未知操作：{op}"})
    except Exception as e:
        import traceback
        return JSONResponse({"success": False,
                             "error": f"处理失败：{str(e)}\n{traceback.format_exc()}"})

    # 写入 Excel
    report_filename = None
    _write_error = None
    try:
        from utils.file_manager import get_output_path
        out_path = get_output_path(file_path.name, op)
        _write_process_excel(result, out_path)
        cleanup_old_reports()
        report_filename = out_path.name
    except Exception as _e:
        _write_error = str(_e)  # 不静默吞掉，供调试

    # 转换结果为前端可渲染的表格（优先取多 Sheet 第一张，兜底取 result_df）
    import math as _math

    def _safe_rows(df: pd.DataFrame, limit: int = 100) -> list:
        """把 DataFrame 转为 JSON 安全的 records，NaN/Inf 替换为 None。"""
        preview = df.head(limit)
        records = preview.where(pd.notnull(preview), None).to_dict("records")
        def _fix(v):
            if isinstance(v, float) and (_math.isnan(v) or _math.isinf(v)):
                return None
            return v
        return [{k: _fix(v) for k, v in row.items()} for row in records]

    table_rows = []
    if result.result_sheets:
        first_sheet = next(iter(result.result_sheets.values()))
        table_rows = _safe_rows(first_sheet)
    elif result.result_df is not None and not result.result_df.empty:
        table_rows = _safe_rows(result.result_df)

    return {
        "success": True,
        "summary_text": result.summary_text,
        "table_data": table_rows,
        "report_filename": report_filename,
        "data_info": {"raw": result.raw_row_count, "valid": result.valid_row_count},
        **({"write_warning": _write_error} if _write_error else {}),
    }


# ──────────────────────────────────────────────────────────
# API：ECharts 图表生成
# ──────────────────────────────────────────────────────────

class ChartRequest(BaseModel):
    file_id: str
    chart_type: str        # bar/pie/line/area/combo/scatter
    x_col: str
    y_cols: list[str]
    title: str = ""
    condition: str = ""    # 可选过滤条件
    extra: dict = {}


@router.post("/api/chart")
async def api_chart(req: ChartRequest):
    """生成 ECharts option JSON，前端直接 setOption()"""
    if req.file_id not in _uploaded_files:
        raise HTTPException(400, "文件不存在，请重新上传")

    file_path = _uploaded_files[req.file_id]
    try:
        df, _ = load_excel(str(file_path))
    except Exception as e:
        raise HTTPException(400, f"文件读取失败：{str(e)}")

    # 可选过滤
    if req.condition:
        from core.excel_processor import _apply_condition
        df = _apply_condition(df, req.condition)

    # 列存在性检查
    all_cols = [req.x_col] + req.y_cols
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        return JSONResponse({"success": False,
                             "error": f"列不存在：{missing}，可用列：{df.columns.tolist()[:10]}"})

    try:
        option = build_chart(
            chart_type=req.chart_type,
            df=df,
            x_col=req.x_col,
            y_cols=req.y_cols,
            title=req.title,
            extra=req.extra,
        )
    except Exception as e:
        return JSONResponse({"success": False, "error": f"图表生成失败：{str(e)}"})

    return {"success": True, "option": option}


# ═══════════════════════════════════════════════════════════════════════════
# 新架构：AI-as-Brain  /api/chat
# ═══════════════════════════════════════════════════════════════════════════
from core.ai_planner import AIPLanner
from core.skill_executor import execute_plan
from core.process_result import ProcessResult


class ChatRequest(BaseModel):
    file_id: str
    instruction: str


@router.post("/api/chat")
async def api_chat(req: ChatRequest):
    """
    新架构入口：AI 大脑 + Skills 工具。
    接收自然语言指令 → AI 规划 skill 序列 → 执行 → 返回结果。
    """
    if req.file_id not in _uploaded_files:
        raise HTTPException(400, "文件不存在，请重新上传")

    # 分析类指令前置检测：自动转发到分析引擎，无需用户手动切换模式
    _analysis_kw = [
        "分析", "回归", "相关", "影响", "因素", "预测", "模型",
        "聚类", "降维", "pca", "anova", "方差分析", "t检验",
        "对比分析", "交叉分析", "时间趋势", "时序",
    ]
    inst_lower = req.instruction.lower()
    if any(kw in inst_lower for kw in _analysis_kw):
        # 构造 AnalyzeRequest 并复用 api_analyze 逻辑
        analyze_req = AnalyzeRequest(
            file_id=req.file_id,
            instruction=req.instruction,
            use_ai=True,
        )
        return await api_analyze(analyze_req)

    file_path = _uploaded_files[req.file_id]
    try:
        df_raw, _ = load_excel(str(file_path))
    except Exception as e:
        raise HTTPException(400, f"文件读取失败：{str(e)}")

    # 1. AI 规划
    try:
        planner = AIPLanner()
        plan = planner.plan(df_raw, req.instruction)
    except Exception as e:
        return JSONResponse({"success": False, "error": f"AI 规划失败：{str(e)}"})

    # 2. 执行 plan
    exec_result = execute_plan(df_raw, plan)

    if not exec_result.success:
        return JSONResponse({"success": False, "error": exec_result.error})

    # 3. 保存结果文件
    report_filename = None
    try:
        proc_result = ProcessResult(
            operation="chat",
            summary_text=exec_result.full_summary,
            result_df=exec_result.df if not exec_result.result_sheets else None,
            result_sheets=exec_result.result_sheets or {},
        )
        proc_result.raw_row_count = len(df_raw)
        proc_result.valid_row_count = len(exec_result.df)
        out_path = get_output_path(file_path.name, "chat")
        _write_process_excel(proc_result, out_path)
        cleanup_old_reports()
        report_filename = out_path.name
    except Exception:
        pass

    import math

    def _sanitize(obj):
        """递归将 NaN/Inf 替换为 None，确保 JSON 序列化安全。"""
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    table_rows = []
    if exec_result.result_sheets:
        first_sheet = next(iter(exec_result.result_sheets.values()))
        _safe_df = first_sheet.head(200).where(pd.notnull(first_sheet.head(200)), None)
        table_rows = _sanitize(_safe_df.to_dict("records"))
    elif exec_result.df is not None and not exec_result.df.empty:
        _safe_df = exec_result.df.head(200).where(pd.notnull(exec_result.df.head(200)), None)
        table_rows = _sanitize(_safe_df.to_dict("records"))

    return JSONResponse(_sanitize({
        "success": True,
        "summary_text": exec_result.full_summary,
        "steps": exec_result.steps_summary,
        "table_data": table_rows,
        "chart_option": exec_result.chart_option,
        "report_filename": report_filename,
        "data_info": {
            "raw": len(df_raw),
            "valid": len(exec_result.df),
        },
    }))
