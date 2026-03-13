"""
新功能集成测试：Excel处理 + 数据运算 + 对比/交叉分析 + ECharts图表生成
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.process_result import ProcessResult
from core.excel_processor import ExcelProcessor, _apply_condition
from core.analysis_engine import analyze_compare, analyze_crosstab
from core.chart_builder import build_chart, build_bar, build_pie, build_line

# 从 routes 模块导入路由辅助函数（Bug 1 验证）
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
from routes import _parse_process_instruction


# ──────────────────────────────────────────────────────────
# 测试数据工厂
# ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "部门": ["销售", "研发", "销售", "研发", "市场", "市场", "销售"],
        "姓名": ["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九"],
        "销售额": [1200, 800, 1500, 900, 1100, 700, 1300],
        "月份": ["2024-01", "2024-01", "2024-02", "2024-02", "2024-01", "2024-02", "2024-03"],
        "评级": [1, 0, 1, 1, 0, 0, 1],
    })


# ──────────────────────────────────────────────────────────
# ProcessResult 数据类
# ──────────────────────────────────────────────────────────

class TestProcessResult:
    def test_default_fields(self):
        r = ProcessResult(operation="clean", summary_text="ok")
        assert r.operation == "clean"
        assert r.result_df is None
        assert r.result_sheets == {}
        assert r.extra == {}

    def test_with_df(self, sample_df):
        r = ProcessResult(operation="lookup", summary_text="found", result_df=sample_df)
        assert len(r.result_df) == 7


# ──────────────────────────────────────────────────────────
# _apply_condition 条件解析
# ──────────────────────────────────────────────────────────

class TestApplyCondition:
    def test_gt(self, sample_df):
        result = _apply_condition(sample_df, "销售额 > 1000")
        assert len(result) == 4
        assert all(result["销售额"] > 1000)

    def test_eq_string(self, sample_df):
        result = _apply_condition(sample_df, "部门 == 销售")
        assert len(result) == 3

    def test_contains(self, sample_df):
        result = _apply_condition(sample_df, "月份 包含 2024-01")
        assert len(result) == 3

    def test_le(self, sample_df):
        result = _apply_condition(sample_df, "销售额 <= 900")
        assert len(result) == 3

    def test_empty_condition_returns_all(self, sample_df):
        result = _apply_condition(sample_df, "")
        assert len(result) == len(sample_df)

    def test_max_natural_language(self, sample_df):
        """Bug 2 修复验证：'列名最大' 极值语义解析"""
        result = _apply_condition(sample_df, "销售额最大")
        assert len(result) == 1
        assert result["销售额"].values[0] == 1500

    def test_min_natural_language(self, sample_df):
        result = _apply_condition(sample_df, "销售额最小的行")
        assert len(result) == 1
        assert result["销售额"].values[0] == 700

    def test_max_case_insensitive(self, sample_df):
        """列名大小写不敏感匹配"""
        df = sample_df.rename(columns={"销售额": "Y"})
        result = _apply_condition(df, "y最大")  # 小写 y 匹配大写 Y 列
        assert result["Y"].values[0] == df["Y"].max()

    # ── Top-N 极值 ──────────────────────────────────────────

    def test_top_n_max_chinese(self, sample_df):
        """'列名最大的三行' → nlargest(3)"""
        result = _apply_condition(sample_df, "销售额最大的三")
        assert len(result) == 3
        assert result["销售额"].max() == 1500
        # 应当按降序排列（nlargest 本身有序）
        vals = result["销售额"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_top_n_max_digit(self, sample_df):
        """'列名最大的5行' → nlargest(5)"""
        result = _apply_condition(sample_df, "销售额最大的5")
        assert len(result) == 5

    def test_top_n_min_chinese(self, sample_df):
        """'列名最小的两行' → nsmallest(2)"""
        result = _apply_condition(sample_df, "销售额最小的两")
        assert len(result) == 2
        assert result["销售额"].min() == 700

    def test_top_n_with_suffix(self, sample_df):
        """带'行'后缀的 top-N"""
        result = _apply_condition(sample_df, "销售额最大的3行")
        assert len(result) == 3

    def test_top_n_case_insensitive(self, sample_df):
        """列名大小写不敏感 + Top-N"""
        df = sample_df.rename(columns={"销售额": "Y"})
        result = _apply_condition(df, "y最大的三")
        assert len(result) == 3
        assert result["Y"].iloc[0] == df["Y"].max()

    # ── 中文比较运算符 ───────────────────────────────────────

    def test_zh_op_gt(self, sample_df):
        result = _apply_condition(sample_df, "销售额大于1000")
        assert len(result) == 4
        assert all(result["销售额"] > 1000)

    def test_zh_op_lt(self, sample_df):
        result = _apply_condition(sample_df, "销售额小于900")
        assert len(result) == 2

    def test_zh_op_gte(self, sample_df):
        result = _apply_condition(sample_df, "销售额大于等于1500")
        assert len(result) == 1

    def test_zh_op_lte(self, sample_df):
        result = _apply_condition(sample_df, "销售额小于等于800")
        assert len(result) == 2

    def test_zh_op_eq_str(self, sample_df):
        result = _apply_condition(sample_df, "部门等于销售")
        assert len(result) == 3

    def test_zh_op_eq_for(self, sample_df):
        """'为' 也映射到 =="""
        result = _apply_condition(sample_df, "部门为销售")
        assert len(result) == 3


# ──────────────────────────────────────────────────────────
# ExcelProcessor — Excel处理操作
# ──────────────────────────────────────────────────────────

class TestExcelProcessorClean:
    def test_deduplicate(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result = ExcelProcessor.process_clean(df, drop_duplicates=True, fill_missing=None)
        assert result.operation == "clean"
        assert result.valid_row_count == 2

    def test_fill_mean(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        result = ExcelProcessor.process_clean(df, drop_duplicates=False, fill_missing="mean")
        assert result.result_df["a"].isnull().sum() == 0
        assert result.result_df["a"].iloc[1] == pytest.approx(2.0)

    def test_summary_contains_stats(self, sample_df):
        result = ExcelProcessor.process_clean(sample_df)
        assert "清洗后" in result.summary_text

    def test_drop_empty_cols(self):
        """去除全空列"""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "空列": [None, None, None],   # 应被删除
            "b": [4, 5, 6],
        })
        result = ExcelProcessor.process_clean(
            df, drop_duplicates=False, fill_missing=None, drop_empty_cols=True
        )
        assert "空列" not in result.result_df.columns
        assert "a" in result.result_df.columns
        assert "删除全空列" in result.summary_text

    def test_keep_non_empty_cols(self):
        """有值的列不能被删除"""
        df = pd.DataFrame({
            "a": [1, None, 3],
            "b": [None, None, None],
        })
        result = ExcelProcessor.process_clean(
            df, drop_duplicates=False, fill_missing=None, drop_empty_cols=True
        )
        assert "a" in result.result_df.columns
        assert "b" not in result.result_df.columns


class TestExcelProcessorLookup:
    def test_basic_filter(self, sample_df):
        result = ExcelProcessor.process_lookup(sample_df, "销售额 > 1200")
        assert result.operation == "lookup"
        assert result.valid_row_count == 2

    def test_empty_condition(self, sample_df):
        result = ExcelProcessor.process_lookup(sample_df, "")
        assert result.valid_row_count == len(sample_df)


class TestExcelProcessorCount:
    def test_group_count(self, sample_df):
        result = ExcelProcessor.process_count(sample_df, group_col="部门")
        assert result.operation == "count"
        assert len(result.result_df) == 3  # 3个部门

    def test_total_count(self, sample_df):
        result = ExcelProcessor.process_count(sample_df)
        assert result.result_df["值"][0] == 7

    def test_count_with_condition(self, sample_df):
        result = ExcelProcessor.process_count(sample_df, condition="销售额 > 1000")
        assert result.valid_row_count == 4


class TestExcelProcessorMerge:
    def test_concat_two_dfs(self, sample_df):
        df2 = sample_df.copy()
        result = ExcelProcessor.process_merge([sample_df, df2])
        assert result.operation == "merge"
        assert result.valid_row_count == 14


class TestExcelProcessorSplit:
    def test_split_by_dept(self, sample_df):
        result = ExcelProcessor.process_split(sample_df, "部门")
        assert result.operation == "split"
        assert len(result.result_sheets) == 3
        assert "销售" in result.result_sheets
        assert len(result.result_sheets["销售"]) == 3

    def test_invalid_col(self, sample_df):
        result = ExcelProcessor.process_split(sample_df, "不存在列")
        assert "不存在" in result.summary_text


# ──────────────────────────────────────────────────────────
# ExcelProcessor — 数据运算
# ──────────────────────────────────────────────────────────

class TestCalcSum:
    def test_total_sum(self, sample_df):
        result = ExcelProcessor.calc_sum(sample_df, "销售额")
        assert result.operation == "calc_sum"
        assert result.result_df["求和"].values[0] == 7500

    def test_group_sum(self, sample_df):
        result = ExcelProcessor.calc_sum(sample_df, "销售额", group_col="部门")
        df = result.result_df
        assert "销售额_求和" in df.columns
        sales_sum = df[df["部门"] == "销售"]["销售额_求和"].values[0]
        assert sales_sum == 4000  # 1200+1500+1300


class TestCalcMean:
    def test_total_mean(self, sample_df):
        result = ExcelProcessor.calc_mean(sample_df, "销售额")
        assert result.operation == "calc_mean"
        avg = sample_df["销售额"].mean()
        assert abs(result.result_df["均值"].values[0] - avg) < 0.01

    def test_group_mean(self, sample_df):
        result = ExcelProcessor.calc_mean(sample_df, "销售额", group_col="部门")
        assert "销售额_均值" in result.result_df.columns


class TestCalcExtremes:
    def test_extremes(self, sample_df):
        result = ExcelProcessor.calc_extremes(sample_df, "销售额")
        assert result.result_df["最大值"].values[0] == 1500
        assert result.result_df["最小值"].values[0] == 700
        assert result.result_df["极差"].values[0] == 800


class TestCalcLogic:
    def test_logic_label(self, sample_df):
        result = ExcelProcessor.calc_logic(sample_df, "销售额", "> 1000", "达标", "未达标")
        assert "逻辑结果" in result.result_df.columns
        assert (result.result_df["逻辑结果"] == "达标").sum() == 4


class TestCalcAggregate:
    def test_aggregate_all(self, sample_df):
        result = ExcelProcessor.calc_aggregate(sample_df, ["销售额"], agg_funcs=["sum", "mean"])
        assert result.operation == "calc_aggregate"
        assert "销售额_sum" in result.result_df.columns
        assert "销售额_mean" in result.result_df.columns


# ──────────────────────────────────────────────────────────
# 对比分析（analyze_compare）
# ──────────────────────────────────────────────────────────

class TestAnalyzeCompare:
    def test_two_groups(self, sample_df):
        df2 = sample_df[sample_df["部门"].isin(["销售", "研发"])].copy()
        result = analyze_compare(df2, "销售额", "部门")
        assert result.mode == "compare"
        assert result.compare_test_name == "独立样本 t 检验"
        assert result.compare_group_stats_df is not None
        assert len(result.compare_group_stats_df) == 2
        assert "对比分析" in result.summary_text

    def test_three_groups(self, sample_df):
        result = analyze_compare(sample_df, "销售额", "部门")
        assert result.compare_test_name == "单因素 ANOVA"
        assert result.compare_p_value >= 0  # p 值在 [0,1]


# ──────────────────────────────────────────────────────────
# 交叉分析（analyze_crosstab）
# ──────────────────────────────────────────────────────────

class TestAnalyzeCrosstab:
    def test_count_crosstab(self, sample_df):
        result = analyze_crosstab(sample_df, "部门", "月份")
        assert result.mode == "crosstab"
        assert result.crosstab_df is not None
        assert result.crosstab_df.shape[0] == 3   # 3 个部门
        assert "交叉分析" in result.summary_text

    def test_sum_pivot(self, sample_df):
        result = analyze_crosstab(sample_df, "部门", "月份", "销售额", agg="sum")
        assert result.crosstab_agg == "sum"
        assert result.crosstab_df is not None


# ──────────────────────────────────────────────────────────
# ECharts 图表生成（chart_builder）
# ──────────────────────────────────────────────────────────

@pytest.fixture
def chart_df():
    return pd.DataFrame({
        "类别": ["A", "B", "C", "D"],
        "数量": [100, 200, 150, 80],
        "金额": [1000, 2000, 1500, 800],
    })


class TestChartBuilder:
    def test_bar_chart(self, chart_df):
        opt = build_bar(chart_df, "类别", ["数量"])
        assert opt["series"][0]["type"] == "bar"
        assert len(opt["series"][0]["data"]) == 4
        assert opt["xAxis"]["data"] == ["A", "B", "C", "D"]

    def test_bar_multi_series(self, chart_df):
        opt = build_bar(chart_df, "类别", ["数量", "金额"])
        assert len(opt["series"]) == 2

    def test_pie_chart(self, chart_df):
        opt = build_pie(chart_df, "类别", "数量")
        assert opt["series"][0]["type"] == "pie"
        assert len(opt["series"][0]["data"]) == 4
        assert opt["series"][0]["data"][0]["name"] == "A"

    def test_line_chart(self, chart_df):
        opt = build_line(chart_df, "类别", ["数量"])
        assert opt["series"][0]["type"] == "line"
        assert opt["series"][0]["smooth"] is True

    def test_area_chart(self, chart_df):
        opt = build_chart("area", chart_df, "类别", ["数量"])
        assert "areaStyle" in opt["series"][0]

    def test_scatter_chart(self, chart_df):
        opt = build_chart("scatter", chart_df, "数量", ["金额"])
        assert opt["series"][0]["type"] == "scatter"

    def test_combo_chart(self, chart_df):
        opt = build_chart("combo", chart_df, "类别", ["数量"],
                          extra={"line_cols": ["金额"]})
        types = [s["type"] for s in opt["series"]]
        assert "bar" in types
        assert "line" in types

    def test_unknown_type_raises(self, chart_df):
        with pytest.raises(ValueError, match="未知图表类型"):
            build_chart("unknown", chart_df, "类别", ["数量"])

    def test_unified_entry_bar(self, chart_df):
        opt = build_chart("bar", chart_df, "类别", ["数量"], title="测试标题")
        assert opt["title"]["text"] == "测试标题"

    def test_has_toolbox(self, chart_df):
        opt = build_bar(chart_df, "类别", ["数量"])
        assert "toolbox" in opt

    def test_color_palette(self, chart_df):
        opt = build_bar(chart_df, "类别", ["数量"])
        assert len(opt["color"]) >= 6


# ──────────────────────────────────────────────────────────
# Bug 修复验证：_parse_process_instruction 路由检测（Bug 1）
# ──────────────────────────────────────────────────────────

class TestParseProcessInstruction:
    COLS = ["部门", "姓名", "销售额", "月份", "评级"]

    def test_clean_detected(self):
        """'清洗' 指令应被识别为 clean，不走 NLP"""
        p = _parse_process_instruction("对数据进行清洗，去除重复行并填充缺失值", self.COLS)
        assert p is not None
        assert p["op"] == "clean"
        assert p["drop_duplicates"] is True
        assert p["fill_missing"] == "mean"

    def test_clean_preprocess(self):
        p = _parse_process_instruction("数据预处理", self.COLS)
        assert p is not None and p["op"] == "clean"

    def test_lookup_detected(self):
        """'查找' 指令应被识别为 lookup"""
        p = _parse_process_instruction("查找销售额最大的行", self.COLS)
        assert p is not None
        assert p["op"] == "lookup"
        assert "销售额最大" in p["condition"]

    def test_lookup_filter(self):
        p = _parse_process_instruction("筛选销售额 > 1000的行", self.COLS)
        assert p is not None and p["op"] == "lookup"

    def test_count_detected(self):
        p = _parse_process_instruction("统计数量", self.COLS)
        assert p is not None and p["op"] == "count"

    def test_calc_sum_detected(self):
        p = _parse_process_instruction("对销售额求和", self.COLS)
        assert p is not None and p["op"] == "calc_sum"

    def test_analysis_not_matched(self):
        """分析类指令不应被误匹配为处理操作"""
        p = _parse_process_instruction("分析哪些因素影响了销售额", self.COLS)
        assert p is None

    def test_none_for_regression(self):
        p = _parse_process_instruction("对销售额做多元回归分析", self.COLS)
        assert p is None

    # ── 新增场景：用户实际触发的问题 ─────────────────────────

    def test_clean_drop_empty_cols(self):
        """'去除全是空元素的列' → clean + drop_empty_cols=True"""
        p = _parse_process_instruction("对数据进行清洗，去除全是空元素的列", self.COLS)
        assert p is not None
        assert p["op"] == "clean"
        assert p["drop_empty_cols"] is True

    def test_clean_drop_empty_cols2(self):
        p = _parse_process_instruction("删除空列并去重", self.COLS)
        assert p is not None and p["op"] == "clean"
        assert p["drop_empty_cols"] is True

    def test_lookup_top_n_chinese(self):
        """'查找y最大的三行' → lookup，条件 = 'y最大的三'"""
        p = _parse_process_instruction("查找y最大的三行", self.COLS)
        assert p is not None
        assert p["op"] == "lookup"
        # 条件应包含列名和极值语义，不应还保留动词或"行"后缀
        cond = p["condition"]
        assert "查找" not in cond
        assert cond.endswith("行") is False  # "行" 后缀应已被去掉

    def test_lookup_condition_extraction(self):
        """'筛选销售额大于1000的行' → condition = '销售额大于1000'"""
        p = _parse_process_instruction("筛选销售额大于1000的行", self.COLS)
        assert p is not None and p["op"] == "lookup"
        assert p["condition"] == "销售额大于1000"

    def test_lookup_no_trailing_de(self):
        """条件提取后不应残留孤立的'的'"""
        p = _parse_process_instruction("查找月份包含2024-01的行", self.COLS)
        assert p is not None
        cond = p["condition"]
        assert cond == "月份包含2024-01"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
