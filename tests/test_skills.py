"""Skills 单元测试"""
import pytest
import pandas as pd
from core.skills import (
    SkillResult,
    skill_delete_columns,
    skill_delete_rows,
    skill_filter_rows,
    skill_fill_missing,
    skill_deduplicate,
    skill_rename_column,
    skill_sort,
    skill_aggregate,
    skill_add_column,
    skill_build_chart,
    skill_describe,
    SKILL_REGISTRY,
)


@pytest.fixture
def df():
    return pd.DataFrame({
        "姓名": ["张三", "李四", "王五", "张三"],
        "年龄": [25, None, 30, 25],
        "部门": ["销售", "研发", "销售", "销售"],
        "销售额": [1200, 800, 1500, 1200],
    })


class TestDeleteColumns:
    def test_delete_one(self, df):
        r = skill_delete_columns(df, columns=["年龄"])
        assert "年龄" not in r.df.columns
        assert "姓名" in r.df.columns
        assert r.error is None

    def test_delete_multiple(self, df):
        r = skill_delete_columns(df, columns=["年龄", "部门"])
        assert "年龄" not in r.df.columns
        assert "部门" not in r.df.columns

    def test_nonexistent_column_returns_error(self, df):
        r = skill_delete_columns(df, columns=["不存在列"])
        assert r.error is not None

    def test_empty_list_returns_error(self, df):
        r = skill_delete_columns(df, columns=[])
        assert r.error is not None


class TestDeleteRows:
    def test_delete_by_condition(self, df):
        r = skill_delete_rows(df, condition="销售额 > 1000")
        assert len(r.df) == 1
        assert all(r.df["销售额"] <= 1000)

    def test_invalid_condition_returns_error(self, df):
        r = skill_delete_rows(df, condition="不存在列 > 0")
        assert r.error is not None


class TestFilterRows:
    def test_gt(self, df):
        r = skill_filter_rows(df, condition="销售额 > 1000")
        assert len(r.df) == 3
        assert all(r.df["销售额"] > 1000)

    def test_eq_string(self, df):
        r = skill_filter_rows(df, condition="部门 == 研发")
        assert len(r.df) == 1

    def test_contains(self, df):
        r = skill_filter_rows(df, condition="姓名 包含 张")
        assert len(r.df) == 2

    def test_chinese_op_gt(self, df):
        r = skill_filter_rows(df, condition="销售额大于1000")
        assert len(r.df) == 3

    def test_top_n_max(self, df):
        r = skill_filter_rows(df, condition="销售额最大的两行")
        assert len(r.df) == 2
        assert r.df["销售额"].max() == 1500

    def test_empty_condition_returns_all(self, df):
        r = skill_filter_rows(df, condition="")
        assert len(r.df) == len(df)


class TestFillMissing:
    def test_fill_mean(self, df):
        r = skill_fill_missing(df, method="mean")
        assert r.df["年龄"].isnull().sum() == 0
        assert r.df["年龄"].iloc[1] == pytest.approx(26.67, rel=0.01)

    def test_fill_zero(self, df):
        r = skill_fill_missing(df, method="zero")
        assert r.df["年龄"].iloc[1] == 0

    def test_fill_specific_columns(self, df):
        r = skill_fill_missing(df, method="mean", columns=["年龄"])
        assert r.df["年龄"].isnull().sum() == 0


class TestDeduplicate:
    def test_dedup_all_cols(self, df):
        r = skill_deduplicate(df)
        assert len(r.df) == 3

    def test_dedup_subset(self, df):
        r = skill_deduplicate(df, columns=["部门"])
        assert len(r.df) == 2


class TestRenameColumn:
    def test_rename(self, df):
        r = skill_rename_column(df, old_name="销售额", new_name="revenue")
        assert "revenue" in r.df.columns
        assert "销售额" not in r.df.columns

    def test_rename_nonexistent_returns_error(self, df):
        r = skill_rename_column(df, old_name="不存在", new_name="x")
        assert r.error is not None


class TestSort:
    def test_sort_desc(self, df):
        r = skill_sort(df, column="销售额", ascending=False)
        vals = r.df["销售额"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_sort_asc(self, df):
        r = skill_sort(df, column="年龄", ascending=True)
        non_null = r.df["年龄"].dropna().tolist()
        assert non_null == sorted(non_null)


class TestAggregate:
    def test_group_sum(self, df):
        r = skill_aggregate(df, group_by=["部门"], agg={"销售额": "sum"})
        assert "销售额_sum" in r.df.columns
        sales_sum = r.df[r.df["部门"] == "销售"]["销售额_sum"].values[0]
        assert sales_sum == 3900

    def test_group_mean(self, df):
        r = skill_aggregate(df, group_by=["部门"], agg={"销售额": "mean"})
        assert "销售额_mean" in r.df.columns

    def test_invalid_col_returns_error(self, df):
        r = skill_aggregate(df, group_by=["不存在列"], agg={"销售额": "sum"})
        assert r.error is not None


class TestAddColumn:
    def test_arithmetic_expression(self, df):
        r = skill_add_column(df, name="提成", expression="销售额 * 0.1")
        assert "提成" in r.df.columns
        assert r.df["提成"].iloc[0] == pytest.approx(120.0)

    def test_invalid_expression_returns_error(self, df):
        r = skill_add_column(df, name="x", expression="不存在列 + 1")
        assert r.error is not None


class TestBuildChart:
    @pytest.fixture
    def agg_df(self):
        return pd.DataFrame({"部门": ["销售", "研发"], "销售额_sum": [3900, 800]})

    def test_bar(self, agg_df):
        r = skill_build_chart(agg_df, chart_type="bar", x="部门", y=["销售额_sum"])
        assert r.chart_option is not None
        assert r.chart_option["series"][0]["type"] == "bar"

    def test_pie(self, agg_df):
        r = skill_build_chart(agg_df, chart_type="pie", x="部门", y=["销售额_sum"])
        assert r.chart_option["series"][0]["type"] == "pie"

    def test_line(self, agg_df):
        r = skill_build_chart(agg_df, chart_type="line", x="部门", y=["销售额_sum"])
        assert r.chart_option["series"][0]["type"] == "line"

    def test_unknown_type_returns_error(self, agg_df):
        r = skill_build_chart(agg_df, chart_type="unknown", x="部门", y=["销售额_sum"])
        assert r.error is not None


class TestDescribe:
    def test_describe_all(self, df):
        r = skill_describe(df)
        assert r.df is not None
        assert len(r.df) > 0

    def test_describe_subset(self, df):
        r = skill_describe(df, columns=["销售额"])
        assert r.df is not None


class TestSkillRegistry:
    def test_all_skills_registered(self):
        expected = {
            "delete_columns", "delete_rows", "filter_rows",
            "fill_missing", "deduplicate", "rename_column",
            "sort", "aggregate", "add_column", "build_chart", "describe",
        }
        assert expected.issubset(set(SKILL_REGISTRY.keys()))

    def test_each_entry_has_description(self):
        for name, meta in SKILL_REGISTRY.items():
            assert "description" in meta, f"{name} 缺少 description"
            assert "params_schema" in meta, f"{name} 缺少 params_schema"
            assert "fn" in meta, f"{name} 缺少 fn"
