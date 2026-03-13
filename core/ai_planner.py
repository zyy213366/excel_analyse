"""
AI Planner — 调用 DeepSeek 将自然语言指令转为 skill 调用序列。
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from openai import OpenAI
import pandas as pd
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from core.skills import SKILL_REGISTRY


@dataclass
class PlanStep:
    skill: str
    params: dict
    reason: str = ""


def plan_to_steps(raw_text: str) -> list:
    """
    将 LLM 返回的原始文本解析为 PlanStep 列表。
    兼容带 ```json ... ``` 的 markdown 代码块格式。
    兼容响应被截断（无闭合 ```）的情况。
    """
    text = raw_text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        text = m.group(1).strip()
    else:
        # 回退：响应被截断时找第一个 { 开始解析
        brace_pos = text.find('{')
        if brace_pos >= 0:
            text = text[brace_pos:]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"无法解析 LLM 响应为 JSON：{e}\n原始内容：{raw_text[:200]}")

    steps_raw = data.get("plan", [])
    steps = []
    for item in steps_raw:
        skill_name = item.get("skill", "").strip()  # 去除 AI 可能附带的空白字符
        if skill_name not in SKILL_REGISTRY:
            raise ValueError(f"未知 skill：{skill_name!r}，可用：{list(SKILL_REGISTRY)}")
        steps.append(PlanStep(
            skill=skill_name,
            params=item.get("params", {}),
            reason=item.get("reason", ""),
        ))
    return steps


class AIPLanner:
    """
    将自然语言指令翻译为 skill 调用序列。
    使用 DeepSeek-V3（通过 SiliconFlow）。
    """

    def __init__(self):
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置，请检查 .env 文件")
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def _build_schema_desc(self, df: pd.DataFrame) -> str:
        lines = [f"- {col}（{dtype}）" for col, dtype in df.dtypes.items()]
        sample = df.head(3).to_dict("records")
        return "\n".join(lines) + f"\n\n前3行样本：\n{json.dumps(sample, ensure_ascii=False, default=str)}"

    def _build_skill_list(self) -> str:
        lines = []
        for name, meta in SKILL_REGISTRY.items():
            params_str = "，".join(f"{k}: {v}" for k, v in meta["params_schema"].items())
            lines.append(f'- **{name}**: {meta["description"]}\n  参数：{params_str}')
        return "\n".join(lines)

    def _build_prompt(self, df: pd.DataFrame, instruction: str) -> str:
        return f"""你是一个专业的 Excel 数据分析规划师。你的职责是：**把用户的自然语言指令翻译成一份精确的 skill 执行计划**。

---

## 第一步：任务分类（先判断类型，再决定用哪些 skill）

拿到指令后，先在脑中完成以下分类判断：

### A. 纯数据处理类
**识别特征**：增删改查、格式整理、无需建模
**典型指令**：筛选/查找、删除列/行、排序、去重、填充缺失值、重命名、新增计算列、聚合统计
**使用 skill**：`filter_rows` / `delete_rows` / `delete_columns` / `sort` / `deduplicate` / `fill_missing` / `rename_column` / `add_column` / `aggregate`

**⚠️ 拆分 vs 删除区分**：
- 「按N列拆分/分组/多张表」→ 用 `split_columns`（**保留所有列**，按组放到不同 Sheet）
- 「删除某列」→ 用 `delete_columns`（**永久删除**该列）
- 绝对不能用 `delete_columns` 来"拆分"数据！

### B. 可视化类
**识别特征**：用户提到"图"、"图表"、"画"、"可视化"、"展示"
**使用 skill**：先用数据处理准备数据，最后用 `build_chart`
**图表选型规则**：
- 分类对比 / 排名 → `bar`（柱状图）
- 时间趋势 / 序列变化 → `line`（折线图）
- 占比 / 构成 → `pie`（饼图）
- 相关性 / 散点分布 → `scatter`
- 累积变化 → `area`
- 量 + 率并存 → `combo`
- ⚠️ 除非用户明确说"折线图"，分组对比默认用 `bar`

**build_chart 参数规则**：
- `x`：X 轴列名（bar/line/area/scatter）；饼图中为**类别/标签列**
- `y`：Y 轴列名列表（必须非空）；饼图中 `y[0]` 为**数值列**
- **饼图必须先聚合**：原始数据行数过多时，先用 `aggregate` 生成「类别→数值」的摘要表，再画饼图
  - 示例：「Y 各值占比」→ 先 `aggregate(group_by=["Y"], agg={{"Y":"count"}})` 得到 `Y_count` 列，再 `build_chart(pie, x="Y", y=["Y_count"])`
  - ⚠️ **绝对禁止** `y=[]`，饼图 y 参数不能为空列表

### C. 统计分析 / 机器学习类
**识别特征**：涉及"影响因素"、"回归"、"模型"、"预测"、"相关性分析"、"特征重要性"、"聚类"、"降维"等
**使用 skill**：`run_analysis`（唯一入口，禁止用数据处理 skill 替代）
**mode 选择规则**：
| 用户说的 | mode |
|---------|------|
| 找影响Y的因素 / 哪些X影响Y | `y_vs_all` |
| 建立多个模型 / 比较哪个模型最好 / 线性+随机森林+神经网络 | `model_comparison` |
| 多元线性回归 / 回归系数 / p值 | `multi_x_vs_y` |
| 神经网络 / MLP / 深度学习 | `neural_reg` |
| 岭回归 / 套索 / lasso / ridge | `ridge_lasso` |
| 逻辑回归 / 分类预测 | `logistic` |
| 聚类 / 分群 / k-means | `cluster` |
| 主成分 / 降维 / PCA | `pca` |
| 方差分析 / 组间差异 / ANOVA | `anova` |
| 时间趋势 / 时序预测 | `time_series` |
| 两列相关 / 散点+相关系数 | `two_column` |

### D. 复合任务（最重要！）
**识别特征**：指令里有"然后"、"并"、"再"、"同时"、"之后"，或描述了两个以上目标
**处理方式**：**拆解为多个 skill 顺序执行**，数据沿 pipeline 传递（上一步输出是下一步输入）

---

## 第二步：复合任务拆解示例

### 示例1：先处理再画图
指令：「筛选销售额大于1000的数据，按部门汇总，画柱状图」
```json
{{
  "plan": [
    {{"skill": "filter_rows", "params": {{"condition": "销售额 > 1000"}}, "reason": "筛选高销售额数据"}},
    {{"skill": "aggregate", "params": {{"group_by": ["部门"], "agg": {{"销售额": "sum"}}}}, "reason": "按部门求和"}},
    {{"skill": "build_chart", "params": {{"chart_type": "bar", "x": "部门", "y": ["销售额_sum"], "title": "各部门销售额汇总"}}, "reason": "可视化结果"}}
  ]
}}
```

### 示例2：先分析再画图
指令：「找出影响Y的主要因素，并画出重要性排名图」
```json
{{
  "plan": [
    {{"skill": "run_analysis", "params": {{"mode": "y_vs_all", "target_y": "Y"}}, "reason": "用随机森林找特征重要性"}},
    {{"skill": "build_chart", "params": {{"chart_type": "bar", "x": "特征", "y": ["重要性"], "title": "特征重要性排名"}}, "reason": "可视化影响因素"}}
  ]
}}
```
> ⚠️ 注意：`run_analysis` 之后 df 变为分析结果表，build_chart 使用该结果的列名

### 示例3：先清洗再分析
指令：「清理数据后建立多个回归模型找最优」
```json
{{
  "plan": [
    {{"skill": "fill_missing", "params": {{"method": "mean"}}, "reason": "填充缺失值避免影响建模"}},
    {{"skill": "deduplicate", "params": {{}}, "reason": "去重"}},
    {{"skill": "run_analysis", "params": {{"mode": "model_comparison", "target_y": "Y"}}, "reason": "多模型赛马找最优"}}
  ]
}}
```

### 示例4：纯数据处理
指令：「删除年龄列，按部门统计销售额均值，按均值降序排列」
```json
{{
  "plan": [
    {{"skill": "delete_columns", "params": {{"columns": ["年龄"]}}, "reason": "删除不需要的列"}},
    {{"skill": "aggregate", "params": {{"group_by": ["部门"], "agg": {{"销售额": "mean"}}}}, "reason": "按部门求均值"}},
    {{"skill": "sort", "params": {{"column": "销售额_mean", "ascending": false}}, "reason": "按均值降序"}}
  ]
}}
```

### 示例5：只用一个 skill（简单任务不要过度拆解）
指令：「筛选Y大于10的行」
```json
{{
  "plan": [
    {{"skill": "filter_rows", "params": {{"condition": "Y > 10"}}, "reason": "筛选满足条件的行"}}
  ]
}}
```

### 示例6：按列数拆分（split_columns）
指令：「按每5列拆分数据为多张表」
```json
{{
  "plan": [
    {{"skill": "split_columns", "params": {{"n": 5}}, "reason": "把宽表按每5列一组拆分为多张子表，所有列均保留"}}
  ]
}}
```
> ⚠️ 「拆分」≠「删除」：split_columns 保留全部列，只是分组放到不同 Sheet

### 示例7：饼图（必须先聚合）
指令：「画饼图展示Y各值占比」
```json
{{
  "plan": [
    {{"skill": "aggregate", "params": {{"group_by": ["Y"], "agg": {{"Y": "count"}}}}, "reason": "统计Y各值频次"}},
    {{"skill": "build_chart", "params": {{"chart_type": "pie", "x": "Y", "y": ["Y_count"], "title": "Y各值占比"}}, "reason": "画饼图，x=类别列，y=数值列"}}
  ]
}}
```

---

## 第三步：参数约束

### 条件格式（filter_rows / delete_rows 的 condition）
只能使用以下格式，**禁止 Python/pandas 表达式**：
- `列名 > 值` / `列名 >= 值` / `列名 == 值` / `列名 != 值`
- `列名大于值` / `列名小于值` / `列名等于值`
- `列名 包含 关键词`
- `列名最大` / `列名最小` / `列名最大的N行`

禁止：`Y == Y.max()` / `df['Y'] > 0` / `lambda x: x > 0`

### aggregate 用法规则
- 生成的列名 = `原列名_聚合函数`，例如：
  - `{{"销售额": "sum"}}` → 列名变为 `销售额_sum`
  - `{{"销售额": "mean"}}` → 列名变为 `销售额_mean`
  - 后续 skill 引用该列必须用 `销售额_sum` 而不是 `销售额`
- **无分组全局聚合**：`group_by` 传 `[]`（空列表），例如「求Y的平均值」：
  ```json
  {{"skill": "aggregate", "params": {{"group_by": [], "agg": {{"Y": "mean"}}}}, "reason": "全局求均值"}}
  ```
  ⚠️ 不要为无分组需求强行添加分组列！

### run_analysis 注意事项
- `target_y` 必须是数据结构中存在的列名
- `x_cols` 可以省略（系统自动选择所有数值列）
- `run_analysis` 执行后 df 内容变为分析结果摘要表，后续 build_chart 要用摘要表的列名

---

## 数据结构
{self._build_schema_desc(df)}

## 完整 Skill 列表
{self._build_skill_list()}

---

## 用户指令
{instruction}

---

## 输出要求
只返回如下 JSON，不要任何解释或 markdown 以外的内容：
```json
{{
  "plan": [
    {{"skill": "技能名", "params": {{...}}, "reason": "这一步的用途"}},
    ...
  ]
}}
```

规则：
1. **只能使用 Skill 列表中的 skill**，不能发明新名称
2. **列名严格来自数据结构**，不能猜测或缩写
3. **复合任务必须拆解**，不要把多个操作塞进一个 skill
4. **简单任务不要过度拆解**，一步能完成的不写两步
5. **分析类任务用 run_analysis**，不要用 filter_rows 模拟"""

    def plan(self, df: pd.DataFrame, instruction: str) -> list:
        """
        主入口：接收 df 和自然语言指令，返回 PlanStep 列表。
        """
        prompt = self._build_prompt(df, instruction)
        resp = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content
        return plan_to_steps(raw)
