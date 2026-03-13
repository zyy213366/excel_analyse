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
    """
    text = raw_text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"无法解析 LLM 响应为 JSON：{e}\n原始内容：{raw_text[:200]}")

    steps_raw = data.get("plan", [])
    steps = []
    for item in steps_raw:
        skill_name = item.get("skill", "")
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
        return f"""你是一个数据处理助手。用户有一份 Excel 数据，结构如下：

## 数据结构
{self._build_schema_desc(df)}

## 可用操作（Skills）
{self._build_skill_list()}

## 用户指令
{instruction}

## 要求
请将用户指令分解为一个或多个 skill 调用序列，返回严格的 JSON 格式：

```json
{{
  "plan": [
    {{"skill": "技能名", "params": {{...}}, "reason": "这一步的用途"}},
    ...
  ]
}}
```

注意：
1. 只能使用上面列出的 skill，不能发明新 skill
2. params 中的列名必须严格使用数据结构中存在的列名
3. 如果需要先聚合再画图，聚合后的列名格式为 "原列名_聚合函数"，如 "销售额_sum"
4. 只返回 JSON，不要额外说明"""

    def plan(self, df: pd.DataFrame, instruction: str) -> list:
        """
        主入口：接收 df 和自然语言指令，返回 PlanStep 列表。
        """
        prompt = self._build_prompt(df, instruction)
        resp = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content
        return plan_to_steps(raw)
