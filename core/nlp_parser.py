"""
DeepSeek 意图解析器
将用户自然语言指令解析为结构化分析参数
"""
import json
import re
from pathlib import Path
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, PROMPTS_DIR


class IntentParser:
    def __init__(self):
        if not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY 未配置，请检查 .env 文件")
        self._client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self._system_prompt_template = (PROMPTS_DIR / "intent_parser.txt").read_text(encoding="utf-8")

    def parse(self, user_instruction: str, available_columns: list[str]) -> dict:
        """
        解析用户指令，返回结构化参数字典。

        Returns:
            {
                "analysis_mode": str,      # y_vs_all / two_column / multi_x_vs_y
                "target_y": str | None,
                "x_columns": list[str],
                "analysis_hint": str,
                "confidence": float,
                "error": str | None,       # 解析失败时的错误信息
            }
        """
        cols_str = "\n".join(f"- {c}" for c in available_columns)
        system_prompt = self._system_prompt_template.replace("{available_columns}", cols_str)

        for attempt in range(2):
            try:
                response = self._client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_instruction},
                    ],
                    temperature=0.1,
                    max_tokens=512,
                )
                raw = response.choices[0].message.content.strip()
                parsed = self._extract_json(raw)
                validated = self._validate(parsed, available_columns)
                return validated

            except json.JSONDecodeError as e:
                if attempt == 1:
                    return self._fallback_result(f"JSON 解析失败：{e}")
            except Exception as e:
                return self._fallback_result(f"API 调用失败：{e}")

        return self._fallback_result("重试后仍解析失败")

    def _extract_json(self, text: str) -> dict:
        """从文本中提取 JSON，兼容 ```json...``` 格式"""
        # 去除 markdown 代码块
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        return json.loads(text)

    def _validate(self, parsed: dict, available_columns: list[str]) -> dict:
        """验证并修正解析结果，确保列名都在 available_columns 中"""
        col_set = set(available_columns)

        # 验证 target_y
        target_y = parsed.get("target_y")
        if target_y not in col_set:
            target_y = self._fuzzy_match(target_y, available_columns)
        parsed["target_y"] = target_y

        # 验证 x_columns
        x_cols = parsed.get("x_columns", [])
        validated_x = []
        for col in x_cols:
            if col in col_set:
                validated_x.append(col)
            else:
                matched = self._fuzzy_match(col, available_columns)
                if matched:
                    validated_x.append(matched)
        parsed["x_columns"] = validated_x

        # 确保必要字段存在
        parsed.setdefault("analysis_mode", "y_vs_all")
        parsed.setdefault("analysis_hint", "")
        parsed.setdefault("confidence", 0.5)
        parsed["error"] = None

        return parsed

    def _fuzzy_match(self, name: str, candidates: list[str]) -> str | None:
        """简单模糊匹配：找包含关系最强的候选列名"""
        if not name:
            return None
        name_lower = name.lower()
        for c in candidates:
            if name_lower in c.lower() or c.lower() in name_lower:
                return c
        return None

    @staticmethod
    def _fallback_result(error_msg: str) -> dict:
        return {
            "analysis_mode": None,
            "target_y": None,
            "x_columns": [],
            "analysis_hint": "",
            "confidence": 0.0,
            "error": error_msg,
        }
