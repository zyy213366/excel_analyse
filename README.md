---
license: mit
language:
- zh
tags:
- excel
- data-analysis
- fastapi
sdk: docker
app_port: 7860
---

# 📊 Excel 智能分析助手

基于 DeepSeek + FastAPI 的智能数据分析工具，支持自然语言指令驱动的三模式分析。

## 功能

- **全因素分析**：分析所有变量对目标的影响（随机森林 + Pearson 相关）
- **双列关系**：分析两列之间的相关性与回归关系
- **多因素回归**：多元回归 + 共线性检测

## 使用方法

1. 上传 Excel / CSV 文件
2. 输入自然语言分析指令，或切换到手动模式
3. 点击「发送」获取分析报告并下载 Excel 图表

## 环境变量

| 变量名 | 说明 |
|--------|------|
| `DEEPSEEK_API_KEY` | API Key |
| `DEEPSEEK_BASE_URL` | API 地址（默认 siliconflow）|
| `DEEPSEEK_MODEL` | 模型名（默认 DeepSeek-V3）|
