---
title: Excel 智能分析助手
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
python_version: "3.11"
---

# 📊 Excel 智能分析助手

基于 DeepSeek + Gradio 的智能数据分析工具，支持自然语言指令驱动的三模式分析。

## 功能

- **全因素分析**：分析所有变量对目标的影响（随机森林 + Pearson 相关）
- **双列关系**：分析两列之间的相关性与回归关系
- **多因素分析**：多元回归 + 共线性检测

## 使用方法

1. 上传 Excel 文件（.xlsx / .xls）
2. 输入自然语言分析指令，或切换到手动模式
3. 点击「开始分析」获取报告
