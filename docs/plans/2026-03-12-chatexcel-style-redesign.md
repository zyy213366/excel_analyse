# ChatExcel 风格前端重设计方案

**日期**：2026-03-12
**状态**：已批准
**作者**：蕾姆 × 昴君

---

## 背景

现有项目使用 Gradio 构建界面，无法实现真正的侧边栏导航和聊天流交互。目标是复刻 ChatExcel Pro 的应用风格，同时保留现有 Python 分析引擎不变，并确保可顺利部署至 ModelScope 创空间。

---

## 决策

| 问题 | 决策 |
|------|------|
| 技术栈 | FastAPI + Jinja2 + Alpine.js（CDN，无构建步骤） |
| 交互模式 | 混合式：左侧固定文件上传，右侧多轮聊天流 |
| 页面范围 | 4页：工作台 / 智能分析 / 历史记录 / 设置 |
| 配色 | ChatExcel 绿（深绿侧边栏 + 薄荷绿卡片）|
| 部署 | ModelScope Docker 空间，追加 Dockerfile |

---

## 技术架构

```
浏览器（HTML + CSS + Alpine.js CDN）
        ↕ HTTP / JSON / multipart
FastAPI + uvicorn（替换 Gradio）
        ↕ 函数调用
core/ + utils/（分析引擎，完全不变）
```

**新增依赖**：`fastapi`, `uvicorn`, `jinja2`, `python-multipart`
**移除依赖**：`gradio`

---

## 项目结构

```
excel-analyzer/
├── app.py                   ← 改为 FastAPI 应用入口
├── api/
│   ├── __init__.py
│   └── routes.py            ← 所有 REST 端点
├── static/
│   ├── css/
│   │   └── main.css         ← ChatExcel 绿配色 + 布局
│   └── js/
│       └── app.js           ← Alpine.js 交互逻辑
├── templates/
│   ├── base.html            ← 侧边栏骨架布局
│   ├── home.html            ← 工作台首页
│   ├── analyze.html         ← 智能分析工作台
│   ├── history.html         ← 历史记录
│   └── settings.html        ← API 设置
├── Dockerfile               ← ModelScope 部署
├── core/ utils/ prompts/    ← 完全不变
└── outputs/                 ← 完全不变
```

---

## 配色系统

| 用途 | 色值 |
|------|------|
| 侧边栏背景 | `#1a3c34` |
| 侧边栏文字 | `#a8c5bb` |
| 侧边栏选中项 | `#2d6a5a` |
| 主内容背景 | `#f5f7f6` |
| 功能卡片背景 | `#e8f5f0` |
| 功能卡片边框 | `#b2d8cc` |
| 强调色/主按钮 | `#2a7d5f` |
| 文字主色 | `#1e293b` |
| 文字次色 | `#64748b` |

---

## 页面设计

### 工作台首页（home.html）
- Banner：吉祥物图 + 标语
- 3个分析模式能力卡片（绿色卡片，含功能列表按钮）
- 行业示例标签云（点击自动跳转分析页并填入指令）

### 智能分析工作台（analyze.html）
- 左栏（30%）：文件上传区 + 已上传文件信息 + 检测到的列 + AI模式开关
- 右栏（70%）：多轮聊天气泡流 + 固定底部输入框
- 气泡类型：用户指令气泡（右对齐）+ AI结论气泡（左对齐，含下载按钮）

### 历史记录（history.html）
- 读取 outputs/ 目录，展示报告列表
- 每行：文件名、分析模式、时间、下载按钮

### 设置（settings.html）
- 表单：API Key、Base URL、模型名
- 保存后写回 .env 文件

---

## API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 工作台首页 |
| `/analyze` | GET | 分析工作台 |
| `/history` | GET | 历史记录 |
| `/settings` | GET/POST | 设置页 |
| `/api/upload` | POST | 上传 Excel，返回列名 |
| `/api/analyze` | POST | 执行分析，返回 JSON |
| `/api/history` | GET | 报告列表 JSON |
| `/api/download/{filename}` | GET | 下载报告 |
| `/api/settings` | GET/POST | 读写配置 |

---

## ModelScope 部署方案

追加 `Dockerfile`，app.py 读取 `PORT` 环境变量（已有 `is_cloud` 检测）。
部署流程与现有 Gradio 版本完全一致：推送代码 → 自动构建。
