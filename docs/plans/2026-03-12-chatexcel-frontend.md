# ChatExcel 风格前端重设计 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** 将现有 Gradio 界面替换为 FastAPI + 纯 HTML/CSS/Alpine.js，复刻 ChatExcel Pro 风格（深绿侧边栏 + 功能卡片首页 + 多轮聊天分析工作台），保持 core/ utils/ 分析引擎完全不变。

**Architecture:** FastAPI 提供页面路由和 REST API，Jinja2 渲染 HTML 模板，Alpine.js 处理前端状态（文件上传、聊天消息流），marked.js 渲染 Markdown 气泡内容。

**Tech Stack:** FastAPI, uvicorn, jinja2, python-multipart, Alpine.js 3.x (CDN), marked.js (CDN)

---

## Task 1：依赖安装 + FastAPI 基础框架

**Files:**
- Modify: `requirements.txt`
- Modify: `app.py`（全量重写）
- Create: `api/__init__.py`

**Step 1: 更新 requirements.txt**

将以下内容完整替换 requirements.txt：

```
pandas>=2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
matplotlib>=3.7
openpyxl>=3.1
xlsxwriter>=3.1
python-dotenv>=1.0
openai>=1.0
fastapi>=0.110
uvicorn[standard]>=0.27
jinja2>=3.1
python-multipart>=0.0.9
```

**Step 2: 安装新依赖**

```bash
cd C:/Users/29571/PycharmProjects/excel-analyzer
pip install fastapi "uvicorn[standard]" jinja2 python-multipart
```

期望输出：`Successfully installed fastapi-... uvicorn-... jinja2-... python-multipart-...`（或 already satisfied）

**Step 3: 创建 api/__init__.py**

```python
```
（空文件）

**Step 4: 重写 app.py**

```python
"""
Excel 智能分析助手 - FastAPI 入口
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from api.routes import router
from config import OUTPUTS_DIR

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Excel 智能分析助手", version="2.0")

# 静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# 路由
app.include_router(router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "8000")))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
```

**Step 5: 验证 FastAPI 启动**

先临时创建空路由文件使其可启动：

```python
# api/routes.py（临时占位）
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/")
async def home():
    return HTMLResponse("<h1>Excel 分析助手 - 建设中</h1>")
```

运行：`python app.py`
期望输出：`Uvicorn running on http://0.0.0.0:8000`
浏览器打开 `http://localhost:8000` 看到"建设中"文字。

**Step 6: Commit**

```bash
git add requirements.txt app.py api/__init__.py api/routes.py
git commit -m "feat: 替换 Gradio 为 FastAPI 基础框架"
```

---

## Task 2：目录结构 + CSS 主题系统

**Files:**
- Create: `static/css/main.css`
- Create: `static/js/app.js`（占位）
- Create: `static/img/`（目录）

**Step 1: 创建目录**

```bash
mkdir -p static/css static/js static/img
```

**Step 2: 创建 static/css/main.css**

```css
/* ═══════════════════════════════════════
   CSS 变量系统 — ChatExcel 绿色主题
═══════════════════════════════════════ */
:root {
  --sidebar-bg:      #1a3c34;
  --sidebar-text:    #a8c5bb;
  --sidebar-active:  #2d6a5a;
  --sidebar-hover:   #224d40;
  --sidebar-border:  #2d6a5a;
  --sidebar-width:   220px;

  --main-bg:         #f5f7f6;
  --card-bg:         #e8f5f0;
  --card-border:     #b2d8cc;
  --card-hover:      #d4ede6;

  --accent:          #2a7d5f;
  --accent-hover:    #1f5f47;
  --accent-light:    #e0f0eb;

  --text-primary:    #1e293b;
  --text-secondary:  #64748b;
  --text-muted:      #94a3b8;

  --white:           #ffffff;
  --border:          #e2e8f0;
  --shadow-sm:       0 1px 3px rgba(0,0,0,0.08);
  --shadow-md:       0 4px 12px rgba(0,0,0,0.10);
  --radius-sm:       6px;
  --radius-md:       10px;
  --radius-lg:       14px;
}

/* ═══════════════════════════════════════
   Reset & Base
═══════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; font-size: 14px; color: var(--text-primary); background: var(--main-bg); }
a { text-decoration: none; color: inherit; }
button { cursor: pointer; border: none; background: none; font-family: inherit; font-size: inherit; }

/* ═══════════════════════════════════════
   应用布局
═══════════════════════════════════════ */
.app-layout {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

/* ═══════════════════════════════════════
   侧边栏
═══════════════════════════════════════ */
.sidebar {
  width: var(--sidebar-width);
  min-width: var(--sidebar-width);
  background: var(--sidebar-bg);
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  overflow-x: hidden;
}

.sidebar-logo {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 20px 16px 16px;
  border-bottom: 1px solid var(--sidebar-border);
}
.sidebar-logo-icon {
  width: 32px; height: 32px;
  background: var(--accent);
  border-radius: var(--radius-sm);
  display: flex; align-items: center; justify-content: center;
  font-size: 18px;
}
.sidebar-logo-text { color: var(--white); font-size: 15px; font-weight: 700; }
.sidebar-logo-sub  { color: var(--sidebar-text); font-size: 11px; margin-top: 1px; }

.sidebar-section { padding: 12px 8px 4px; }
.sidebar-section-label {
  color: var(--sidebar-text);
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 0 8px 6px;
  opacity: 0.6;
}

.sidebar-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 12px;
  border-radius: var(--radius-sm);
  color: var(--sidebar-text);
  font-size: 13.5px;
  transition: background 0.15s, color 0.15s;
  margin: 1px 0;
  cursor: pointer;
}
.sidebar-item:hover   { background: var(--sidebar-hover); color: var(--white); }
.sidebar-item.active  { background: var(--sidebar-active); color: var(--white); font-weight: 600; }
.sidebar-item .icon   { font-size: 16px; width: 20px; text-align: center; flex-shrink: 0; }
.sidebar-item .badge  {
  margin-left: auto;
  font-size: 10px; font-weight: 700;
  padding: 1px 6px; border-radius: 10px;
  background: var(--accent); color: var(--white);
}

/* ═══════════════════════════════════════
   主内容区
═══════════════════════════════════════ */
.main-content {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.main-header {
  background: var(--white);
  border-bottom: 1px solid var(--border);
  padding: 14px 24px;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}
.main-header-title { font-size: 15px; font-weight: 700; color: var(--text-primary); }
.main-header-sub   { font-size: 12px; color: var(--text-muted); margin-top: 1px; }

.main-body { padding: 24px; flex: 1; }

/* ═══════════════════════════════════════
   首页 Banner
═══════════════════════════════════════ */
.home-banner {
  background: var(--white);
  border-radius: var(--radius-lg);
  padding: 28px 32px;
  display: flex;
  align-items: center;
  gap: 24px;
  margin-bottom: 28px;
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--border);
}
.home-banner-mascot { font-size: 64px; flex-shrink: 0; }
.home-banner-title  { font-size: 22px; font-weight: 700; color: var(--text-primary); }
.home-banner-sub    { font-size: 14px; color: var(--text-secondary); margin-top: 6px; }

/* ═══════════════════════════════════════
   能力演示卡片
═══════════════════════════════════════ */
.section-title {
  font-size: 15px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 14px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--card-border);
}

.cards-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 28px;
}

.feature-card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: var(--radius-md);
  padding: 16px;
  transition: box-shadow 0.2s, transform 0.2s;
}
.feature-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }

.feature-card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
  font-weight: 700;
  color: var(--accent);
  font-size: 14px;
}
.feature-card-icon { font-size: 18px; }

.feature-btn {
  display: block;
  width: 100%;
  padding: 7px 12px;
  margin-bottom: 6px;
  background: var(--white);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text-primary);
  font-size: 13px;
  text-align: center;
  transition: background 0.15s, border-color 0.15s, color 0.15s;
}
.feature-btn:hover {
  background: var(--accent);
  border-color: var(--accent);
  color: var(--white);
}

/* ═══════════════════════════════════════
   标签云
═══════════════════════════════════════ */
.tags-cloud {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 24px;
}
.tag-btn {
  padding: 6px 14px;
  border: 1px solid var(--card-border);
  border-radius: 20px;
  background: var(--white);
  color: var(--text-secondary);
  font-size: 13px;
  transition: all 0.15s;
}
.tag-btn:hover {
  background: var(--accent);
  border-color: var(--accent);
  color: var(--white);
}

/* ═══════════════════════════════════════
   分析工作台布局
═══════════════════════════════════════ */
.analyze-layout {
  display: flex;
  gap: 0;
  height: calc(100vh - 57px); /* 减去 header 高度 */
  overflow: hidden;
}

/* 左侧文件面板 */
.file-panel {
  width: 300px;
  min-width: 300px;
  background: var(--white);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  padding: 20px;
}
.file-panel-title {
  font-size: 13px;
  font-weight: 700;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 12px;
}

/* 文件上传区 */
.upload-zone {
  border: 2px dashed var(--card-border);
  border-radius: var(--radius-md);
  padding: 24px 16px;
  text-align: center;
  background: var(--card-bg);
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
  margin-bottom: 16px;
}
.upload-zone:hover, .upload-zone.drag-over {
  border-color: var(--accent);
  background: var(--accent-light);
}
.upload-zone-icon { font-size: 32px; margin-bottom: 8px; }
.upload-zone-text { font-size: 13px; color: var(--text-secondary); }
.upload-zone-sub  { font-size: 11px; color: var(--text-muted); margin-top: 4px; }

/* 文件信息 */
.file-info {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: var(--radius-sm);
  padding: 10px 12px;
  margin-bottom: 14px;
  font-size: 13px;
}
.file-info-name { font-weight: 600; color: var(--accent); word-break: break-all; }
.file-info-meta { color: var(--text-muted); font-size: 11px; margin-top: 3px; }

/* 列名展示 */
.columns-section { margin-bottom: 16px; }
.columns-label { font-size: 12px; color: var(--text-muted); margin-bottom: 6px; }
.columns-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.column-chip {
  padding: 2px 8px;
  background: var(--accent-light);
  border: 1px solid var(--card-border);
  border-radius: 10px;
  font-size: 11px;
  color: var(--accent);
  font-family: monospace;
}

/* AI 模式开关 */
.ai-toggle {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  background: var(--card-bg);
  border-radius: var(--radius-sm);
  border: 1px solid var(--card-border);
  font-size: 13px;
  color: var(--text-primary);
}
.toggle-switch {
  position: relative;
  width: 36px; height: 20px;
}
.toggle-switch input { opacity: 0; width: 0; height: 0; }
.toggle-slider {
  position: absolute; inset: 0;
  background: #ccc; border-radius: 20px;
  cursor: pointer; transition: background 0.2s;
}
.toggle-slider::before {
  content: '';
  position: absolute;
  width: 14px; height: 14px;
  left: 3px; top: 3px;
  background: white; border-radius: 50%;
  transition: transform 0.2s;
}
.toggle-switch input:checked + .toggle-slider { background: var(--accent); }
.toggle-switch input:checked + .toggle-slider::before { transform: translateX(16px); }

/* ═══════════════════════════════════════
   聊天流区域
═══════════════════════════════════════ */
.chat-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--main-bg);
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* 空状态 */
.chat-empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  gap: 10px;
}
.chat-empty-icon { font-size: 48px; opacity: 0.4; }
.chat-empty-text { font-size: 14px; }

/* 气泡 */
.message-user {
  align-self: flex-end;
  max-width: 60%;
  background: var(--accent);
  color: var(--white);
  padding: 10px 16px;
  border-radius: var(--radius-md) var(--radius-md) 4px var(--radius-md);
  font-size: 14px;
  line-height: 1.5;
  box-shadow: var(--shadow-sm);
}

.message-assistant {
  align-self: flex-start;
  max-width: 80%;
  background: var(--white);
  border: 1px solid var(--border);
  padding: 16px 18px;
  border-radius: var(--radius-md) var(--radius-md) var(--radius-md) 4px;
  box-shadow: var(--shadow-sm);
}

/* Intent 解析标签 */
.intent-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  background: var(--accent-light);
  border: 1px solid var(--card-border);
  border-radius: 12px;
  font-size: 12px;
  color: var(--accent);
  font-weight: 600;
  margin-bottom: 10px;
}

/* Markdown 内容 */
.md-content { font-size: 13.5px; line-height: 1.7; color: var(--text-primary); }
.md-content h2 { font-size: 15px; font-weight: 700; margin: 0 0 8px; color: var(--accent); }
.md-content h3 { font-size: 13.5px; font-weight: 700; margin: 12px 0 6px; }
.md-content p  { margin: 6px 0; }
.md-content ul { padding-left: 18px; margin: 6px 0; }
.md-content li { margin: 3px 0; }
.md-content table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 12.5px; }
.md-content th { background: var(--card-bg); padding: 6px 10px; border: 1px solid var(--border); font-weight: 600; }
.md-content td { padding: 5px 10px; border: 1px solid var(--border); }
.md-content code { background: var(--card-bg); padding: 1px 5px; border-radius: 3px; font-size: 12px; color: var(--accent); font-family: monospace; }
.md-content strong { font-weight: 700; }

/* 下载按钮 */
.download-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-top: 12px;
  padding: 8px 16px;
  background: var(--accent);
  color: var(--white);
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 600;
  transition: background 0.15s;
}
.download-btn:hover { background: var(--accent-hover); }

/* 加载中 */
.typing-indicator {
  align-self: flex-start;
  background: var(--white);
  border: 1px solid var(--border);
  padding: 12px 18px;
  border-radius: var(--radius-md) var(--radius-md) var(--radius-md) 4px;
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-secondary);
  font-size: 13px;
}
.dots span {
  display: inline-block;
  width: 6px; height: 6px;
  background: var(--text-muted);
  border-radius: 50%;
  animation: bounce 1.2s infinite;
}
.dots span:nth-child(2) { animation-delay: 0.2s; }
.dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-6px)} }

/* ═══════════════════════════════════════
   输入区
═══════════════════════════════════════ */
.chat-input-area {
  background: var(--white);
  border-top: 1px solid var(--border);
  padding: 14px 20px;
  display: flex;
  align-items: flex-end;
  gap: 10px;
}
.chat-input {
  flex: 1;
  padding: 10px 14px;
  border: 1.5px solid var(--border);
  border-radius: var(--radius-md);
  font-size: 14px;
  font-family: inherit;
  resize: none;
  outline: none;
  transition: border-color 0.2s;
  max-height: 120px;
  line-height: 1.5;
}
.chat-input:focus  { border-color: var(--accent); }
.chat-input:disabled { background: var(--main-bg); }

.send-btn {
  padding: 10px 18px;
  background: var(--accent);
  color: var(--white);
  border-radius: var(--radius-sm);
  font-size: 14px;
  font-weight: 600;
  transition: background 0.15s;
  flex-shrink: 0;
  height: 42px;
}
.send-btn:hover:not(:disabled)   { background: var(--accent-hover); }
.send-btn:disabled { background: var(--text-muted); cursor: not-allowed; }

/* ═══════════════════════════════════════
   历史记录
═══════════════════════════════════════ */
.history-table {
  width: 100%;
  background: var(--white);
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}
.history-table th {
  background: var(--card-bg);
  padding: 12px 16px;
  text-align: left;
  font-size: 12px;
  font-weight: 700;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border-bottom: 1px solid var(--border);
}
.history-table td {
  padding: 12px 16px;
  font-size: 13px;
  border-bottom: 1px solid var(--border);
  color: var(--text-primary);
}
.history-table tr:last-child td { border-bottom: none; }
.history-table tr:hover td { background: var(--card-bg); }

.mode-badge {
  display: inline-flex;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 600;
  background: var(--accent-light);
  color: var(--accent);
  border: 1px solid var(--card-border);
}

/* ═══════════════════════════════════════
   设置页
═══════════════════════════════════════ */
.settings-card {
  background: var(--white);
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  padding: 28px 32px;
  max-width: 560px;
  box-shadow: var(--shadow-sm);
}
.settings-card h2 { font-size: 15px; font-weight: 700; margin-bottom: 20px; color: var(--text-primary); }
.form-group { margin-bottom: 18px; }
.form-label { display: block; font-size: 13px; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px; }
.form-input {
  width: 100%;
  padding: 9px 12px;
  border: 1.5px solid var(--border);
  border-radius: var(--radius-sm);
  font-size: 14px;
  font-family: inherit;
  outline: none;
  transition: border-color 0.2s;
}
.form-input:focus { border-color: var(--accent); }
.form-hint { font-size: 11px; color: var(--text-muted); margin-top: 4px; }
.save-btn {
  padding: 10px 28px;
  background: var(--accent);
  color: var(--white);
  border-radius: var(--radius-sm);
  font-size: 14px;
  font-weight: 600;
  transition: background 0.15s;
}
.save-btn:hover { background: var(--accent-hover); }
.save-success {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--accent);
  font-size: 13px;
  margin-left: 12px;
}

/* ═══════════════════════════════════════
   滚动条美化
═══════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
```

**Step 3: 创建 static/js/app.js（占位）**

```javascript
// Alpine.js 应用逻辑 - 将在 Task 6 中完善
console.log('Excel 智能分析助手 v2.0 已加载');
```

**Step 4: Commit**

```bash
git add static/css/main.css static/js/app.js
git commit -m "feat: 添加 ChatExcel 绿色主题 CSS 系统"
```

---

## Task 3：侧边栏骨架布局（base.html）

**Files:**
- Create: `templates/base.html`

**Step 1: 创建 templates/base.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{% block title %}Excel 智能分析助手{% endblock %}</title>
  <link rel="stylesheet" href="/static/css/main.css">
  <!-- Alpine.js -->
  <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
  <!-- marked.js (Markdown 渲染) -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  {% block head %}{% endblock %}
</head>
<body>
<div class="app-layout">

  <!-- ── 侧边栏 ── -->
  <aside class="sidebar">
    <!-- Logo -->
    <div class="sidebar-logo">
      <div class="sidebar-logo-icon">📊</div>
      <div>
        <div class="sidebar-logo-text">Excel 分析助手</div>
        <div class="sidebar-logo-sub">AI 驱动的数据洞察</div>
      </div>
    </div>

    <!-- 主导航 -->
    <div class="sidebar-section">
      <a href="/" class="sidebar-item {% if active_page == 'home' %}active{% endif %}">
        <span class="icon">🏠</span> 工作台
      </a>
      <a href="/analyze" class="sidebar-item {% if active_page == 'analyze' %}active{% endif %}">
        <span class="icon">🔍</span> 智能分析
        <span class="badge">NEW</span>
      </a>
      <a href="/history" class="sidebar-item {% if active_page == 'history' %}active{% endif %}">
        <span class="icon">📋</span> 历史记录
      </a>
    </div>

    <!-- 工具箱 -->
    <div class="sidebar-section">
      <div class="sidebar-section-label">工具箱</div>
      <a href="/settings" class="sidebar-item {% if active_page == 'settings' %}active{% endif %}">
        <span class="icon">⚙️</span> API 设置
      </a>
    </div>

    <!-- 资源中心 -->
    <div class="sidebar-section" style="margin-top: auto; padding-bottom: 16px;">
      <div class="sidebar-section-label">分析模式</div>
      <div class="sidebar-item" style="cursor: default; font-size: 12px; opacity: 0.8;">
        <span class="icon">📈</span> 全因素分析
      </div>
      <div class="sidebar-item" style="cursor: default; font-size: 12px; opacity: 0.8;">
        <span class="icon">🔗</span> 双列关系
      </div>
      <div class="sidebar-item" style="cursor: default; font-size: 12px; opacity: 0.8;">
        <span class="icon">📐</span> 多因素回归
      </div>
    </div>
  </aside>

  <!-- ── 主内容区 ── -->
  <div class="main-content">
    <!-- 顶栏 -->
    <header class="main-header">
      <div>
        <div class="main-header-title">{% block header_title %}工作台{% endblock %}</div>
        <div class="main-header-sub">{% block header_sub %}Excel 智能分析助手{% endblock %}</div>
      </div>
    </header>

    <!-- 页面内容 -->
    {% block content %}{% endblock %}
  </div>

</div>
<script src="/static/js/app.js"></script>
{% block scripts %}{% endblock %}
</body>
</html>
```

**Step 2: 更新 api/routes.py，添加页面路由**

```python
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {
        "request": request,
        "active_page": "home"
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
```

**Step 3: 创建临时 templates/home.html 验证布局**

```html
{% extends "base.html" %}
{% block title %}工作台 - Excel 智能分析助手{% endblock %}
{% block header_title %}工作台{% endblock %}
{% block content %}
<div class="main-body">
  <p style="color: var(--text-secondary);">首页建设中...</p>
</div>
{% endblock %}
```

同样创建 `templates/analyze.html`, `templates/history.html`, `templates/settings.html`（都继承 base.html，内容先空着）。

**Step 4: 启动验证**

```bash
python app.py
```

打开 `http://localhost:8000`，期望：看到深绿色侧边栏 + 顶栏。
点击侧边栏各项，URL 正确跳转，active 高亮正确。

**Step 5: Commit**

```bash
git add templates/ api/routes.py
git commit -m "feat: 添加侧边栏骨架布局和四个页面路由"
```

---

## Task 4：API 路由层（upload + analyze）

**Files:**
- Modify: `api/routes.py`（追加 API 端点）

**Step 1: 在 api/routes.py 顶部追加 import**

```python
import uuid
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_excel, get_numeric_columns, preprocess_for_analysis
from utils.file_manager import get_output_path, cleanup_old_reports
from core.analysis_engine import analyze_y_vs_all, analyze_two_column, analyze_multi_x_vs_y
from core.report_builder import build_report
from config import OUTPUTS_DIR

# 内存中存储上传文件路径（fileId → 临时文件路径）
_uploaded_files: dict[str, Path] = {}
UPLOADS_DIR = OUTPUTS_DIR.parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
```

**Step 2: 添加 /api/upload 端点**

```python
@router.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """接收 Excel 文件，返回 file_id 和列名列表"""
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(400, "仅支持 .xlsx 或 .xls 格式")

    file_id = str(uuid.uuid4())
    save_path = UPLOADS_DIR / f"{file_id}_{file.filename}"

    content = await file.read()
    save_path.write_bytes(content)
    _uploaded_files[file_id] = save_path

    try:
        df, all_cols = load_excel(str(save_path))
        numeric_cols = get_numeric_columns(df)
        row_count = len(df)
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
```

**Step 3: 添加 AnalyzeRequest 模型 + /api/analyze 端点**

```python
class AnalyzeRequest(BaseModel):
    file_id: str
    instruction: str = ""
    use_ai: bool = True
    manual_mode: str = "y_vs_all"
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
            {"指标": "Pearson r", "值": f"{analysis.pearson_r:.4f}", "显著性": "p<0.001" if analysis.pearson_p < 0.001 else f"p={analysis.pearson_p:.3f}"},
            {"指标": "Spearman ρ", "值": f"{analysis.spearman_r:.4f}", "显著性": "p<0.001" if analysis.spearman_p < 0.001 else f"p={analysis.spearman_p:.3f}"},
        ]
    elif analysis.mode == "multi_x_vs_y" and analysis.multi_reg_coef_df is not None:
        for _, r in analysis.multi_reg_coef_df.iterrows():
            rows.append({
                "特征": r["Feature"],
                "系数": f"{r['Coefficient']:.4f}",
                "P值": f"{r['p_value']:.4f}" if r['p_value'] == r['p_value'] else "N/A",
                "显著性": r.get("Significant", ""),
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
            intent_info = {
                "mode": mode, "target_y": target_y, "x_cols": x_cols,
                "hint": result.get("analysis_hint", ""),
                "confidence": result.get("confidence", 0.5),
            }
        except Exception as e:
            return JSONResponse({"success": False, "error": f"AI 解析失败：{str(e)}"})
    else:
        mode = req.manual_mode
        target_y = req.manual_y
        x_cols = [c for c in req.manual_x_cols if c != target_y]

    if not target_y or target_y not in numeric_cols:
        return JSONResponse({"success": False, "error": f"目标变量 `{target_y}` 不存在"})

    # 数据预处理
    try:
        if mode == "y_vs_all":
            cols_needed = numeric_cols
        elif mode == "two_column":
            if not x_cols:
                return JSONResponse({"success": False, "error": "two_column 模式需要指定第二列"})
            cols_needed = [target_y, x_cols[0]]
        else:
            if not x_cols:
                return JSONResponse({"success": False, "error": "multi_x_vs_y 需要指定自变量列"})
            cols_needed = [target_y] + x_cols

        clean_df, raw_count, valid_count = preprocess_for_analysis(df_raw, cols_needed)
        if valid_count < 10:
            return JSONResponse({"success": False, "error": f"有效数据不足（{valid_count} 行）"})
    except Exception as e:
        return JSONResponse({"success": False, "error": f"数据预处理失败：{str(e)}"})

    # 执行分析
    try:
        if mode == "y_vs_all":
            analysis = analyze_y_vs_all(clean_df, target_y)
        elif mode == "two_column":
            analysis = analyze_two_column(clean_df, x_cols[0], target_y)
        else:
            valid_x = [c for c in x_cols if c in clean_df.columns]
            analysis = analyze_multi_x_vs_y(clean_df, target_y, valid_x)
        analysis.raw_row_count = raw_count
        analysis.valid_row_count = valid_count
    except Exception as e:
        return JSONResponse({"success": False, "error": f"分析失败：{str(e)}"})

    # 生成报告
    try:
        out_path = get_output_path(file_path.name, "分析报告")
        build_report(analysis, out_path)
        cleanup_old_reports()
        report_filename = out_path.name
    except Exception as e:
        report_filename = None

    return {
        "success": True,
        "summary_text": analysis.summary_text,
        "table_data": _build_table_data(analysis),
        "report_filename": report_filename,
        "intent": intent_info,
        "data_info": {"raw": raw_count, "valid": valid_count},
    }
```

**Step 4: 添加下载 + 历史 + 设置 API 端点**

```python
@router.get("/api/download/{filename}")
async def api_download(filename: str):
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "文件不存在")
    return FileResponse(str(file_path), filename=filename,
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@router.get("/api/history")
async def api_history():
    files = sorted(OUTPUTS_DIR.glob("*.xlsx"), key=lambda f: f.stat().st_mtime, reverse=True)
    result = []
    for f in files[:50]:
        name = f.stem
        # 从文件名提取分析模式（名称中含 "分析报告" 的文件）
        mode = "未知"
        if "分析报告" in name:
            mode = "分析报告"
        result.append({
            "filename": f.name,
            "display_name": name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "modified": f.stat().st_mtime,
        })
    return result


from dotenv import dotenv_values, set_key
from pathlib import Path as _P

ENV_PATH = _P(__file__).parent.parent / ".env"

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
```

**Step 5: 安装 python-dotenv set_key 所需版本**

```bash
pip install "python-dotenv>=1.0"
```

**Step 6: 测试 API 端点**

```bash
python app.py  # 启动
# 另一终端：
curl -X POST http://localhost:8000/api/upload \
  -F "file=@C:/Users/29571/PycharmProjects/贾维斯/test_data.xlsx"
```

期望：返回包含 `file_id` 和 `numeric_columns` 的 JSON。

**Step 7: Commit**

```bash
git add api/routes.py
git commit -m "feat: 添加 upload/analyze/history/settings API 端点"
```

---

## Task 5：工作台首页（home.html）

**Files:**
- Modify: `templates/home.html`（完整实现）

**Step 1: 完整实现 templates/home.html**

```html
{% extends "base.html" %}
{% block title %}工作台 - Excel 智能分析助手{% endblock %}
{% block header_title %}工作台{% endblock %}
{% block header_sub %}在开始之前，可以拿例子练练手哦{% endblock %}

{% block content %}
<div class="main-body">

  <!-- Banner -->
  <div class="home-banner">
    <div class="home-banner-mascot">🤖</div>
    <div>
      <div class="home-banner-title">在开始之前，可以拿例子练练手哦</div>
      <div class="home-banner-sub">上传你的 Excel 文件，用自然语言描述分析需求，AI 帮你完成统计分析并生成专业报告</div>
    </div>
    <a href="/analyze" style="margin-left:auto; padding:10px 24px; background:var(--accent); color:white; border-radius:var(--radius-sm); font-weight:700; white-space:nowrap; flex-shrink:0;">
      开始分析 →
    </a>
  </div>

  <!-- 官方能力演示 -->
  <div class="section-title">📋 官方能力演示</div>
  <div class="cards-grid">

    <!-- 卡片1：全因素分析 -->
    <div class="feature-card">
      <div class="feature-card-header">
        <span class="feature-card-icon">📊</span> 全因素影响分析
      </div>
      <a href="/analyze?instruction=分析所有因素对目标变量的影响" class="feature-btn">分析所有影响因素</a>
      <a href="/analyze?instruction=哪个因素对结果影响最大" class="feature-btn">找出最重要因素</a>
      <a href="/analyze?instruction=对目标列做全量相关性分析" class="feature-btn">全量相关性分析</a>
      <a href="/analyze?instruction=生成因子重要性报告" class="feature-btn">因子重要性报告</a>
      <a href="/analyze?instruction=找出正相关和负相关因素" class="feature-btn">正负相关因素拆分</a>
    </div>

    <!-- 卡片2：双列关系 -->
    <div class="feature-card">
      <div class="feature-card-header">
        <span class="feature-card-icon">🔗</span> 双列关系分析
      </div>
      <a href="/analyze?instruction=分析两列之间的相关性" class="feature-btn">双列相关性检验</a>
      <a href="/analyze?instruction=计算Pearson和Spearman相关系数" class="feature-btn">Pearson + Spearman</a>
      <a href="/analyze?instruction=分析A列和B列的线性关系" class="feature-btn">双向线性回归</a>
      <a href="/analyze?instruction=绘制两列的分布对比图" class="feature-btn">分布对比图表</a>
      <a href="/analyze?instruction=检验两列相关性的显著性" class="feature-btn">显著性检验</a>
    </div>

    <!-- 卡片3：多因素回归 -->
    <div class="feature-card">
      <div class="feature-card-header">
        <span class="feature-card-icon">📐</span> 多因素回归分析
      </div>
      <a href="/analyze?instruction=多元线性回归分析" class="feature-btn">多元线性回归</a>
      <a href="/analyze?instruction=分析多个自变量对因变量的影响" class="feature-btn">多X对单Y分析</a>
      <a href="/analyze?instruction=检测自变量之间的多重共线性" class="feature-btn">多重共线性检测</a>
      <a href="/analyze?instruction=随机森林特征重要性排序" class="feature-btn">RF 特征重要性</a>
      <a href="/analyze?instruction=生成回归系数和P值报告" class="feature-btn">回归系数 + P 值</a>
    </div>

  </div>

  <!-- 行业示例标签云 -->
  <div class="section-title">🏭 行业专题样例演示</div>
  <div class="tags-cloud">
    {% set examples = [
      "员工工资分析", "员工考勤分析", "员工信息整理",
      "销售额趋势分析", "销售额与温度的关系",
      "广告投放效果分析", "客服数据分析",
      "学生成绩整理", "学生成绩查找", "成绩相关性分析",
      "预算分析", "多表数据分析", "进销存数据分析",
      "资产负债分析", "利润分析", "流量趋势分析",
      "订单管理", "物流管理", "GDP数据整理分析",
      "CPI多表对比分析",
    ] %}
    {% for example in examples %}
    <a href="/analyze?instruction={{ example }}" class="tag-btn">{{ example }}</a>
    {% endfor %}
  </div>

</div>
{% endblock %}
```

**Step 2: 验证首页**

```bash
python app.py
```

打开 `http://localhost:8000`，期望：看到 Banner + 3个绿色卡片 + 标签云，点击任一标签跳转到 `/analyze?instruction=xxx`。

**Step 3: Commit**

```bash
git add templates/home.html
git commit -m "feat: 实现工作台首页（能力卡片 + 行业示例标签云）"
```

---

## Task 6：智能分析工作台（analyze.html）—— 核心页面

**Files:**
- Modify: `templates/analyze.html`（完整实现）
- Modify: `static/js/app.js`（核心交互逻辑）

**Step 1: 完整实现 templates/analyze.html**

```html
{% extends "base.html" %}
{% block title %}智能分析 - Excel 智能分析助手{% endblock %}
{% block header_title %}智能分析工作台{% endblock %}
{% block header_sub %}上传 Excel，用自然语言描述分析需求{% endblock %}

{% block content %}
<div class="analyze-layout" x-data="analyzeApp('{{ prefill_instruction }}')" x-init="init()">

  <!-- ── 左侧：文件面板 ── -->
  <div class="file-panel">
    <div class="file-panel-title">📂 数据文件</div>

    <!-- 上传区 -->
    <div class="upload-zone"
         :class="{'drag-over': isDragging}"
         @dragover.prevent="isDragging=true"
         @dragleave="isDragging=false"
         @drop.prevent="handleDrop($event)"
         @click="$refs.fileInput.click()">
      <input type="file" x-ref="fileInput" accept=".xlsx,.xls"
             style="display:none" @change="handleFileChange($event)">
      <div class="upload-zone-icon">📄</div>
      <div class="upload-zone-text" x-text="fileId ? '点击重新上传' : '点击或拖拽上传'"></div>
      <div class="upload-zone-sub">.xlsx / .xls 格式</div>
    </div>

    <!-- 文件信息 -->
    <template x-if="fileId">
      <div class="file-info">
        <div class="file-info-name" x-text="filename"></div>
        <div class="file-info-meta" x-text="`${rowCount} 行数据`"></div>
      </div>
    </template>

    <!-- 列名 -->
    <template x-if="columns.length > 0">
      <div class="columns-section">
        <div class="columns-label">检测到的数值列：</div>
        <div class="columns-chips">
          <template x-for="col in columns" :key="col">
            <span class="column-chip" x-text="col"></span>
          </template>
        </div>
      </div>
    </template>

    <!-- 分隔线 -->
    <div style="border-top:1px solid var(--border); margin: 14px 0;"></div>

    <!-- AI 模式开关 -->
    <div class="ai-toggle">
      <span>AI 自然语言解析</span>
      <label class="toggle-switch">
        <input type="checkbox" x-model="useAI">
        <span class="toggle-slider"></span>
      </label>
    </div>

    <!-- 手动模式（AI 关闭时显示） -->
    <template x-if="!useAI">
      <div style="margin-top:12px;">
        <div class="form-group" style="margin-bottom:10px;">
          <label class="form-label">分析模式</label>
          <select x-model="manualMode" class="form-input">
            <option value="y_vs_all">全因素分析</option>
            <option value="two_column">双列关系</option>
            <option value="multi_x_vs_y">多因素回归</option>
          </select>
        </div>
        <div class="form-group" style="margin-bottom:10px;">
          <label class="form-label">目标变量 (Y)</label>
          <select x-model="manualY" class="form-input">
            <template x-for="col in columns" :key="col">
              <option :value="col" x-text="col"></option>
            </template>
          </select>
        </div>
        <template x-if="manualMode !== 'y_vs_all'">
          <div class="form-group">
            <label class="form-label">自变量 (X)</label>
            <template x-for="col in columns.filter(c => c !== manualY)" :key="col">
              <label style="display:flex;align-items:center;gap:6px;margin-bottom:4px;font-size:13px;cursor:pointer;">
                <input type="checkbox" :value="col" x-model="manualXCols"> <span x-text="col"></span>
              </label>
            </template>
          </div>
        </template>
      </div>
    </template>
  </div>

  <!-- ── 右侧：聊天面板 ── -->
  <div class="chat-panel">
    <!-- 消息流 -->
    <div class="chat-messages" x-ref="chatMessages">

      <!-- 空状态 -->
      <template x-if="messages.length === 0">
        <div class="chat-empty">
          <div class="chat-empty-icon">💬</div>
          <div class="chat-empty-text">上传文件后，输入分析指令开始对话</div>
          <div style="font-size:12px;color:var(--text-muted);margin-top:4px;">
            例：帮我分析哪些因素对销售额影响最大
          </div>
        </div>
      </template>

      <!-- 消息列表 -->
      <template x-for="(msg, idx) in messages" :key="idx">
        <div>
          <!-- 用户气泡 -->
          <template x-if="msg.role === 'user'">
            <div class="message-user" x-text="msg.content"></div>
          </template>

          <!-- 助手气泡 -->
          <template x-if="msg.role === 'assistant'">
            <div class="message-assistant">
              <!-- Intent Badge -->
              <template x-if="msg.intent">
                <div>
                  <span class="intent-badge">
                    🤖 <span x-text="modeLabel(msg.intent.mode)"></span>
                    · <span x-text="`置信度 ${Math.round(msg.intent.confidence * 100)}%`"></span>
                  </span>
                  <div style="font-size:12px;color:var(--text-secondary);margin-bottom:10px;" x-text="msg.intent.hint"></div>
                </div>
              </template>
              <!-- 数据信息 -->
              <template x-if="msg.dataInfo">
                <div style="font-size:12px;color:var(--text-muted);margin-bottom:8px;"
                     x-text="`📊 原始 ${msg.dataInfo.raw} 行 → 有效 ${msg.dataInfo.valid} 行`"></div>
              </template>
              <!-- Markdown 内容 -->
              <div class="md-content" x-html="renderMd(msg.content)"></div>
              <!-- 下载按钮 -->
              <template x-if="msg.reportFile">
                <a :href="`/api/download/${msg.reportFile}`"
                   class="download-btn" download>
                  📥 下载 Excel 分析报告
                </a>
              </template>
            </div>
          </template>
        </div>
      </template>

      <!-- 加载中 -->
      <template x-if="isLoading">
        <div class="typing-indicator">
          <span>🤖 分析中</span>
          <div class="dots">
            <span></span><span></span><span></span>
          </div>
        </div>
      </template>
    </div>

    <!-- 输入区 -->
    <div class="chat-input-area">
      <textarea
        class="chat-input"
        x-model="instruction"
        :disabled="!fileId || isLoading"
        :placeholder="fileId ? '输入分析指令，例：帮我分析销售额受哪些因素影响...' : '请先上传 Excel 文件'"
        rows="1"
        @keydown.enter.prevent="if(!$event.shiftKey) sendMessage()"
        @input="autoResize($el)"
      ></textarea>
      <button class="send-btn"
              :disabled="!fileId || !instruction.trim() || isLoading"
              @click="sendMessage()">
        发送 ▶
      </button>
    </div>
  </div>

</div>
{% endblock %}

{% block scripts %}
<script>
function analyzeApp(prefillInstruction) {
  return {
    // 状态
    fileId: null,
    filename: '',
    rowCount: 0,
    columns: [],
    messages: [],
    instruction: prefillInstruction || '',
    isLoading: false,
    isDragging: false,
    useAI: true,
    manualMode: 'y_vs_all',
    manualY: '',
    manualXCols: [],

    init() {
      // 如果 URL 带了 instruction 参数，自动聚焦输入框
      if (this.instruction) {
        this.$nextTick(() => {
          const el = document.querySelector('.chat-input');
          if (el) el.focus();
        });
      }
    },

    // ── 文件处理 ──
    handleFileChange(event) {
      const file = event.target.files[0];
      if (file) this.uploadFile(file);
    },
    handleDrop(event) {
      this.isDragging = false;
      const file = event.dataTransfer.files[0];
      if (file) this.uploadFile(file);
    },
    async uploadFile(file) {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.file_id) {
          this.fileId = data.file_id;
          this.filename = data.filename;
          this.rowCount = data.row_count;
          this.columns = data.numeric_columns;
          this.manualY = this.columns[0] || '';
          this.messages = []; // 切换文件时清空对话
        } else {
          alert('上传失败：' + (data.detail || '未知错误'));
        }
      } catch (e) {
        alert('上传失败：' + e.message);
      }
    },

    // ── 发送分析 ──
    async sendMessage() {
      if (!this.fileId || !this.instruction.trim() || this.isLoading) return;

      const userMsg = this.instruction.trim();
      this.messages.push({ role: 'user', content: userMsg });
      this.instruction = '';
      this.isLoading = true;
      this.scrollToBottom();

      try {
        const res = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            file_id: this.fileId,
            instruction: userMsg,
            use_ai: this.useAI,
            manual_mode: this.manualMode,
            manual_y: this.manualY,
            manual_x_cols: this.manualXCols,
          }),
        });
        const data = await res.json();

        if (data.success) {
          this.messages.push({
            role: 'assistant',
            content: data.summary_text,
            intent: data.intent,
            reportFile: data.report_filename,
            dataInfo: data.data_info,
          });
        } else {
          this.messages.push({
            role: 'assistant',
            content: `❌ **分析失败**\n\n${data.error || '未知错误'}`,
            intent: null, reportFile: null,
          });
        }
      } catch (e) {
        this.messages.push({
          role: 'assistant',
          content: `❌ **请求失败**\n\n${e.message}`,
          intent: null, reportFile: null,
        });
      } finally {
        this.isLoading = false;
        this.scrollToBottom();
      }
    },

    // ── 工具函数 ──
    renderMd(text) {
      if (!text) return '';
      return marked.parse(text);
    },
    modeLabel(mode) {
      const map = { y_vs_all: '全因素分析', two_column: '双列关系', multi_x_vs_y: '多因素回归' };
      return map[mode] || mode;
    },
    scrollToBottom() {
      this.$nextTick(() => {
        const el = this.$refs.chatMessages;
        if (el) el.scrollTop = el.scrollHeight;
      });
    },
    autoResize(el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 120) + 'px';
    },
  };
}
</script>
{% endblock %}
```

**Step 2: 验证分析工作台**

```bash
python app.py
```

1. 打开 `http://localhost:8000/analyze`
2. 上传 `C:/Users/29571/PycharmProjects/贾维斯/test_data.xlsx`
3. 期望：左侧显示文件信息和列名 `X1 X2 X3 Target_Y`
4. 输入"分析所有因素对 Target_Y 的影响"，点发送
5. 期望：右侧出现用户气泡 → 加载动画 → AI 结论气泡（含下载按钮）

**Step 3: Commit**

```bash
git add templates/analyze.html static/js/app.js
git commit -m "feat: 实现智能分析工作台（文件上传 + 多轮聊天流 + 报告下载）"
```

---

## Task 7：历史记录页 + 设置页

**Files:**
- Modify: `templates/history.html`
- Modify: `templates/settings.html`

**Step 1: 完整实现 templates/history.html**

```html
{% extends "base.html" %}
{% block title %}历史记录 - Excel 智能分析助手{% endblock %}
{% block header_title %}历史记录{% endblock %}
{% block header_sub %}过去生成的分析报告{% endblock %}

{% block content %}
<div class="main-body" x-data="historyApp()" x-init="load()">
  <div class="section-title">📋 分析报告列表</div>

  <!-- 加载中 -->
  <template x-if="loading">
    <div style="color:var(--text-muted);padding:40px;text-align:center;">加载中...</div>
  </template>

  <!-- 空状态 -->
  <template x-if="!loading && records.length === 0">
    <div style="color:var(--text-muted);padding:60px;text-align:center;">
      <div style="font-size:48px;margin-bottom:12px;">📭</div>
      <div>暂无历史记录，去<a href="/analyze" style="color:var(--accent);">智能分析</a>生成第一份报告吧</div>
    </div>
  </template>

  <!-- 列表 -->
  <template x-if="!loading && records.length > 0">
    <table class="history-table">
      <thead>
        <tr>
          <th>文件名</th>
          <th>大小</th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>
        <template x-for="r in records" :key="r.filename">
          <tr>
            <td>
              <span style="font-size:16px;margin-right:6px;">📊</span>
              <span x-text="r.display_name" style="font-family:monospace;font-size:12px;"></span>
            </td>
            <td style="color:var(--text-muted);" x-text="`${r.size_kb} KB`"></td>
            <td>
              <a :href="`/api/download/${r.filename}`"
                 class="download-btn" download
                 style="font-size:12px;padding:5px 12px;">
                📥 下载
              </a>
            </td>
          </tr>
        </template>
      </tbody>
    </table>
  </template>
</div>
{% endblock %}

{% block scripts %}
<script>
function historyApp() {
  return {
    records: [],
    loading: true,
    async load() {
      const res = await fetch('/api/history');
      this.records = await res.json();
      this.loading = false;
    }
  };
}
</script>
{% endblock %}
```

**Step 2: 完整实现 templates/settings.html**

```html
{% extends "base.html" %}
{% block title %}设置 - Excel 智能分析助手{% endblock %}
{% block header_title %}API 设置{% endblock %}
{% block header_sub %}配置 AI 服务连接参数{% endblock %}

{% block content %}
<div class="main-body" x-data="settingsApp()" x-init="load()">
  <div class="settings-card">
    <h2>⚙️ DeepSeek API 配置</h2>

    <div class="form-group">
      <label class="form-label">API Key</label>
      <input type="password" class="form-input" x-model="form.api_key"
             placeholder="sk-...">
      <div class="form-hint">来自 SiliconFlow 或 DeepSeek 官方的 API Key</div>
    </div>

    <div class="form-group">
      <label class="form-label">Base URL</label>
      <input type="text" class="form-input" x-model="form.base_url"
             placeholder="https://api.siliconflow.cn/v1">
    </div>

    <div class="form-group">
      <label class="form-label">模型名称</label>
      <input type="text" class="form-input" x-model="form.model"
             placeholder="deepseek-ai/DeepSeek-V3">
    </div>

    <div style="display:flex;align-items:center;margin-top:8px;">
      <button class="save-btn" @click="save()" :disabled="saving">
        <span x-text="saving ? '保存中...' : '保存配置'"></span>
      </button>
      <span class="save-success" x-show="saved">
        ✅ 保存成功
      </span>
    </div>
  </div>
</div>
{% endblock %}

{% block scripts %}
<script>
function settingsApp() {
  return {
    form: { api_key: '', base_url: '', model: '' },
    saving: false,
    saved: false,
    async load() {
      const res = await fetch('/api/settings');
      this.form = await res.json();
    },
    async save() {
      this.saving = true;
      this.saved = false;
      await fetch('/api/settings', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(this.form),
      });
      this.saving = false;
      this.saved = true;
      setTimeout(() => this.saved = false, 3000);
    }
  };
}
</script>
{% endblock %}
```

**Step 3: 验证两个页面**

1. `http://localhost:8000/history` — 应显示 outputs/ 目录中的 5 个测试报告
2. `http://localhost:8000/settings` — 应显示已有的 API Key（密码遮盖），修改后保存验证 .env 文件内容变更

**Step 4: Commit**

```bash
git add templates/history.html templates/settings.html
git commit -m "feat: 实现历史记录页和设置页"
```

---

## Task 8：Dockerfile + 最终整合验证

**Files:**
- Create: `Dockerfile`
- Modify: `requirements.txt`（最终版）

**Step 1: 创建 Dockerfile**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖（先复制 requirements 利用 Docker 缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要目录
RUN mkdir -p outputs uploads

# ModelScope 创空间默认暴露 7860 端口
EXPOSE 7860

# 启动（PORT 环境变量由平台注入，默认 7860）
CMD ["python", "app.py"]
```

**Step 2: 确认 requirements.txt 最终版**

```
pandas>=2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
matplotlib>=3.7
openpyxl>=3.1
xlsxwriter>=3.1
python-dotenv>=1.0
openai>=1.0
fastapi>=0.110
uvicorn[standard]>=0.27
jinja2>=3.1
python-multipart>=0.0.9
```

**Step 3: 确认 config.py 中 OUTPUTS_DIR 和 UPLOADS_DIR 在 Docker 内可写**

验证 `config.py` 中路径使用 `Path(__file__).parent / "outputs"` 这样的相对路径（Docker 内 WORKDIR=/app，相对路径正常工作）。

**Step 4: 端到端完整流程测试**

```bash
python app.py
```

按序验证：
1. 打开 `http://localhost:8000` → 首页正常，卡片和标签可见
2. 点击标签"员工工资分析" → 跳转 `/analyze?instruction=员工工资分析`，指令自动填入输入框
3. 上传 `test_data.xlsx` → 左侧显示文件名、行数、列名
4. 点发送 → 等待约 5-10 秒 → 右侧出现分析结论气泡和下载按钮
5. 点下载 → 下载报告 xlsx 文件
6. 再发一条"分析 X1 和 Target_Y 的关系" → 第二轮对话追加
7. 打开 `http://localhost:8000/history` → 看到刚生成的报告
8. 打开 `http://localhost:8000/settings` → 看到 API 配置

**Step 5: 最终 Commit**

```bash
git add Dockerfile requirements.txt
git commit -m "feat: 添加 Dockerfile，完成 ChatExcel 风格前端完整实现"
```

---

## 快速参考：文件改动总览

| 操作 | 文件 |
|------|------|
| 全量重写 | `app.py`, `requirements.txt` |
| 全量新建 | `api/__init__.py`, `api/routes.py` |
| 全量新建 | `static/css/main.css`, `static/js/app.js` |
| 全量新建 | `templates/base.html`, `home.html`, `analyze.html`, `history.html`, `settings.html` |
| 新建 | `Dockerfile` |
| **完全不动** | `core/`, `utils/`, `prompts/`, `config.py`, `.env`, `outputs/` |
