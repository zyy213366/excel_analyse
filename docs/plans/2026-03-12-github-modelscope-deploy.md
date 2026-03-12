# GitHub + ModelScope 部署计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 Excel 智能分析助手干净地推送到 GitHub，并通过 GitHub Actions 自动同步部署到 ModelScope 创空间。

**Architecture:** 新增 SQLite 轻量数据库替代纯内存的文件映射（重启不丢失），增加 ModelScope CI/CD workflow，修复 gitignore，在创空间通过环境变量注入 API Key。整体不改动分析逻辑，只动基础设施层。

**Tech Stack:** SQLite（Python 内置）、GitHub Actions、ModelScope Git 推送、Docker（已有 Dockerfile）

---

## 背景：当前存在的问题

| 问题 | 现状 | 目标 |
|------|------|------|
| `uploads/` 未 gitignore | 用户文件会上传 GitHub | 加入忽略，启动时从 DB 重建 |
| `_uploaded_files` 纯内存 | 重启丢失所有会话 | SQLite 持久化 file_id→路径映射 |
| 历史记录仅扫描目录 | 无元数据（谁上传、用什么指令）| SQLite 记录完整分析历史 |
| 无 ModelScope CI/CD | 只有 HF workflow | 新增 sync-to-modelscope.yml |
| API Key 写在 .env | 云端部署需环境变量注入 | 支持纯环境变量（已部分支持）|

---

## Task 1：修复 .gitignore + 清理敏感/无关文件

**Files:**
- Modify: `.gitignore`

**Step 1：在 .gitignore 中补充缺失条目**

在文件末尾追加：
```
# 用户上传文件（含个人数据，不应进入版本库）
uploads/

# 服务器运行日志
server.log

# SQLite 数据库（运行时生成）
*.db
*.db-journal
*.db-wal
*.db-shm
```

**Step 2：从 git 追踪中移除已被追踪的 uploads/ 内容（如果有）**

```bash
git rm -r --cached uploads/ 2>/dev/null || true
git rm --cached server.log 2>/dev/null || true
```

**Step 3：验证**

```bash
git status
# 预期：uploads/ 和 server.log 不再出现在 tracked 列表中
```

**Step 4：提交**

```bash
git add .gitignore
git commit -m "chore: exclude uploads/, server.log, *.db from version control"
```

---

## Task 2：新增 SQLite 持久化层（`db.py`）

**Files:**
- Create: `db.py`

**目标：** 替代 `api/routes.py` 中的 `_uploaded_files: dict`，让重启后文件映射依然存在；同时记录每次分析历史（比纯目录扫描更丰富）。

**Step 1：创建 `db.py`**

```python
"""
SQLite 持久化层
- uploads  表：记录上传文件的 file_id ↔ 磁盘路径映射
- analyses 表：记录每次分析的元数据（历史记录）
"""
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from config import BASE_DIR

DB_PATH = BASE_DIR / "data.db"

# 线程本地连接（SQLite 连接不能跨线程共享）
_local = threading.local()


def get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def init_db() -> None:
    """建表（幂等，已存在则忽略）"""
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS uploads (
            file_id          TEXT PRIMARY KEY,
            original_name    TEXT NOT NULL,
            save_path        TEXT NOT NULL,
            row_count        INTEGER DEFAULT 0,
            numeric_cols     TEXT DEFAULT '',   -- JSON 数组字符串
            created_at       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS analyses (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id          TEXT NOT NULL,
            original_name    TEXT NOT NULL,
            instruction      TEXT DEFAULT '',
            mode             TEXT DEFAULT '',
            report_filename  TEXT DEFAULT '',
            created_at       TEXT NOT NULL
        );
    """)
    conn.commit()


# ── uploads 表操作 ──────────────────────────────

def save_upload(file_id: str, original_name: str,
                save_path: str, row_count: int,
                numeric_cols: list[str]) -> None:
    import json
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO uploads
           (file_id, original_name, save_path, row_count, numeric_cols, created_at)
           VALUES (?,?,?,?,?,?)""",
        (file_id, original_name, save_path,
         row_count, json.dumps(numeric_cols, ensure_ascii=False),
         datetime.now().isoformat()),
    )
    conn.commit()


def get_upload(file_id: str) -> sqlite3.Row | None:
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM uploads WHERE file_id = ?", (file_id,)
    ).fetchone()


def load_all_uploads() -> dict[str, Path]:
    """启动时重建内存 dict：file_id → Path（只保留磁盘上仍存在的文件）"""
    conn = get_conn()
    rows = conn.execute("SELECT file_id, save_path FROM uploads").fetchall()
    result = {}
    for row in rows:
        p = Path(row["save_path"])
        if p.exists():
            result[row["file_id"]] = p
    return result


# ── analyses 表操作 ─────────────────────────────

def save_analysis(file_id: str, original_name: str,
                  instruction: str, mode: str,
                  report_filename: str) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT INTO analyses
           (file_id, original_name, instruction, mode, report_filename, created_at)
           VALUES (?,?,?,?,?,?)""",
        (file_id, original_name, instruction, mode,
         report_filename, datetime.now().isoformat()),
    )
    conn.commit()


def get_analyses(limit: int = 50) -> list[sqlite3.Row]:
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
```

**Step 2：验证建表逻辑（不依赖任何外部服务）**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from db import init_db, save_upload, get_upload
init_db()
save_upload('test-id', 'test.xlsx', '/tmp/test.xlsx', 100, ['A','B'])
row = get_upload('test-id')
assert row['original_name'] == 'test.xlsx'
assert row['row_count'] == 100
print('DB 测试通过')
import os; os.remove('data.db')
"
```

预期输出：`DB 测试通过`

**Step 3：提交**

```bash
git add db.py
git commit -m "feat: add SQLite persistence layer for uploads and analysis history"
```

---

## Task 3：将 `api/routes.py` 接入 SQLite

**Files:**
- Modify: `api/routes.py`

**Step 1：替换内存 dict 为 SQLite**

在 `routes.py` 顶部导入部分，替换：
```python
# 内存中存储上传文件路径（file_id → 临时文件路径）
_uploaded_files: dict[str, Path] = {}
```
为：
```python
from db import init_db, save_upload, get_upload, load_all_uploads, save_analysis

# 启动时从 SQLite 重建（重启不丢失已上传文件的路径映射）
init_db()
_uploaded_files: dict[str, Path] = load_all_uploads()
```

**Step 2：在 `api_upload` 中同步写入 SQLite**

在 `_uploaded_files[file_id] = save_path` 这行之后追加：
```python
save_upload(file_id, file.filename, str(save_path), row_count, numeric_cols)
```

**Step 3：在 `api_analyze` 中记录分析历史**

在 `report_filename = out_path.name` 那行之后追加：
```python
try:
    save_analysis(
        req.file_id,
        _uploaded_files[req.file_id].name.split("_", 1)[-1],  # 原始文件名
        req.instruction,
        mode,
        report_filename or "",
    )
except Exception:
    pass  # 历史记录失败不影响主流程
```

**Step 4：升级 `api_history` 接口，读取 SQLite 历史**

将原来只扫描 `outputs/*.xlsx` 的逻辑改为从 `analyses` 表读取：
```python
@router.get("/api/history")
async def api_history():
    from db import get_analyses
    rows = get_analyses(limit=50)
    result = []
    for r in rows:
        fname = r["report_filename"]
        fpath = OUTPUTS_DIR / fname if fname else None
        result.append({
            "filename":      fname,
            "display_name":  r["original_name"],
            "instruction":   r["instruction"],
            "mode":          r["mode"],
            "size_kb":       round(fpath.stat().st_size / 1024, 1) if fpath and fpath.exists() else 0,
            "modified":      r["created_at"],
        })
    return result
```

**Step 5：验证服务正常启动**

```bash
python app.py &
sleep 3
curl http://localhost:7860/api/history
# 预期：返回 [] 或历史列表，不报错
```

**Step 6：提交**

```bash
git add api/routes.py
git commit -m "feat: persist uploads/analyses to SQLite, restore on restart"
```

---

## Task 4：更新 Dockerfile 和 config.py 以支持环境变量注入

**Files:**
- Modify: `Dockerfile`
- Modify: `config.py`

**背景：** ModelScope 创空间通过 Space Settings → "环境变量" 面板设置 secrets，不会有 `.env` 文件。`config.py` 已经使用 `os.getenv()` 所以 API Key 本身没问题，但需要确保 `data.db` 写在可写目录，且 `uploads/` 和 `outputs/` 目录在容器启动时存在。

**Step 1：`config.py` 增加 DATA_DIR 配置**

在 `BASE_DIR` 下面增加：
```python
# 云端部署时可通过环境变量将数据目录指向持久化挂载点
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR)))
```

然后把 `db.py` 中的 `DB_PATH` 改为：
```python
from config import DATA_DIR
DB_PATH = DATA_DIR / "data.db"
```

**Step 2：更新 Dockerfile，确保目录和非 root 权限**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建运行时目录
RUN mkdir -p outputs uploads

# ModelScope 创空间默认暴露 7860 端口
EXPOSE 7860

# 启动
CMD ["python", "app.py"]
```

（和现在一样，只是注释更清晰）

**Step 3：本地 Docker 构建验证**

```bash
docker build -t excel-analyzer-test .
docker run --rm -p 7860:7860 \
  -e DEEPSEEK_API_KEY=sk-test \
  excel-analyzer-test
# 预期：服务启动，访问 http://localhost:7860 正常
```

**Step 4：提交**

```bash
git add Dockerfile config.py db.py
git commit -m "chore: support DATA_DIR env var for cloud storage path"
```

---

## Task 5：写 ModelScope 创空间 README 头

**Files:**
- Modify: `README.md`

**背景：** ModelScope 创空间（Studio）识别 README.md 开头的 YAML Front Matter 来确定 SDK 类型、端口等信息。Docker 类型的格式如下：

**Step 1：在 `README.md` 最顶部插入 Front Matter**

```yaml
---
title: Excel 智能分析助手
emoji: 📊
colorFrom: green
colorTo: teal
sdk: docker
app_port: 7860
license: mit
---
```

注意：这段必须是文件的第 1 行，前面不能有任何内容。

**Step 2：提交**

```bash
git add README.md
git commit -m "docs: add ModelScope Space front matter for Docker SDK"
```

---

## Task 6：新增 GitHub Actions → ModelScope 自动同步 workflow

**Files:**
- Create: `.github/workflows/sync-to-modelscope.yml`

**原理：** 每次 push master 时，通过 git push 将代码同步到 ModelScope 的创空间 git 仓库，ModelScope 检测到推送后自动重新构建 Docker 镜像并部署。

**Step 1：创建 workflow 文件**

```yaml
name: Sync to ModelScope Studio

on:
  push:
    branches:
      - master

jobs:
  sync-to-modelscope:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          lfs: true

      - name: Push to ModelScope Studio
        env:
          MS_TOKEN:    ${{ secrets.MS_TOKEN }}
          MS_USERNAME: ${{ secrets.MS_USERNAME }}
        run: |
          git config --global user.email "action@github.com"
          git config --global user.name  "GitHub Action"
          git push https://$MS_USERNAME:$MS_TOKEN@www.modelscope.cn/studios/$MS_USERNAME/excel-analyzer.git \
            master:master --force
```

**Step 2：在 GitHub 仓库配置 Secrets**

进入 GitHub 仓库 → Settings → Secrets and variables → Actions → New repository secret，添加：

| Secret 名 | 值 |
|-----------|---|
| `MS_TOKEN` | ModelScope 访问令牌（个人中心 → 访问令牌） |
| `MS_USERNAME` | ModelScope 用户名（如 `zhangsan`） |

**Step 3：提交 workflow**

```bash
git add .github/workflows/sync-to-modelscope.yml
git commit -m "ci: add GitHub Actions workflow to sync to ModelScope Studio"
```

---

## Task 7：在 ModelScope 创建创空间并配置环境变量

**（本步骤在浏览器中手动操作，无需写代码）**

**Step 1：创建创空间**

1. 登录 [modelscope.cn](https://www.modelscope.cn)
2. 进入「我的创空间」→「新建创空间」
3. 填写：
   - 空间名称：`excel-analyzer`
   - 运行环境：**Docker**
   - 可见性：公开

**Step 2：配置环境变量（相当于 .env）**

进入创空间 → 设置 → 环境变量，添加：

| 变量名 | 值 |
|--------|---|
| `DEEPSEEK_API_KEY` | 你的 DeepSeek/SiliconFlow API Key |
| `DEEPSEEK_BASE_URL` | `https://api.siliconflow.cn/v1` |
| `DEEPSEEK_MODEL` | `deepseek-ai/DeepSeek-V3` |

> ⚠️ **绝对不要**把 API Key 写进代码或 README。

**Step 3：触发首次构建**

执行 Task 6 中的 git push，GitHub Actions 会自动同步代码到 ModelScope，触发 Docker 构建。在创空间的「构建日志」中查看进度（约 2-5 分钟）。

---

## Task 8：推送到 GitHub 并验证全流程

**Step 1：确认 .env 不在追踪列表**

```bash
git status
# 确认没有 .env、uploads/、*.db 出现
```

**Step 2：推送到 GitHub**

```bash
git push origin master
```

**Step 3：观察 GitHub Actions**

进入 GitHub 仓库 → Actions，确认两个 workflow 都成功：
- `Sync to ModelScope Studio` ✅

**Step 4：验证 ModelScope 部署**

访问 `https://modelscope.cn/studios/{你的用户名}/excel-analyzer`

测试清单：
- [ ] 首页正常加载
- [ ] 上传 Excel/CSV 文件成功
- [ ] 自然语言分析正常（需 API Key 配置正确）
- [ ] 下载报告正常
- [ ] 历史记录页面有数据（SQLite 工作）
- [ ] 重启创空间后已上传文件路径仍存在（SQLite 持久化生效）

---

## 补充说明：关于 SQLite 在 ModelScope 的持久性

ModelScope Docker 创空间的文件系统**在两次构建之间会被重置**（镜像重新构建），但**进程重启不会**。

因此 SQLite 解决了「进程崩溃/重启后丢失会话」问题，但无法解决「重新部署后丢失数据」问题。对于公开 Demo 类应用这完全够用——用户重新上传文件即可。

如果未来需要跨部署持久化，可将 `DATA_DIR` 指向 ModelScope 的对象存储挂载点（OSS），届时只需改一个环境变量。
