"""
Excel 智能分析助手 - FastAPI 入口
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from api.routes import router
from config import OUTPUTS_DIR

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# 确保 uploads 目录存在
uploads_dir = OUTPUTS_DIR.parent / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Excel 智能分析助手", version="2.0")

# 静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# 路由
app.include_router(router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
