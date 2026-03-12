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
