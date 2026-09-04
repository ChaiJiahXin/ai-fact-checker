# ======================================================================
#  AI 事实核查器 (AI Fact Checker) —— Streamlit 应用容器化镜像
#  MUBA Hackathon 2026 · GonkaRouter
# ======================================================================
#  构建（在 v1 目录下执行）:
#     docker build -f docker/Dockerfile -t ai-fact-checker:v1 .
# ======================================================================

FROM python:3.11-slim

# 容器环境配置：关闭输出缓冲 / 字节码缓存 / pip 缓存，减小镜像体积
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先复制依赖清单并安装（充分利用 Docker 层缓存：依赖不变时无需重新安装）
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用源码
COPY app.py ./

# Streamlit 无头服务配置（容器内必须以 0.0.0.0 监听并固定端口）
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# 健康检查：探测 Streamlit 内置健康端点
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).status == 200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
