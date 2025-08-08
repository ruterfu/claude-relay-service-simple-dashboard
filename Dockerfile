# 使用官方Python运行时作为父镜像
FROM python:3.10-slim

# 安装所需的Python包和系统工具
RUN apt-get update && apt-get install -y curl && \
    pip install --no-cache-dir fastapi uvicorn redis pydantic && \
    apt-get clean && rm -rf /var/lib/apt/lists/*



# 设置工作目录
WORKDIR /app

# 复制当前目录内容到工作目录
COPY index.html /app/index.html
COPY server/api_stats_server_v2.py /app

# 下载模型定价数据
# COPY server/model_pricing.json /app/model_pricing.json
RUN curl -s https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json -o model_pricing.json

# 暴露端口
EXPOSE 8001

# 设置环境变量
ENV PYTHONPATH=/app

# 运行应用程序
CMD ["python", "api_stats_server_v2.py"]