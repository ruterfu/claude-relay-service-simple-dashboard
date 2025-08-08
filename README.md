# 简单的 CC Relay Service的一个快速查看面板

基于项目 [Wei-Shaw/claude-relay-service](https://github.com/Wei-Shaw/claude-relay-service) 糊了一个简单的统计信息查看面板，纯静态的单文件html + 只要能连接到Claude-relay-service同一个redis的python后端就行。

纯用Claude写的，里面的 /model_pricing.json 就是 [Wei-Shaw/claude-relay-service](https://github.com/Wei-Shaw/claude-relay-service) 的 Model Pricing Data

## Demo页面

[**Ruter的CC统计面板 - 演示版**](https://oss.ruterfu.com/public/cc-simple-static-dashboard.html)

## 快速开始

### 方法一：直接运行

```bash
pip install fastapi uvicorn redis pydantic

export ENCRYPTION_KEY="和启动Claude Relay Service的ENCRYPTION_KEY一样，用来计算api_key-id用"
export REDIS_HOST=127.0.0.1
export REDIS_PORT=6379
python3 server/api_stats_server_v2.py
```

### 方法二：Docker 部署

```bash
# 构建镜像
docker build -t ruterfu/claude-relay-simple-board:v1 .

# 运行容器
docker run -d \
  -p 8001:8001 \
  --name claude-relay-simple-board \
  --restart=unless-stopped \
  -e ENCRYPTION_KEY="your_encryption_key" \
  -e REDIS_HOST="your_redis_host" \
  -e REDIS_PORT="6379" \
  ruterfu/claude-relay-simple-board:v1
```

## API 端点

- `GET /index.html` - 仪表板页面
- `POST /simple-board/query_all` - 查询所有 API Key 统计信息
- `GET /health` - 健康检查
- `GET /` - API 信息

## 配置

### 环境变量

- `ENCRYPTION_KEY` - **必需**，与 Claude Relay Service 相同的加密密钥
- `REDIS_HOST` - Redis 主机地址（默认：生产模式使用 192.168.118.2）
- `REDIS_PORT` - Redis 端口（默认：生产模式使用 6379）

### 认证

系统支持两种认证方式：
1. API Key 认证：使用 `cr_` 开头的 API Key
2. Token 认证：使用生成的认证令牌（同一个机器免登录）

## 服务器信息

服务器将在 `http://127.0.0.1:8001` 启动

访问仪表板：`http://127.0.0.1:8001/index.html`

