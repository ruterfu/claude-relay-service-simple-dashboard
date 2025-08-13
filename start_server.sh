#!/bin/bash
# 启动服务器脚本

echo "======================================"
echo "启动 API 统计服务器"
echo "======================================"

# 设置管理员列表（根据需要修改）
export ADMIN_LIST="test0,admin1,admin2"
echo "✅ 管理员列表: $ADMIN_LIST"

# 设置调试模式
export DEBUG=1
echo "✅ 调试模式: 已启用"

# 切换到服务器目录
cd server

# 启动服务器
echo ""
echo "🚀 启动服务器..."
echo "======================================"
python api_stats_server_v2.py