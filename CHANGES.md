# 系统更新说明

## 2025-01-13 更新内容 (第二次更新)

### 权限控制增强
1. **API 响应字段调整**：
   - 移除 `is_admin` 字段
   - 新增 `normal` 字段（`true` 表示普通用户，`false` 表示本地）

2. **普通用户限制**：
   - **状态显示**：所有 API Key 的状态永远显示为 `active`（活跃）
   - **模型标签**：不可点击，鼠标悬停时不显示手型指针
   - **今日消耗**：不可点击，无法查看详细消耗明细
   - **数据范围**：只能查看自己的数据，无法查看其他用户信息

3. **本地权限**（`normal=false`）：
   - 可以查看真实的 API Key 状态（`active`/`window`/`ok`）
   - 可以点击模型标签查看详细使用情况
   - 可以点击今日消耗查看每小时消耗明细
   - 可以查看所有用户的完整数据

## 2025-01-13 更新内容 (第一次更新)

### 1. 统计计算后移到后端
- **周统计 (Weekly Statistics)**: 所有周统计计算现在在后端完成，前端只负责显示
- **窗口统计 (Window Statistics)**: 窗口期消耗统计由后端计算并返回
- **模型统计 (Model Statistics)**: 每个模型的详细统计在后端完成计算

### 2. 权限管理系统
- **ADMIN_LIST 环境变量**: 替换原有的 `SHOW_EXACT_TIME_LIST`，现在称为 `ADMIN_LIST`
- **本地权限**:
  - 可以查看所有用户的详细统计数据 (`detailed_stats`)
  - 可以查看所有模型的详细使用数据 (`model_detailed_stats`)
  - 可以看到精确的最后使用时间（格式：YYYY-MM-DD HH:MM:SS）
- **普通用户权限**:
  - 只能查看自己的详细统计数据
  - 只能查看自己使用过的模型的统计数据
  - 只能看到相对的最后使用时间（如：正在使用、刚刚使用、一天内等）

### 3. API 响应优化
- 新增 `weekly_stats` 字段，包含：
  - `timeProgress`: 时间进度百分比
  - `remainingText`: 剩余时间文字描述
  - `weeklyCost`: 本周总消耗
  - `weeklyLimit`: 周限额
  - `costPercentage`: 消耗百分比
  - `weekRange`: 周范围描述
  - `userWeeklyCosts`: 各用户周消耗列表
- 新增 `is_admin` 字段，标识当前用户是否为本地

### 4. 前端优化
- 移除前端的周统计计算逻辑，直接使用后端返回的数据
- 简化前端代码，减少重复计算
- 提高页面加载速度

## 配置说明

### 环境变量设置
```bash
# 设置本地列表（逗号分隔）
export ADMIN_LIST="admin1,admin2,admin3"

# 启动服务器（调试模式）
DEBUG=1 python server/api_stats_server_v2.py

# 启动服务器（生产模式）
python server/api_stats_server_v2.py
```

### 权限验证
- 系统会根据 API Key 对应的用户名判断权限
- 本地用户名必须在 `ADMIN_LIST` 环境变量中
- 非本地用户访问时会自动过滤数据，只显示其有权查看的内容

## 技术细节

### 后端实现
1. `calculate_weekly_statistics()`: 计算周统计数据
2. `generate_all_detailed_stats()`: 生成所有详细统计
3. 数据过滤逻辑在 `/simple-board/query_all` 端点中实现
4. 根据 `requesting_user_name` 是否在 `ADMIN_LIST` 中决定返回数据范围

### 前端实现
1. `createWeeklyWindowCard()`: 使用后端提供的周统计数据
2. `displayAccounts()`: 直接使用后端的 `userWeeklyCosts`
3. 移除了前端的统计计算逻辑，提高性能

## 注意事项
- 确保在启动服务器前设置好 `ADMIN_LIST` 环境变量
- 本地列表中的用户名必须与 Redis 中的 API Key 名称完全匹配
- 更改 `ADMIN_LIST` 后需要重启服务器才能生效