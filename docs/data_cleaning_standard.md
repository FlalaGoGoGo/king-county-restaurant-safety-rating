# 数据清洗标准 (Food Establishment Inspection Data)

版本: v0.2  
更新时间: 2026-02-27  
适用数据源: `f29f-zza5` (Socrata Open Data)

## 1. 目标
- 建立可日更的标准化流水线，保证:
  - 查询友好（搜索/筛选/地理计算）
  - 建模友好（检查级样本、稳定标签）
  - 可追溯（每条规则可解释、可回放）

## 2. 分层模型

### 2.1 Bronze (原始层)
- 原样落地 API 拉取结果，不修改字段值。
- 保留 `ingest_time_utc`, `source_url`, `source_snapshot_date`。

### 2.2 Silver (标准层)
- 对字段做类型化、规范化、异常标记。
- 输出两张核心表:
  - `silver_inspection_row`（原始粒度: 检查x违规项）
  - `silver_inspection_event`（检查粒度: 一次检查一行）

### 2.3 Gold (应用层)
- `gold_business_profile`: 商户搜索与详情页
- `gold_inspection_timeline`: 历史检查序列
- `gold_violation_fact`: 违规主题分析
- `gold_model_dataset`: 模型训练集（按任务定义标签）

## 3. 主键与去重策略

### 3.1 行级主键
- 优先使用 `violation_record_id`。
- 若为空，构造替代键:
  - `sha1(business_id|inspection_serial_num|inspection_date|violation_type|violation_description|violation_points)`

### 3.2 检查级主键
- 优先使用 `inspection_serial_num`。
- 若为空，构造替代键:
  - `sha1(business_id|inspection_date|inspection_type|inspection_result|inspection_score)`

### 3.3 去重规则
- 完全重复记录: 去重保留最新 `ingest_time_utc`。
- 替代键冲突: 保留信息更完整（非空字段数更多）的一条，并写入冲突日志。

## 4. 字段标准化规则

### 4.1 文本通用规则
- `trim` 前后空格。
- 连续空白折叠为单空格。
- 原值保留为 `*_raw`，标准值存 `*_clean`。

### 4.2 名称字段
- 原始字段: `name`, `program_identifier`, `inspection_business_name`
- 生成:
  - `business_name_official`（优先 `name`）
  - `business_name_alt`（`program_identifier`）
  - `search_name_norm`（去标点、统一大小写、常见公司后缀弱化）

### 4.3 地址字段
- 标准化 `address/city/zip_code`，构造 `full_address_clean`。
- 生成 `address_norm_key` 用于模糊匹配（去标点、街道缩写统一）。

### 4.4 城市字段
- `city_canonical = UPPER(TRIM(city))`。
- 常见拼写归一:
  - `SEA TAC -> SEATAC`
  - `SEATTLE, -> SEATTLE`
  - `SEATLLE -> SEATTLE`
  - `BELLEUE -> BELLEVUE`
  - `TUWKILA -> TUKWILA`
- 子区域/别名归一:
  - `WEST SEATTLE -> SEATTLE`
  - `VASHON ISLAND -> VASHON`
- 允许值分三类:
  - King County incorporated cities
  - King County unincorporated localities（如 `FALL CITY / HOBART / PRESTON / RAVENSDALE / VASHON / SNOQUALMIE PASS`）
  - `Outside King County`
- 二次修正规则:
  - 若 `city` 不在允许值内，但 `zip_code` 能稳定映射到某个 King County locality（高频主值），则按 zip 主值修正。
  - 当前已验证样例: `ABERDEEN + 98052 -> REDMOND`
  - 若既不在允许值内，也无法通过 zip/地址稳定修正，则统一归为 `Outside King County`，同时保留原始值。
- 额外输出:
  - `city_is_suspect`（疑似异常城市或拼写）
  - `city_norm_reason`（规则命中说明）
  - `city_scope`（incorporated / unincorporated / outside）
  - `city_cleaning_reason`
  - `city_raw_original`

### 4.5 时间字段
- `inspection_date` 统一为 `date`。
- 新增 `inspection_year`, `inspection_month`, `inspection_week`。

### 4.6 数值字段
- `inspection_score`, `violation_points` 转数值类型。
- 额外标记:
  - `inspection_score_is_negative`（例如 -10/-2/-1）
  - `score_outlier_flag`（按业务阈值）

### 4.7 地理字段
- `latitude/longitude` 需在 King County 合理范围（近似 bbox）内。
- 超出范围记录 `geo_out_of_bounds=1`。

### 4.8 违规字段
- 解析违规代码:
  - 正则 `^([0-9]{4})\\s*-\\s*(.*)$`
  - 生成 `violation_code`, `violation_desc_clean`
- 标记 `violation_desc_truncated=1` 当描述含 `...`。
- 建立 `violation_dictionary`:
  - `violation_code`
  - `canonical_description`
  - `violation_type`（RED/BLUE）
  - `default_points`
  - `recommended_actions`（整改建议）

### 4.9 评级字段
- 标准化官方字段:
  - 保留 `grade` 原始值并转为整数标签:
  - 1=Excellent
  - 2=Good
  - 3=Okay
  - 4=Needs to Improve
  - 同步生成 `grade_label_standard`
- 从原始 `description` 解析并沉淀:
  - `risk_description_raw`
  - `risk_level`（1/2/3）
- 基于官网规则生成解释性字段:
  - `rating_window_target`:
    - Risk 3 -> 最近 4 次 `Routine Inspection/Field Review`
    - Risk 1/2 -> 最近 2 次 `Routine Inspection/Field Review`
  - `rating_avg_red_points_recent`
  - `rating_recent_routine_count_used`
  - `rating_has_required_routine_window`（是否已经达到完整窗口上限）
  - `rating_recent_closure_90d_flag`
  - `rating_recent_return_inspection_count_90d`
  - `recalculated_rating_label`
- 最终展示字段 `effective_rating_label` 规则:
  - 先判断 `Needs to Improve`:
    - 最近 90 天存在 `inspection_closed_business=1`
    - 或最近 90 天 `Return Inspection` 次数 `>= 2`
  - 若未触发 `Needs to Improve`，则按“最近可用 routine inspection 次数”计算平均分:
    - Risk 3: 使用最近 `min(已有 routine inspections, 4)` 次
    - Risk 1/2: 使用最近 `min(已有 routine inspections, 2)` 次
  - 只要至少存在 1 次 routine inspection，就按官网阈值把平均 red points 映射到 `Excellent / Good / Okay`
  - 若没有任何 routine inspection，才标记为 `Rating not available`
  - 官方 `grade` 保留在 `official_rating_label`，仅用于审计对照，不再回填本站主显示评级

## 5. 质检规则 (DQ Checks)

每次增量运行后输出 `dq_report.json` 与 `dq_report.md`。

必检项:
- `inspection_serial_num` 缺失率
- `violation_record_id` 缺失率
- `city_canonical` 异常值 TopN
- `geo_out_of_bounds` 数量
- `violation_desc_truncated` 数量
- `grade` 分布漂移（与近 90 天比较）

门槛策略:
- `blocker`: 主键冲突率超阈值、日期解析失败率超阈值
- `warning`: 城市异常升高、坐标异常升高、截断描述激增

## 6. 增量更新流程 (Daily)

1. 读取上次 `max(inspection_date)` 与 `max(ingest_time_utc)`。  
2. 拉取最新数据快照（Socrata API）。  
3. 写入 Bronze，保留原始快照。  
4. 执行 Silver 标准化与去重。  
5. 生成检查级表（event-level）与 Gold 主题数据集。  
6. 运行 DQ 检查并输出报告。  
7. 若 `blocker` 出现，停止发布并告警。  

## 7. 面向后续功能的最小输出字段

搜索必需:
- `business_id`, `business_name_official`, `business_name_alt`, `full_address_clean`, `city_canonical`, `zip_code`, `latitude`, `longitude`, `grade`, `risk_level`, `effective_rating_label`, `official_rating_label`, `city_scope`, `city_cleaning_reason`

时间序列与预测必需:
- `inspection_event_id`, `business_id`, `inspection_date`, `inspection_type`, `inspection_result`, `inspection_score`, `grade`, `risk_level`, `red_points_total`, `blue_points_total`, `red_violation_count`, `blue_violation_count`, `rating_avg_red_points_recent`, `rating_recent_routine_count_used`, `effective_rating_label`

违规解释必需:
- 优先使用 `Data/gold/f29f-zza5/<run_id>/dashboard_violation_explained.csv`
- 核心字段:
  - `violation_code`, `violation_type`, `violation_desc_clean`, `violation_points`
  - `action_category`, `action_priority`
  - `action_summary_zh`, `action_steps_zh`, `safe_food_handling_refs`
  - `action_source`（区分字典命中与兜底规则）

## 8. 当前已发现问题摘要 (用于回归检查)
- `name != program_identifier`: 68,954 行
- `city` 大小写不一致: 77,065 行
- `violation_description` 含 `...`: 32,286 行
- `inspection_serial_num` 为空: 419 行
- 经纬度疑似越界: 7 行
