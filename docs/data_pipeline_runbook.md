# Data Pipeline Runbook

更新时间: 2026-02-25

## 1. 入口脚本
- [run_food_inspection_pipeline.py](/Users/zhangziling/Documents/Project_King_County_Safety_Rating/scripts/run_food_inspection_pipeline.py)

## 2. 常用命令

全量运行:
```bash
python3 /Users/zhangziling/Documents/Project_King_County_Safety_Rating/scripts/run_food_inspection_pipeline.py \
  --root /Users/zhangziling/Documents/Project_King_County_Safety_Rating \
  --mode full \
  --page-size 50000 \
  --fetcher auto
```

增量运行（按 state 自动回看窗口）:
```bash
python3 /Users/zhangziling/Documents/Project_King_County_Safety_Rating/scripts/run_food_inspection_pipeline.py \
  --root /Users/zhangziling/Documents/Project_King_County_Safety_Rating \
  --mode incremental \
  --lookback-days 3 \
  --page-size 50000 \
  --fetcher auto
```

增量运行（手动指定起始日期）:
```bash
python3 /Users/zhangziling/Documents/Project_King_County_Safety_Rating/scripts/run_food_inspection_pipeline.py \
  --root /Users/zhangziling/Documents/Project_King_County_Safety_Rating \
  --mode incremental \
  --since-date 2025-11-20 \
  --page-size 50000 \
  --fetcher auto
```

快速验证（开发调试）:
```bash
python3 /Users/zhangziling/Documents/Project_King_County_Safety_Rating/scripts/run_food_inspection_pipeline.py \
  --root /Users/zhangziling/Documents/Project_King_County_Safety_Rating \
  --mode full \
  --max-rows 5000 \
  --page-size 2000 \
  --fetcher auto
```

说明:
- `--fetcher auto` 会优先用 `urllib`，失败时自动回退 `curl`（适配本机 SSL 证书问题）。
- 可选参数: `--max-pages`, `--timeout-seconds`, `--max-retries`, `--app-token`。
- `--mode incremental` 使用 `inspection_date` 窗口抽取；若无历史 checkpoint，会自动回退为 full。
- 增量模式下若当日无新数据，不会 blocker，只会记录 `incremental_no_new_rows` warning。

每日任务脚本（内置策略: 周日 full，其他天 incremental；若增量失败自动回退 full 一次）:
```bash
/Users/zhangziling/Documents/Project_King_County_Safety_Rating/scripts/run_daily_pipeline.sh
```

cron 示例（每天 06:15 执行）:
```bash
15 6 * * * /Users/zhangziling/Documents/Project_King_County_Safety_Rating/scripts/run_daily_pipeline.sh >> /Users/zhangziling/Documents/Project_King_County_Safety_Rating/outputs/dq/f29f-zza5/cron.log 2>&1
```

## 3. 输出目录

按 `run_id` 分批次写入:
- Bronze: `Data/bronze/f29f-zza5/<run_id>/raw.csv`
- Silver:
  - `Data/silver/f29f-zza5/<run_id>/inspection_row.csv`
  - `Data/silver/f29f-zza5/<run_id>/inspection_event.csv`
- Gold:
  - `Data/gold/f29f-zza5/<run_id>/business_profile.csv`
  - `Data/gold/f29f-zza5/<run_id>/violation_fact.csv`
  - `Data/gold/f29f-zza5/<run_id>/violation_dictionary.csv`
  - `Data/gold/f29f-zza5/<run_id>/dashboard_violation_explained.csv`
- DQ:
  - `outputs/dq/f29f-zza5/<run_id>/dq_report.json`
  - `outputs/dq/f29f-zza5/<run_id>/dq_report.md`

整改字典（全局最新）:
- `Data/reference/violation_dictionary_latest.csv`
- `Data/reference/violation_dictionary_<run_id>.csv`
- `docs/violation_remediation_dictionary.md`

Dashboard 违规解释使用建议:
- 直接读取 `dashboard_violation_explained.csv`（已按 `violation_type + violation_code` 关联整改字典）。
- 字段 `action_source` 含义:
  - `dictionary_code_type`: 成功匹配整改字典
  - `fallback_rule`: 代码未命中字典，使用规则兜底
  - `missing_violation_code`: 原始记录无违规代码，展示通用整改提示

状态文件:
- 最新批次指针: `Data/state/f29f-zza5_latest_run.json`
- 运行历史: `Data/state/f29f-zza5_pipeline_state.json`

## 4. 当前全量结果（已完成）
- Full run_id: `20260225T234039Z`
- DQ 报告: [dq_report.md](/Users/zhangziling/Documents/Project_King_County_Safety_Rating/outputs/dq/f29f-zza5/20260225T234039Z/dq_report.md)
- 关键规模:
  - Raw rows: 278,425
  - Inspection events: 197,312
  - Businesses: 12,874

## 5. DQ 状态解读
- `status=ok`: 无告警/阻断。
- `status=warning`: 存在质量告警，但可用于分析。
- `status=blocker`: 阻断问题，需要先处理再发布。

当前全量批次为 `warning`（无 blocker）。
