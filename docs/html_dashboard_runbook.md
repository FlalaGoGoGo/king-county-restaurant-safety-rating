# HTML Dashboard Runbook

更新时间: 2026-02-26

## 1. 已生成文件
- 可直接打开（双击）:
  - [index.html](/Users/zhangziling/Documents/Project_King_County_Safety_Rating/outputs/dashboard/index.html)

## 2. 直接使用
- 在 Finder 中双击 `outputs/dashboard/index.html` 即可。
- 浏览器建议: Chrome / Edge（大文件加载更稳定）。

## 3. 重新生成（当数据更新后）
- 推荐命令:
```bash
/Users/zhangziling/Documents/Project_King_County_Safety_Rating/scripts/build_html_dashboard.sh
```

- 或手动执行:
```bash
python3 /Users/zhangziling/Documents/Project_King_County_Safety_Rating/scripts/export_html_dashboard.py \
  --root /Users/zhangziling/Documents/Project_King_County_Safety_Rating \
  --output-html /Users/zhangziling/Documents/Project_King_County_Safety_Rating/outputs/dashboard/index.html \
  --max-events-per-business 200 \
  --max-violations-per-business 300
```

## 4. 体积与性能
- 当前 `index.html` 是“单文件内嵌数据”模式，打开非常方便，但文件会较大。
- 若想更轻量，可下调导出参数:
```bash
MAX_EVENTS_PER_BIZ=80 MAX_VIOLATIONS_PER_BIZ=120 \
/Users/zhangziling/Documents/Project_King_County_Safety_Rating/scripts/build_html_dashboard.sh
```

## 5. 与 Streamlit 版本差异
- 同步范围（当前版本）:
  - 顶层结构一致: `项目overview / 寻找餐厅 / 历史数据总结 / 模型预测`
  - `Overview` 一致包含:
    - 官方评级 1-4 含义
    - 项目高风险口径
    - Red/Blue 违规类型说明和高频违规代码
  - `Search` 一致支持:
    - 模糊搜索
    - 最新评级筛选（`Excellent / Good / Okay / Needs to Improve`）
    - 最新评级筛选新增 `Rating not available`
    - 结果字段排序
    - 结果行点击直接选中餐厅（无需二次下拉选择）
    - 左侧 `Inspection` 选择联动右侧当日违规明细
  - `Summary` 一致保留三视角切换: `Consumer / Restaurant Owner / Regulator`
  - `Violations & Remediation` 字段一致为英文，不含 CN/Refs/Source
  - `Predict` 一致改为离线 `manifest` 驱动，可切换不同模型查看对应指标与餐厅预测结果
  - `Predict` 一致展示 MLP 调优结果（grid + top config 图）
  - 在 `Predict` 内包含作业 4 Tab 对应分区（Executive Summary / Descriptive Analytics / Model Performance / Explainability & Interactive Prediction）
- 运行机制差异:
  - HTML 版: 无需启动服务，双击即可打开；预测结果按“模型+餐厅”离线预计算。
  - Streamlit 版: 需 `streamlit run` 启动；支持更多交互（含可编辑输入、自定义 SHAP waterfall 生成）。
