# Violation Remediation Dictionary

- Source run_id: `20260225T234301Z`
- Generated at (UTC): `2026-02-25T23:43:01+00:00`
- Dictionary entries (code+type): `66`
- Unique violation codes: `53`

## Fields

- `violation_code`: 违规代码
- `violation_type`: RED / BLUE
- `default_points_mode`: 该代码最常见扣分
- `canonical_description`: 规范化违规描述
- `action_category`: 整改主题类别
- `action_priority`: high / medium / low
- `action_summary_zh`: 面向公众/店主的整改摘要
- `action_steps_zh`: 建议整改动作
- `safe_food_handling_refs`: 对应官方安全操作主题

## Dictionary

| code | type | points | priority | category | summary_zh | refs |
|---|---|---:|---|---|---|---|
| 0100 | RED | 5 | high | management_training | 负责人在岗并落实食品安全管理，员工证照与上岗培训必须完整。 | 13 (Personal hygiene), 1 (Receiving food) |
| 0200 | BLUE | 5 | medium | management_training | 负责人在岗并落实食品安全管理，员工证照与上岗培训必须完整。 | 13 (Personal hygiene), 1 (Receiving food) |
| 0200 | RED | 5 | high | management_training | 负责人在岗并落实食品安全管理，员工证照与上岗培训必须完整。 | 13 (Personal hygiene), 1 (Receiving food) |
| 0300 | RED | 25 | high | employee_health | 建立员工健康排查与隔离制度，杜绝患病员工接触食品。 | 13 (Personal hygiene) |
| 0400 | BLUE | 15 | high | handwashing_and_contact | 确保洗手设施可用并严格执行洗手与避免徒手接触即食食品。 | 13 (Handwashing), 10 (Cross contamination), 12 (Serving food) |
| 0400 | RED | 25 | high | handwashing_and_contact | 确保洗手设施可用并严格执行洗手与避免徒手接触即食食品。 | 13 (Handwashing), 10 (Cross contamination), 12 (Serving food) |
| 0500 | BLUE | 15 | high | handwashing_and_contact | 确保洗手设施可用并严格执行洗手与避免徒手接触即食食品。 | 13 (Handwashing), 10 (Cross contamination), 12 (Serving food) |
| 0500 | RED | 25 | high | handwashing_and_contact | 确保洗手设施可用并严格执行洗手与避免徒手接触即食食品。 | 13 (Handwashing), 10 (Cross contamination), 12 (Serving food) |
| 0600 | BLUE | 10 | high | handwashing_and_contact | 确保洗手设施可用并严格执行洗手与避免徒手接触即食食品。 | 13 (Handwashing), 10 (Cross contamination), 12 (Serving food) |
| 0600 | RED | 10 | high | handwashing_and_contact | 确保洗手设施可用并严格执行洗手与避免徒手接触即食食品。 | 13 (Handwashing), 10 (Cross contamination), 12 (Serving food) |
| 0700 | RED | 15 | high | approved_source_and_food_condition | 食材与水冰必须来自合规来源，并确保食品状态安全可追溯。 | 1 (Receiving food), 2 (Storage) |
| 0800 | RED | 15 | high | approved_source_and_food_condition | 食材与水冰必须来自合规来源，并确保食品状态安全可追溯。 | 1 (Receiving food), 2 (Storage) |
| 0900 | RED | 10 | high | approved_source_and_food_condition | 食材与水冰必须来自合规来源，并确保食品状态安全可追溯。 | 1 (Receiving food), 2 (Storage) |
| 1000 | RED | 10 | high | approved_source_and_food_condition | 食材与水冰必须来自合规来源，并确保食品状态安全可追溯。 | 1 (Receiving food), 2 (Storage) |
| 1100 | RED | 10 | high | approved_source_and_food_condition | 食材与水冰必须来自合规来源，并确保食品状态安全可追溯。 | 1 (Receiving food), 2 (Storage) |
| 1200 | RED | 5 | high | approved_source_and_food_condition | 食材与水冰必须来自合规来源，并确保食品状态安全可追溯。 | 1 (Receiving food), 2 (Storage) |
| 1300 | RED | 15 | high | cross_contamination | 生熟分离、工器具分离、流程分区，降低交叉污染风险。 | 10 (Cross contamination), 2 (Storage), 12 (Serving food) |
| 1400 | RED | 5 | high | cross_contamination | 生熟分离、工器具分离、流程分区，降低交叉污染风险。 | 10 (Cross contamination), 2 (Storage), 12 (Serving food) |
| 1500 | RED | 5 | high | cross_contamination | 生熟分离、工器具分离、流程分区，降低交叉污染风险。 | 10 (Cross contamination), 2 (Storage), 12 (Serving food) |
| 1600 | BLUE | 30 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 1600 | RED | 25 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 1710 | BLUE | 25 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 1710 | RED | 25 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 1720 | RED | 5 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 1800 | RED | 25 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 1900 | BLUE | 25 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 1900 | RED | 25 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 2000 | RED | 15 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 2110 | BLUE | 10 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 2110 | RED | 10 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 2120 | RED | 5 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 2200 | BLUE | 5 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 2200 | RED | 5 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 2300 | RED | 5 | high | consumer_info_and_labeling | 消费者提示、标签和日期标识要准确完整，确保信息透明。 | 11 (Room temperature), 2 (Storage) |
| 2400 | RED | 10 | high | approved_source_and_food_condition | 食材与水冰必须来自合规来源，并确保食品状态安全可追溯。 | 1 (Receiving food), 2 (Storage) |
| 2500 | BLUE | 10 | high | chemical_safety | 化学品分类标识并与食品隔离存放，防止误用和污染。 | 2 (Storage), 10 (Cross contamination) |
| 2500 | RED | 10 | high | chemical_safety | 化学品分类标识并与食品隔离存放，防止误用和污染。 | 2 (Storage), 10 (Cross contamination) |
| 2600 | BLUE | 10 | high | permit_and_haccp | 许可、风险控制计划和 HACCP 文档需持续有效并按现场一致执行。 | Regulatory compliance, 1 (Receiving food), 6 (Cooking) |
| 2600 | RED | 10 | high | permit_and_haccp | 许可、风险控制计划和 HACCP 文档需持续有效并按现场一致执行。 | Regulatory compliance, 1 (Receiving food), 6 (Cooking) |
| 2700 | RED | 10 | high | permit_and_haccp | 许可、风险控制计划和 HACCP 文档需持续有效并按现场一致执行。 | Regulatory compliance, 1 (Receiving food), 6 (Cooking) |
| 2800 | BLUE | 5 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 2900 | BLUE | 5 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 3000 | BLUE | 3 | high | temperature_control | 建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。 | 3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing) |
| 3100 | BLUE | 5 | medium | consumer_info_and_labeling | 消费者提示、标签和日期标识要准确完整，确保信息透明。 | 11 (Room temperature), 2 (Storage) |
| 3200 | BLUE | 5 | medium | pest_control | 落实防虫防鼠与入口封闭措施，阻断害虫进入与繁殖。 | 2 (Storage), 10 (Cross contamination) |
| 3300 | BLUE | 5 | high | cross_contamination | 生熟分离、工器具分离、流程分区，降低交叉污染风险。 | 10 (Cross contamination), 2 (Storage), 12 (Serving food) |
| 3400 | BLUE | 5 | medium | cleaning_and_sanitizing | 设备工器具和接触面按频次清洗消毒，保持有效消毒浓度。 | 10 (Cross contamination), 13 (Personal hygiene) |
| 3400 | RED | 5 | high | cleaning_and_sanitizing | 设备工器具和接触面按频次清洗消毒，保持有效消毒浓度。 | 10 (Cross contamination), 13 (Personal hygiene) |
| 3500 | BLUE | 3 | high | handwashing_and_contact | 确保洗手设施可用并严格执行洗手与避免徒手接触即食食品。 | 13 (Handwashing), 10 (Cross contamination), 12 (Serving food) |
| 3600 | BLUE | 3 | high | handwashing_and_contact | 确保洗手设施可用并严格执行洗手与避免徒手接触即食食品。 | 13 (Handwashing), 10 (Cross contamination), 12 (Serving food) |
| 3700 | BLUE | 3 | medium | utensil_and_single_use_management | 工器具与一次性用品规范存放，防止二次污染。 | 10 (Cross contamination), 12 (Serving food) |
| 3800 | BLUE | 3 | medium | utensil_and_single_use_management | 工器具与一次性用品规范存放，防止二次污染。 | 10 (Cross contamination), 12 (Serving food) |
| 3900 | BLUE | 3 | medium | utensil_and_single_use_management | 工器具与一次性用品规范存放，防止二次污染。 | 10 (Cross contamination), 12 (Serving food) |
| 4000 | BLUE | 5 | medium | cleaning_and_sanitizing | 设备工器具和接触面按频次清洗消毒，保持有效消毒浓度。 | 10 (Cross contamination), 13 (Personal hygiene) |
| 4100 | BLUE | 5 | medium | cleaning_and_sanitizing | 设备工器具和接触面按频次清洗消毒，保持有效消毒浓度。 | 10 (Cross contamination), 13 (Personal hygiene) |
| 4200 | BLUE | 5 | medium | cleaning_and_sanitizing | 设备工器具和接触面按频次清洗消毒，保持有效消毒浓度。 | 10 (Cross contamination), 13 (Personal hygiene) |
| 4200 | RED | 5 | high | cleaning_and_sanitizing | 设备工器具和接触面按频次清洗消毒，保持有效消毒浓度。 | 10 (Cross contamination), 13 (Personal hygiene) |
| 4300 | BLUE | 3 | medium | cleaning_and_sanitizing | 设备工器具和接触面按频次清洗消毒，保持有效消毒浓度。 | 10 (Cross contamination), 13 (Personal hygiene) |
| 4400 | BLUE | 5 | medium | plumbing_waste_toilet | 排水、污水和卫生间设施应完好可用，避免环境性污染。 | 2 (Storage), 13 (Personal hygiene) |
| 4500 | BLUE | 5 | medium | plumbing_waste_toilet | 排水、污水和卫生间设施应完好可用，避免环境性污染。 | 2 (Storage), 13 (Personal hygiene) |
| 4600 | BLUE | 3 | medium | plumbing_waste_toilet | 排水、污水和卫生间设施应完好可用，避免环境性污染。 | 2 (Storage), 13 (Personal hygiene) |
| 4700 | BLUE | 3 | low | facility_and_environment | 完善场地、通风、照明与垃圾管理，保证运营环境卫生。 | 2 (Storage), 10 (Cross contamination) |
| 4800 | BLUE | 2 | low | facility_and_environment | 完善场地、通风、照明与垃圾管理，保证运营环境卫生。 | 2 (Storage), 10 (Cross contamination) |
| 4900 | BLUE | 2 | low | facility_and_environment | 完善场地、通风、照明与垃圾管理，保证运营环境卫生。 | 2 (Storage), 10 (Cross contamination) |
| 5000 | BLUE | 2 | low | posting_and_public_display | 按要求张贴许可和评级标识，确保公众可见且信息最新。 | Regulatory posting requirements |
| 5100 | BLUE | 2 | low | posting_and_public_display | 按要求张贴许可和评级标识，确保公众可见且信息最新。 | Regulatory posting requirements |