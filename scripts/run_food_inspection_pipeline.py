#!/usr/bin/env python3
"""
Run the King County food inspection data pipeline:
1) Extract raw rows from Socrata API (Bronze)
2) Clean/standardize rows (Silver row-level)
3) Aggregate inspection events and business profile (Silver/Gold)
4) Generate DQ reports (JSON + Markdown)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

DATASET_ID = "f29f-zza5"
SOCRATA_RESOURCE_URL = f"https://data.kingcounty.gov/resource/{DATASET_ID}.csv"

LAT_MIN = 47.0
LAT_MAX = 48.0
LON_MIN = -123.0
LON_MAX = -121.0

SPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^A-Z0-9 ]+")
VIOLATION_CODE_RE = re.compile(r"^\s*(\d{4})\s*-\s*(.+)$")
TRAILING_WA_RE = re.compile(r"[, ]+WA$")
RISK_LEVEL_RE = re.compile(r"RISK\s+CATEGORY\s+(III|II|I)\b", re.IGNORECASE)

SEARCH_STOP_WORDS = {
    "INC",
    "INCORPORATED",
    "LLC",
    "LTD",
    "CO",
    "COMPANY",
    "CORP",
    "CORPORATION",
    "THE",
}

ADDRESS_TOKEN_MAP = {
    "STREET": "ST",
    "AVENUE": "AVE",
    "ROAD": "RD",
    "BOULEVARD": "BLVD",
    "DRIVE": "DR",
    "LANE": "LN",
    "COURT": "CT",
    "PLACE": "PL",
    "NORTH": "N",
    "SOUTH": "S",
    "EAST": "E",
    "WEST": "W",
    "NORTHEAST": "NE",
    "NORTHWEST": "NW",
    "SOUTHEAST": "SE",
    "SOUTHWEST": "SW",
    "HIGHWAY": "HWY",
}

CITY_CORRECTIONS = {
    "SEA TAC": "SEATAC",
    "SEATTLE,": "SEATTLE",
    "SEATLLE": "SEATTLE",
    "SEATTLE WA": "SEATTLE",
    "BELLEUE": "BELLEVUE",
    "TUWKILA": "TUKWILA",
    "LAKE FOREST": "LAKE FOREST PARK",
    "WEST SEATTLE": "SEATTLE",
    "VASHON ISLAND": "VASHON",
}

KING_COUNTY_CITY_SET = {
    "ALGONA",
    "AUBURN",
    "BEAUX ARTS VILLAGE",
    "BELLEVUE",
    "BLACK DIAMOND",
    "BOTHELL",
    "BURIEN",
    "CARNATION",
    "CLYDE HILL",
    "COVINGTON",
    "DES MOINES",
    "DUVALL",
    "ENUMCLAW",
    "FALL CITY",
    "FEDERAL WAY",
    "HOBART",
    "HUNTS POINT",
    "ISSAQUAH",
    "KENMORE",
    "KENT",
    "KIRKLAND",
    "LAKE FOREST PARK",
    "MAPLE VALLEY",
    "MEDINA",
    "MERCER ISLAND",
    "MILTON",
    "NEWCASTLE",
    "NORMANDY PARK",
    "NORTH BEND",
    "PACIFIC",
    "PRESTON",
    "RAVENSDALE",
    "REDMOND",
    "RENTON",
    "SAMMAMISH",
    "SEATTLE",
    "SEATAC",
    "SHORELINE",
    "SKYKOMISH",
    "SNOQUALMIE",
    "SNOQUALMIE PASS",
    "TUKWILA",
    "VASHON",
    "WOODINVILLE",
    "YARROW POINT",
}

GRADE_LABELS = {
    "1": "Excellent",
    "2": "Good",
    "3": "Okay",
    "4": "Needs to Improve",
}
RISK_ROMAN_TO_LEVEL = {"I": "1", "II": "2", "III": "3"}

SILVER_ROW_FIELDS = [
    "row_id",
    "inspection_event_id",
    "inspection_serial_num",
    "business_id",
    "name_raw",
    "program_identifier_raw",
    "inspection_business_name_raw",
    "business_name_official",
    "business_name_alt",
    "search_name_norm",
    "address_raw",
    "full_address_clean",
    "address_norm_key",
    "city_raw",
    "city_canonical",
    "city_norm_reason",
    "city_is_suspect",
    "zip_code",
    "phone",
    "latitude",
    "longitude",
    "geo_out_of_bounds",
    "inspection_date",
    "inspection_year",
    "inspection_month",
    "inspection_week",
    "inspection_type",
    "inspection_result",
    "inspection_score",
    "inspection_score_is_negative",
    "inspection_closed_business",
    "risk_description_raw",
    "risk_level",
    "grade",
    "grade_label",
    "rating_not_available",
    "violation_record_id",
    "violation_type",
    "violation_points",
    "violation_code",
    "violation_desc_raw",
    "violation_desc_clean",
    "violation_desc_truncated",
    "ingest_run_id",
    "ingest_time_utc",
]

SILVER_EVENT_FIELDS = [
    "inspection_event_id",
    "inspection_serial_num",
    "business_id",
    "business_name_official",
    "business_name_alt",
    "search_name_norm",
    "full_address_clean",
    "city_canonical",
    "zip_code",
    "latitude",
    "longitude",
    "inspection_date",
    "inspection_type",
    "inspection_result",
    "inspection_score",
    "inspection_closed_business",
    "risk_description_raw",
    "risk_level",
    "grade",
    "grade_label",
    "rating_not_available",
    "red_points_total",
    "blue_points_total",
    "red_violation_count",
    "blue_violation_count",
    "violation_count_total",
    "source_row_count",
    "generated_from_missing_serial",
]

GOLD_BUSINESS_FIELDS = [
    "business_id",
    "primary_name",
    "alt_names_top3",
    "search_name_norm",
    "latest_inspection_date",
    "latest_risk_level",
    "latest_grade",
    "latest_grade_label",
    "latest_inspection_score",
    "inspection_count",
    "avg_inspection_score",
    "red_points_total_all_time",
    "blue_points_total_all_time",
    "red_points_last_12m",
    "blue_points_last_12m",
    "full_address_clean",
    "city_canonical",
    "zip_code",
    "latitude",
    "longitude",
]

GOLD_VIOLATION_FIELDS = [
    "violation_type",
    "violation_code",
    "violation_desc_clean",
    "occurrences",
    "businesses_affected",
    "points_mode",
]

GOLD_DASHBOARD_VIOLATION_FIELDS = [
    "row_id",
    "inspection_event_id",
    "inspection_serial_num",
    "business_id",
    "business_name_official",
    "business_name_alt",
    "search_name_norm",
    "full_address_clean",
    "city_canonical",
    "zip_code",
    "latitude",
    "longitude",
    "inspection_date",
    "inspection_type",
    "inspection_result",
    "inspection_score",
    "grade",
    "grade_label",
    "violation_type",
    "violation_code",
    "violation_points",
    "violation_desc_clean",
    "violation_desc_raw",
    "dictionary_default_points_mode",
    "dictionary_canonical_description",
    "action_category",
    "action_priority",
    "action_summary_zh",
    "action_steps_zh",
    "action_summary_en",
    "safe_food_handling_refs",
    "action_source",
]

VIOLATION_DICTIONARY_FIELDS = [
    "violation_code",
    "violation_type",
    "default_points_mode",
    "canonical_description",
    "action_category",
    "action_priority",
    "action_summary_zh",
    "action_steps_zh",
    "action_summary_en",
    "safe_food_handling_refs",
    "occurrences",
    "businesses_affected",
    "source_run_id",
    "generated_at_utc",
]

VIOLATION_CATEGORY_ACTIONS = {
    "management_training": {
        "priority_default": "medium",
        "summary_zh": "负责人在岗并落实食品安全管理，员工证照与上岗培训必须完整。",
        "steps_zh": "安排值班负责人全程在岗；逐一核验 Food Worker Card 有效期；新员工入职即完成食品安全培训并留存记录",
        "summary_en": "Ensure person-in-charge oversight and maintain complete worker card/training compliance.",
        "refs": "13 (Personal hygiene), 1 (Receiving food)",
    },
    "employee_health": {
        "priority_default": "high",
        "summary_zh": "建立员工健康排查与隔离制度，杜绝患病员工接触食品。",
        "steps_zh": "班前健康问询并记录；有症状员工立即调离食品岗位；恢复上岗需满足法规要求",
        "summary_en": "Implement illness screening and exclusion policy for food workers.",
        "refs": "13 (Personal hygiene)",
    },
    "handwashing_and_contact": {
        "priority_default": "high",
        "summary_zh": "确保洗手设施可用并严格执行洗手与避免徒手接触即食食品。",
        "steps_zh": "补齐洗手池热水肥皂擦手纸；高风险工位张贴洗手时点；即食食品统一使用夹具手套或隔离介质",
        "summary_en": "Enforce handwashing and no bare-hand contact with ready-to-eat food.",
        "refs": "13 (Handwashing), 10 (Cross contamination), 12 (Serving food)",
    },
    "approved_source_and_food_condition": {
        "priority_default": "high",
        "summary_zh": "食材与水冰必须来自合规来源，并确保食品状态安全可追溯。",
        "steps_zh": "仅向批准供应商采购；保留进货单据和批次信息；发现可疑或回收食品立即隔离报废",
        "summary_en": "Use approved sources and maintain safe, traceable food condition.",
        "refs": "1 (Receiving food), 2 (Storage)",
    },
    "cross_contamination": {
        "priority_default": "high",
        "summary_zh": "生熟分离、工器具分离、流程分区，降低交叉污染风险。",
        "steps_zh": "原料与即食食品分层分区存放；专板专刀专夹并颜色标识；关键转换步骤执行清洗消毒",
        "summary_en": "Prevent cross-contamination through separation and sanitation controls.",
        "refs": "10 (Cross contamination), 2 (Storage), 12 (Serving food)",
    },
    "temperature_control": {
        "priority_default": "high",
        "summary_zh": "建立全流程温控（冷藏、热保温、烹饪、冷却、复热、解冻）与时间控制。",
        "steps_zh": "每班记录冷藏和热保温温度；大批量食品按分浅盘快速冷却；复热用于保温时须先达标后上台；室温暴露时间严格受控",
        "summary_en": "Apply strict time-temperature control across holding, cooking, cooling, reheating, and thawing.",
        "refs": "3/4 (Cold holding), 6 (Cooking), 7 (Reheating), 8 (Hot holding), 9 (Cooling), 11 (Room temperature), 5 (Thawing)",
    },
    "consumer_info_and_labeling": {
        "priority_default": "medium",
        "summary_zh": "消费者提示、标签和日期标识要准确完整，确保信息透明。",
        "steps_zh": "含生食风险菜单张贴 consumer advisory；预包装与在制品按规定标注；冷藏即食食品执行日期管理",
        "summary_en": "Maintain accurate consumer advisories, labels, and date-marking.",
        "refs": "11 (Room temperature), 2 (Storage)",
    },
    "chemical_safety": {
        "priority_default": "high",
        "summary_zh": "化学品分类标识并与食品隔离存放，防止误用和污染。",
        "steps_zh": "化学品容器统一标签；仅在指定区域存放并下置；使用后及时回位并记录",
        "summary_en": "Separate, label, and control toxic substances to prevent contamination.",
        "refs": "2 (Storage), 10 (Cross contamination)",
    },
    "permit_and_haccp": {
        "priority_default": "high",
        "summary_zh": "许可、风险控制计划和 HACCP 文档需持续有效并按现场一致执行。",
        "steps_zh": "核对许可状态与经营范围一致；复杂工艺保留最新版 HACCP/方差批准文件；按计划执行并留存记录",
        "summary_en": "Keep permits, risk control plans, and HACCP procedures valid and implemented.",
        "refs": "Regulatory compliance, 1 (Receiving food), 6 (Cooking)",
    },
    "cleaning_and_sanitizing": {
        "priority_default": "medium",
        "summary_zh": "设备工器具和接触面按频次清洗消毒，保持有效消毒浓度。",
        "steps_zh": "配置并校验消毒液浓度；建立清洗消毒频次表；清洁区与污染区工具分开",
        "summary_en": "Maintain cleaning/sanitizing schedules and proper sanitizer concentration.",
        "refs": "10 (Cross contamination), 13 (Personal hygiene)",
    },
    "pest_control": {
        "priority_default": "medium",
        "summary_zh": "落实防虫防鼠与入口封闭措施，阻断害虫进入与繁殖。",
        "steps_zh": "封堵缝隙并加门底挡；保持干燥无积水和食物残渣；定期巡查并与专业公司联动",
        "summary_en": "Control pests through exclusion, sanitation, and routine monitoring.",
        "refs": "2 (Storage), 10 (Cross contamination)",
    },
    "utensil_and_single_use_management": {
        "priority_default": "medium",
        "summary_zh": "工器具与一次性用品规范存放，防止二次污染。",
        "steps_zh": "一次性用品离地防尘存放；在用工器具按规范放置；回收与清洁流程分离",
        "summary_en": "Store utensils and single-use items properly to prevent contamination.",
        "refs": "10 (Cross contamination), 12 (Serving food)",
    },
    "plumbing_waste_toilet": {
        "priority_default": "medium",
        "summary_zh": "排水、污水和卫生间设施应完好可用，避免环境性污染。",
        "steps_zh": "及时修复漏水和回流风险点；污水排放按规范；卫生间补齐洗手用品并保持清洁",
        "summary_en": "Maintain plumbing, sewage disposal, and toilet facilities in sanitary condition.",
        "refs": "2 (Storage), 13 (Personal hygiene)",
    },
    "facility_and_environment": {
        "priority_default": "low",
        "summary_zh": "完善场地、通风、照明与垃圾管理，保证运营环境卫生。",
        "steps_zh": "修复破损地面墙面天花；确保通风照明达标；垃圾容器加盖并及时清运",
        "summary_en": "Maintain physical facilities, ventilation, lighting, and refuse management.",
        "refs": "2 (Storage), 10 (Cross contamination)",
    },
    "posting_and_public_display": {
        "priority_default": "low",
        "summary_zh": "按要求张贴许可和评级标识，确保公众可见且信息最新。",
        "steps_zh": "将许可与评级牌置于入口明显位置；变更后及时更换；定期巡检张贴状态",
        "summary_en": "Post permits and rating placards visibly and keep displays current.",
        "refs": "Regulatory posting requirements",
    },
    "unknown": {
        "priority_default": "medium",
        "summary_zh": "该违规项需人工复核并补充整改措施。",
        "steps_zh": "核对原始描述和检查报告；联系监管说明；形成现场纠正措施并留档",
        "summary_en": "Manual review required to define corrective actions.",
        "refs": "Manual review",
    },
}

VIOLATION_CODE_CATEGORY = {
    "0100": "management_training",
    "0200": "management_training",
    "0300": "employee_health",
    "0400": "handwashing_and_contact",
    "0500": "handwashing_and_contact",
    "0600": "handwashing_and_contact",
    "0700": "approved_source_and_food_condition",
    "0800": "approved_source_and_food_condition",
    "0900": "approved_source_and_food_condition",
    "1000": "approved_source_and_food_condition",
    "1100": "approved_source_and_food_condition",
    "1200": "approved_source_and_food_condition",
    "1300": "cross_contamination",
    "1400": "cross_contamination",
    "1500": "cross_contamination",
    "1600": "temperature_control",
    "1710": "temperature_control",
    "1720": "temperature_control",
    "1800": "temperature_control",
    "1900": "temperature_control",
    "2000": "temperature_control",
    "2110": "temperature_control",
    "2120": "temperature_control",
    "2200": "temperature_control",
    "2300": "consumer_info_and_labeling",
    "2400": "approved_source_and_food_condition",
    "2500": "chemical_safety",
    "2600": "permit_and_haccp",
    "2700": "permit_and_haccp",
    "2800": "temperature_control",
    "2900": "temperature_control",
    "3000": "temperature_control",
    "3100": "consumer_info_and_labeling",
    "3200": "pest_control",
    "3300": "cross_contamination",
    "3400": "cleaning_and_sanitizing",
    "3500": "handwashing_and_contact",
    "3600": "handwashing_and_contact",
    "3700": "utensil_and_single_use_management",
    "3800": "utensil_and_single_use_management",
    "3900": "utensil_and_single_use_management",
    "4000": "cleaning_and_sanitizing",
    "4100": "cleaning_and_sanitizing",
    "4200": "cleaning_and_sanitizing",
    "4300": "cleaning_and_sanitizing",
    "4400": "plumbing_waste_toilet",
    "4500": "plumbing_waste_toilet",
    "4600": "plumbing_waste_toilet",
    "4700": "facility_and_environment",
    "4800": "facility_and_environment",
    "4900": "facility_and_environment",
    "5000": "posting_and_public_display",
    "5100": "posting_and_public_display",
}


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_utc(ts: Optional[dt.datetime] = None) -> str:
    value = ts or now_utc()
    return value.replace(microsecond=0).isoformat()


def clean_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = value.strip()
    text = SPACE_RE.sub(" ", text)
    return text


def normalize_search_text(value: str) -> str:
    text = clean_text(value).upper()
    text = text.replace("&", " AND ")
    text = NON_ALNUM_RE.sub(" ", text)
    tokens = [token for token in text.split() if token and token not in SEARCH_STOP_WORDS]
    return " ".join(tokens)


def normalize_address(value: str) -> str:
    normalized = normalize_search_text(value)
    tokens = [ADDRESS_TOKEN_MAP.get(token, token) for token in normalized.split()]
    return " ".join(tokens)


def canonicalize_city(raw_city: str) -> Tuple[str, str, int]:
    city = clean_text(raw_city).upper()
    city = city.rstrip(",")
    city = TRAILING_WA_RE.sub("", city).strip()
    city = SPACE_RE.sub(" ", city)
    reason = "exact"
    if city in CITY_CORRECTIONS:
        reason = f"mapped:{city}->{CITY_CORRECTIONS[city]}"
        city = CITY_CORRECTIONS[city]
    if not city:
        reason = "empty"
    is_suspect = 0 if (city and city in KING_COUNTY_CITY_SET) else 1
    return city, reason, is_suspect


def parse_date(value: str) -> Optional[dt.date]:
    raw = clean_text(value)
    if not raw:
        return None
    raw = raw.replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def parse_float(value: str) -> Optional[float]:
    raw = clean_text(value)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_int(value: str) -> Optional[int]:
    raw = clean_text(value)
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def grade_label(grade: str) -> str:
    return GRADE_LABELS.get(clean_text(grade), "Rating not available")


def parse_risk_level(description: str) -> str:
    match = RISK_LEVEL_RE.search(clean_text(description))
    if not match:
        return ""
    return RISK_ROMAN_TO_LEVEL.get(match.group(1).upper(), "")


def to_bool_string(value: str) -> str:
    raw = clean_text(value).lower()
    if raw in {"true", "t", "yes", "y", "1"}:
        return "1"
    if raw in {"false", "f", "no", "n", "0"}:
        return "0"
    return ""


def fetch_csv_pages(
    *,
    page_size: int,
    max_rows: Optional[int],
    max_pages: Optional[int],
    timeout_seconds: int,
    max_retries: int,
    app_token: Optional[str],
    fetcher: str,
    where_clause: str,
) -> Iterator[Tuple[int, List[Dict[str, str]], List[str]]]:
    offset = 0
    fetched_rows = 0
    page_num = 0

    while True:
        params = {"$limit": str(page_size), "$offset": str(offset)}
        if clean_text(where_clause):
            params["$where"] = where_clause
        query = urllib.parse.urlencode(params)
        url = f"{SOCRATA_RESOURCE_URL}?{query}"

        headers = {"User-Agent": "KC-Food-Pipeline/1.0"}
        if app_token:
            headers["X-App-Token"] = app_token

        raw_data = None
        for attempt in range(1, max_retries + 1):
            try:
                if fetcher in {"urllib", "auto"}:
                    try:
                        req = urllib.request.Request(url=url, headers=headers, method="GET")
                        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                            raw_data = resp.read()
                    except Exception:
                        if fetcher == "urllib":
                            raise
                        raw_data = None

                if raw_data is None and fetcher in {"curl", "auto"}:
                    cmd = [
                        "curl",
                        "--silent",
                        "--show-error",
                        "--fail",
                        "--retry",
                        "5",
                        "--retry-delay",
                        "2",
                        "--retry-all-errors",
                        "--connect-timeout",
                        str(timeout_seconds),
                        "--max-time",
                        str(timeout_seconds * 2),
                    ]
                    for key, value in headers.items():
                        cmd.extend(["-H", f"{key}: {value}"])
                    cmd.append(url)
                    proc = subprocess.run(cmd, check=True, capture_output=True)
                    raw_data = proc.stdout

                break
            except (
                urllib.error.URLError,
                TimeoutError,
                subprocess.CalledProcessError,
                FileNotFoundError,
            ) as exc:
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to fetch page at offset={offset}: {exc}") from exc
                sleep_seconds = min(2**attempt, 15)
                print(
                    f"[warn] request failed offset={offset}, attempt={attempt}/{max_retries}, "
                    f"retry in {sleep_seconds}s",
                    file=sys.stderr,
                )
                time.sleep(sleep_seconds)

        if raw_data is None:
            break

        text = raw_data.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        fieldnames = reader.fieldnames or []
        rows = [dict(row) for row in reader]
        page_num += 1

        if not rows:
            break

        if max_rows is not None:
            remaining = max_rows - fetched_rows
            if remaining <= 0:
                break
            if len(rows) > remaining:
                rows = rows[:remaining]

        yield page_num, rows, fieldnames

        fetched_rows += len(rows)
        offset += len(rows)

        if len(rows) < page_size:
            break
        if max_pages is not None and page_num >= max_pages:
            break
        if max_rows is not None and fetched_rows >= max_rows:
            break


def first_nonempty(values: Iterable[str]) -> str:
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def to_iso_date(value: Optional[dt.date]) -> str:
    return value.isoformat() if value else ""


def safe_float_str(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def safe_int_str(value: Optional[int]) -> str:
    return "" if value is None else str(value)


def resolve_extraction_window(args: argparse.Namespace, state_json_path: Path) -> Dict[str, str]:
    mode_requested = args.mode
    mode_effective = mode_requested
    since_date = ""
    where_clause = ""
    note = ""

    if mode_requested == "incremental":
        if args.since_date:
            since_obj = parse_date(args.since_date)
            if since_obj is None:
                raise ValueError(
                    f"Invalid --since-date value: {args.since_date}. Expected YYYY-MM-DD."
                )
            since_date = since_obj.isoformat()
            note = "from --since-date"
        else:
            latest_max_date = ""
            if state_json_path.exists():
                try:
                    state = json.loads(state_json_path.read_text(encoding="utf-8"))
                    latest_run_id = clean_text(str(state.get("latest_run_id", "")))
                    runs = state.get("runs", [])
                    run_by_id = {
                        clean_text(str(run.get("run_id", ""))): run
                        for run in runs
                        if isinstance(run, dict)
                    }
                    if latest_run_id and latest_run_id in run_by_id:
                        latest_max_date = clean_text(str(run_by_id[latest_run_id].get("max_inspection_date", "")))
                    if not latest_max_date:
                        for run in reversed(runs):
                            latest_max_date = clean_text(str(run.get("max_inspection_date", "")))
                            if latest_max_date:
                                break
                except json.JSONDecodeError:
                    latest_max_date = ""

            if latest_max_date:
                latest_date_obj = parse_date(latest_max_date)
                if latest_date_obj:
                    lookback_days = max(0, int(args.lookback_days))
                    since_date_obj = latest_date_obj - dt.timedelta(days=lookback_days)
                    since_date = since_date_obj.isoformat()
                    note = f"from state latest_max_date={latest_max_date}, lookback_days={lookback_days}"

        if since_date:
            where_clause = f"inspection_date >= '{since_date}'"
        else:
            mode_effective = "full"
            note = "fallback to full because no checkpoint date found"

    return {
        "mode_requested": mode_requested,
        "mode_effective": mode_effective,
        "since_date": since_date,
        "where_clause": where_clause,
        "window_note": note,
    }


def resolve_violation_action(code: str) -> Dict[str, str]:
    normalized_code = clean_text(code)
    category = VIOLATION_CODE_CATEGORY.get(normalized_code, "unknown")
    action = VIOLATION_CATEGORY_ACTIONS[category]
    return {
        "action_category": category,
        "action_priority": action["priority_default"],
        "action_summary_zh": action["summary_zh"],
        "action_steps_zh": action["steps_zh"],
        "action_summary_en": action["summary_en"],
        "safe_food_handling_refs": action["refs"],
    }


def build_violation_dictionary_rows(
    violation_map: Dict[Tuple[str, str, str], Dict[str, object]],
    run_id: str,
    generated_at_utc: str,
) -> List[Dict[str, str]]:
    by_code: Dict[Tuple[str, str], Dict[str, object]] = {}
    for (v_type, v_code, v_desc), agg in violation_map.items():
        code = clean_text(v_code)
        if not code:
            continue
        key = (clean_text(v_type), code)
        bucket = by_code.get(key)
        if bucket is None:
            bucket = {
                "desc_counter": Counter(),
                "points_counter": Counter(),
                "occurrences": 0,
                "business_ids": set(),
            }
            by_code[key] = bucket

        bucket["desc_counter"][clean_text(v_desc)] += int(agg["occurrences"])
        bucket["occurrences"] = int(bucket["occurrences"]) + int(agg["occurrences"])
        bucket["business_ids"].update(agg["business_ids"])
        bucket["points_counter"].update(agg["points_counter"])

    rows: List[Dict[str, str]] = []
    for (v_type, v_code) in sorted(by_code.keys(), key=lambda x: (x[1], x[0])):
        bucket = by_code[(v_type, v_code)]
        canonical_description = (
            bucket["desc_counter"].most_common(1)[0][0] if bucket["desc_counter"] else ""
        )
        default_points_mode = (
            str(bucket["points_counter"].most_common(1)[0][0]) if bucket["points_counter"] else ""
        )

        action = resolve_violation_action(v_code)
        # RED violations are generally higher priority for operational follow-up.
        action_priority = action["action_priority"]
        if clean_text(v_type) == "RED" and action_priority != "high":
            action_priority = "high"

        rows.append(
            {
                "violation_code": v_code,
                "violation_type": v_type,
                "default_points_mode": default_points_mode,
                "canonical_description": canonical_description,
                "action_category": action["action_category"],
                "action_priority": action_priority,
                "action_summary_zh": action["action_summary_zh"],
                "action_steps_zh": action["action_steps_zh"],
                "action_summary_en": action["action_summary_en"],
                "safe_food_handling_refs": action["safe_food_handling_refs"],
                "occurrences": str(bucket["occurrences"]),
                "businesses_affected": str(len(bucket["business_ids"])),
                "source_run_id": run_id,
                "generated_at_utc": generated_at_utc,
            }
        )
    return rows


def write_violation_dictionary_markdown(md_path: Path, rows: List[Dict[str, str]], run_id: str, generated_at_utc: str) -> None:
    unique_codes = len({clean_text(row.get("violation_code", "")) for row in rows if clean_text(row.get("violation_code", ""))})
    lines = [
        "# Violation Remediation Dictionary",
        "",
        f"- Source run_id: `{run_id}`",
        f"- Generated at (UTC): `{generated_at_utc}`",
        f"- Dictionary entries (code+type): `{len(rows)}`",
        f"- Unique violation codes: `{unique_codes}`",
        "",
        "## Fields",
        "",
        "- `violation_code`: 违规代码",
        "- `violation_type`: RED / BLUE",
        "- `default_points_mode`: 该代码最常见扣分",
        "- `canonical_description`: 规范化违规描述",
        "- `action_category`: 整改主题类别",
        "- `action_priority`: high / medium / low",
        "- `action_summary_zh`: 面向公众/店主的整改摘要",
        "- `action_steps_zh`: 建议整改动作",
        "- `safe_food_handling_refs`: 对应官方安全操作主题",
        "",
        "## Dictionary",
        "",
        "| code | type | points | priority | category | summary_zh | refs |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda item: (item["violation_code"], item["violation_type"])):
        lines.append(
            "| {code} | {vtype} | {points} | {priority} | {category} | {summary} | {refs} |".format(
                code=row["violation_code"],
                vtype=row["violation_type"],
                points=row["default_points_mode"],
                priority=row["action_priority"],
                category=row["action_category"],
                summary=row["action_summary_zh"].replace("|", "/"),
                refs=row["safe_food_handling_refs"].replace("|", "/"),
            )
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def write_dashboard_violation_explained_csv(
    silver_row_csv: Path,
    output_csv: Path,
    violation_dictionary_rows: List[Dict[str, str]],
) -> Dict[str, int]:
    dictionary_lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in violation_dictionary_rows:
        code = clean_text(row.get("violation_code", ""))
        v_type = clean_text(row.get("violation_type", "")).upper()
        if code and v_type:
            dictionary_lookup[(v_type, code)] = row

    stats = {
        "rows_written": 0,
        "dictionary_matched_count": 0,
        "fallback_rule_count": 0,
        "missing_violation_code_count": 0,
    }

    with silver_row_csv.open("r", encoding="utf-8") as f_input, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as f_output:
        reader = csv.DictReader(f_input)
        writer = csv.DictWriter(f_output, fieldnames=GOLD_DASHBOARD_VIOLATION_FIELDS)
        writer.writeheader()

        for row in reader:
            v_code = clean_text(row.get("violation_code", ""))
            v_type = clean_text(row.get("violation_type", "")).upper()
            dict_row = dictionary_lookup.get((v_type, v_code)) if v_code and v_type else None

            if dict_row:
                action_category = clean_text(dict_row.get("action_category", ""))
                action_priority = clean_text(dict_row.get("action_priority", ""))
                action_summary_zh = clean_text(dict_row.get("action_summary_zh", ""))
                action_steps_zh = clean_text(dict_row.get("action_steps_zh", ""))
                action_summary_en = clean_text(dict_row.get("action_summary_en", ""))
                refs = clean_text(dict_row.get("safe_food_handling_refs", ""))
                dictionary_default_points_mode = clean_text(
                    dict_row.get("default_points_mode", "")
                )
                dictionary_canonical_description = clean_text(
                    dict_row.get("canonical_description", "")
                )
                action_source = "dictionary_code_type"
                stats["dictionary_matched_count"] += 1
            else:
                action = resolve_violation_action(v_code)
                action_category = action["action_category"]
                action_priority = action["action_priority"]
                action_summary_zh = action["action_summary_zh"]
                action_steps_zh = action["action_steps_zh"]
                action_summary_en = action["action_summary_en"]
                refs = action["safe_food_handling_refs"]
                dictionary_default_points_mode = ""
                dictionary_canonical_description = ""
                if v_code:
                    action_source = "fallback_rule"
                    stats["fallback_rule_count"] += 1
                else:
                    action_source = "missing_violation_code"
                    stats["missing_violation_code_count"] += 1

            writer.writerow(
                {
                    "row_id": clean_text(row.get("row_id", "")),
                    "inspection_event_id": clean_text(row.get("inspection_event_id", "")),
                    "inspection_serial_num": clean_text(row.get("inspection_serial_num", "")),
                    "business_id": clean_text(row.get("business_id", "")),
                    "business_name_official": clean_text(row.get("business_name_official", "")),
                    "business_name_alt": clean_text(row.get("business_name_alt", "")),
                    "search_name_norm": clean_text(row.get("search_name_norm", "")),
                    "full_address_clean": clean_text(row.get("full_address_clean", "")),
                    "city_canonical": clean_text(row.get("city_canonical", "")),
                    "zip_code": clean_text(row.get("zip_code", "")),
                    "latitude": clean_text(row.get("latitude", "")),
                    "longitude": clean_text(row.get("longitude", "")),
                    "inspection_date": clean_text(row.get("inspection_date", "")),
                    "inspection_type": clean_text(row.get("inspection_type", "")),
                    "inspection_result": clean_text(row.get("inspection_result", "")),
                    "inspection_score": clean_text(row.get("inspection_score", "")),
                    "grade": clean_text(row.get("grade", "")),
                    "grade_label": clean_text(row.get("grade_label", "")),
                    "violation_type": v_type,
                    "violation_code": v_code,
                    "violation_points": clean_text(row.get("violation_points", "")),
                    "violation_desc_clean": clean_text(row.get("violation_desc_clean", "")),
                    "violation_desc_raw": clean_text(row.get("violation_desc_raw", "")),
                    "dictionary_default_points_mode": dictionary_default_points_mode,
                    "dictionary_canonical_description": dictionary_canonical_description,
                    "action_category": action_category,
                    "action_priority": action_priority,
                    "action_summary_zh": action_summary_zh,
                    "action_steps_zh": action_steps_zh,
                    "action_summary_en": action_summary_en,
                    "safe_food_handling_refs": refs,
                    "action_source": action_source,
                }
            )
            stats["rows_written"] += 1
    return stats


def build_paths(root: Path, run_id: str) -> Dict[str, Path]:
    base = root
    paths = {
        "bronze_dir": base / "Data" / "bronze" / DATASET_ID / run_id,
        "silver_dir": base / "Data" / "silver" / DATASET_ID / run_id,
        "gold_dir": base / "Data" / "gold" / DATASET_ID / run_id,
        "dq_dir": base / "outputs" / "dq" / DATASET_ID / run_id,
        "state_dir": base / "Data" / "state",
        "reference_dir": base / "Data" / "reference",
        "docs_dir": base / "docs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    paths["bronze_raw_csv"] = paths["bronze_dir"] / "raw.csv"
    paths["silver_row_csv"] = paths["silver_dir"] / "inspection_row.csv"
    paths["silver_event_csv"] = paths["silver_dir"] / "inspection_event.csv"
    paths["gold_business_csv"] = paths["gold_dir"] / "business_profile.csv"
    paths["gold_violation_csv"] = paths["gold_dir"] / "violation_fact.csv"
    paths["gold_violation_dictionary_csv"] = paths["gold_dir"] / "violation_dictionary.csv"
    paths["gold_dashboard_violation_csv"] = (
        paths["gold_dir"] / "dashboard_violation_explained.csv"
    )
    paths["dq_json"] = paths["dq_dir"] / "dq_report.json"
    paths["dq_md"] = paths["dq_dir"] / "dq_report.md"
    paths["reference_violation_dictionary_csv"] = (
        paths["reference_dir"] / "violation_dictionary_latest.csv"
    )
    paths["reference_violation_dictionary_run_csv"] = (
        paths["reference_dir"] / f"violation_dictionary_{run_id}.csv"
    )
    paths["docs_violation_dictionary_md"] = (
        paths["docs_dir"] / "violation_remediation_dictionary.md"
    )
    paths["latest_run_json"] = paths["state_dir"] / f"{DATASET_ID}_latest_run.json"
    paths["state_json"] = paths["state_dir"] / f"{DATASET_ID}_pipeline_state.json"
    return paths


def run_pipeline(args: argparse.Namespace) -> int:
    run_started_at = now_utc()
    run_id = run_started_at.strftime("%Y%m%dT%H%M%SZ")
    root = Path(args.root).resolve()
    paths = build_paths(root, run_id)
    ingest_time = iso_utc(run_started_at)
    extraction = resolve_extraction_window(args, paths["state_json"])
    where_clause = extraction["where_clause"]

    print(f"[info] run_id={run_id}")
    print(f"[info] output_root={root}")
    print(
        f"[info] mode={extraction['mode_effective']} "
        f"(requested={extraction['mode_requested']})"
    )
    if extraction["window_note"]:
        print(f"[info] extraction_window={extraction['window_note']}")
    if where_clause:
        print(f"[info] where_clause={where_clause}")
    print(f"[info] extracting from {SOCRATA_RESOURCE_URL}")

    seen_row_ids = set()
    event_map: Dict[str, Dict[str, object]] = {}
    violation_map: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    stats = {
        "pages_fetched": 0,
        "rows_raw_written": 0,
        "rows_after_dedup": 0,
        "rows_dedup_dropped": 0,
        "missing_inspection_serial_num_count": 0,
        "missing_violation_record_id_count": 0,
        "city_case_mismatch_count": 0,
        "city_suspect_count": 0,
        "geo_out_of_bounds_count": 0,
        "violation_desc_truncated_count": 0,
        "invalid_inspection_date_count": 0,
        "negative_inspection_score_count": 0,
        "no_new_rows_count": 0,
    }
    city_suspect_counter = Counter()
    grade_counter_row = Counter()

    bronze_writer: Optional[csv.DictWriter] = None
    silver_writer: Optional[csv.DictWriter] = None

    with paths["bronze_raw_csv"].open("w", newline="", encoding="utf-8") as f_bronze, paths[
        "silver_row_csv"
    ].open("w", newline="", encoding="utf-8") as f_silver:
        silver_writer = csv.DictWriter(f_silver, fieldnames=SILVER_ROW_FIELDS)
        silver_writer.writeheader()

        for page_num, rows, raw_fieldnames in fetch_csv_pages(
            page_size=args.page_size,
            max_rows=args.max_rows,
            max_pages=args.max_pages,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            app_token=args.app_token,
            fetcher=args.fetcher,
            where_clause=where_clause,
        ):
            stats["pages_fetched"] += 1

            if bronze_writer is None:
                bronze_fields = raw_fieldnames or sorted(rows[0].keys())
                bronze_writer = csv.DictWriter(f_bronze, fieldnames=bronze_fields)
                bronze_writer.writeheader()

            for raw_row in rows:
                bronze_writer.writerow(raw_row)
                stats["rows_raw_written"] += 1

                name_raw = clean_text(raw_row.get("name"))
                program_identifier_raw = clean_text(raw_row.get("program_identifier"))
                inspection_business_name_raw = clean_text(raw_row.get("inspection_business_name"))
                risk_description_raw = clean_text(raw_row.get("description"))
                address_raw = clean_text(raw_row.get("address"))
                city_raw = clean_text(raw_row.get("city"))
                zip_code = clean_text(raw_row.get("zip_code"))
                phone = clean_text(raw_row.get("phone"))
                business_id = clean_text(raw_row.get("business_id"))
                inspection_serial_num = clean_text(raw_row.get("inspection_serial_num"))
                violation_record_id = clean_text(raw_row.get("violation_record_id"))
                violation_type = clean_text(raw_row.get("violation_type")).upper()
                violation_desc_raw = clean_text(raw_row.get("violation_description"))
                violation_points_int = parse_int(raw_row.get("violation_points"))
                inspection_type = clean_text(raw_row.get("inspection_type"))
                inspection_result = clean_text(raw_row.get("inspection_result"))
                inspection_score = parse_float(raw_row.get("inspection_score"))
                inspection_closed_business = to_bool_string(raw_row.get("inspection_closed_business"))
                grade = clean_text(raw_row.get("grade"))
                risk_level = parse_risk_level(risk_description_raw)
                lat = parse_float(raw_row.get("latitude"))
                lon = parse_float(raw_row.get("longitude"))
                inspection_date_obj = parse_date(raw_row.get("inspection_date"))

                if city_raw and city_raw != city_raw.upper():
                    stats["city_case_mismatch_count"] += 1
                city_canonical, city_norm_reason, city_is_suspect = canonicalize_city(city_raw)
                if city_is_suspect:
                    stats["city_suspect_count"] += 1
                    city_suspect_counter[city_canonical or "__EMPTY__"] += 1

                full_address_clean = clean_text(
                    " ".join(part for part in [address_raw, city_canonical, zip_code] if part)
                )
                address_norm_key = normalize_address(address_raw)

                business_name_official = first_nonempty(
                    [name_raw, program_identifier_raw, inspection_business_name_raw]
                )
                business_name_alt = first_nonempty(
                    [
                        program_identifier_raw
                        if program_identifier_raw != business_name_official
                        else "",
                        name_raw if name_raw != business_name_official else "",
                    ]
                )
                search_name_norm = normalize_search_text(
                    " ".join(
                        part for part in [business_name_official, business_name_alt] if part
                    )
                )

                if inspection_serial_num:
                    inspection_event_id = inspection_serial_num
                    generated_from_missing_serial = "0"
                else:
                    stats["missing_inspection_serial_num_count"] += 1
                    event_seed = "|".join(
                        [
                            business_id,
                            to_iso_date(inspection_date_obj),
                            inspection_type,
                            inspection_result,
                            safe_float_str(inspection_score),
                        ]
                    )
                    inspection_event_id = f"EVT_{sha1_text(event_seed)}"
                    generated_from_missing_serial = "1"

                if violation_record_id:
                    row_id = violation_record_id
                else:
                    stats["missing_violation_record_id_count"] += 1
                    row_seed = "|".join(
                        [
                            inspection_event_id,
                            violation_type,
                            violation_desc_raw,
                            safe_int_str(violation_points_int),
                        ]
                    )
                    row_id = f"ROW_{sha1_text(row_seed)}"

                if row_id in seen_row_ids:
                    stats["rows_dedup_dropped"] += 1
                    continue
                seen_row_ids.add(row_id)

                if inspection_date_obj is None:
                    stats["invalid_inspection_date_count"] += 1

                geo_out_of_bounds = 0
                if lat is not None and lon is not None:
                    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
                        geo_out_of_bounds = 1
                        stats["geo_out_of_bounds_count"] += 1

                if inspection_score is not None and inspection_score < 0:
                    stats["negative_inspection_score_count"] += 1
                    inspection_score_is_negative = "1"
                else:
                    inspection_score_is_negative = "0"

                parsed_match = VIOLATION_CODE_RE.match(violation_desc_raw)
                violation_code = parsed_match.group(1) if parsed_match else ""
                violation_desc_clean = clean_text(parsed_match.group(2) if parsed_match else violation_desc_raw)
                violation_desc_truncated = "1" if "..." in violation_desc_raw else "0"
                if violation_desc_truncated == "1":
                    stats["violation_desc_truncated_count"] += 1

                row_grade_label = grade_label(grade)
                if clean_text(grade):
                    grade_counter_row[clean_text(grade)] += 1
                else:
                    grade_counter_row["__NULL__"] += 1

                clean_row = {
                    "row_id": row_id,
                    "inspection_event_id": inspection_event_id,
                    "inspection_serial_num": inspection_serial_num,
                    "business_id": business_id,
                    "name_raw": name_raw,
                    "program_identifier_raw": program_identifier_raw,
                    "inspection_business_name_raw": inspection_business_name_raw,
                    "business_name_official": business_name_official,
                    "business_name_alt": business_name_alt,
                    "search_name_norm": search_name_norm,
                    "address_raw": address_raw,
                    "full_address_clean": full_address_clean,
                    "address_norm_key": address_norm_key,
                    "city_raw": city_raw,
                    "city_canonical": city_canonical,
                    "city_norm_reason": city_norm_reason,
                    "city_is_suspect": str(city_is_suspect),
                    "zip_code": zip_code,
                    "phone": phone,
                    "latitude": safe_float_str(lat),
                    "longitude": safe_float_str(lon),
                    "geo_out_of_bounds": str(geo_out_of_bounds),
                    "inspection_date": to_iso_date(inspection_date_obj),
                    "inspection_year": str(inspection_date_obj.year) if inspection_date_obj else "",
                    "inspection_month": f"{inspection_date_obj.month:02d}" if inspection_date_obj else "",
                    "inspection_week": f"{inspection_date_obj.isocalendar().week:02d}" if inspection_date_obj else "",
                    "inspection_type": inspection_type,
                    "inspection_result": inspection_result,
                    "inspection_score": safe_float_str(inspection_score),
                    "inspection_score_is_negative": inspection_score_is_negative,
                    "inspection_closed_business": inspection_closed_business,
                    "risk_description_raw": risk_description_raw,
                    "risk_level": risk_level,
                    "grade": grade,
                    "grade_label": row_grade_label,
                    "rating_not_available": "0" if clean_text(grade) else "1",
                    "violation_record_id": violation_record_id,
                    "violation_type": violation_type,
                    "violation_points": safe_int_str(violation_points_int),
                    "violation_code": violation_code,
                    "violation_desc_raw": violation_desc_raw,
                    "violation_desc_clean": violation_desc_clean,
                    "violation_desc_truncated": violation_desc_truncated,
                    "ingest_run_id": run_id,
                    "ingest_time_utc": ingest_time,
                }
                silver_writer.writerow(clean_row)
                stats["rows_after_dedup"] += 1

                event = event_map.get(inspection_event_id)
                if event is None:
                    event = {
                        "inspection_event_id": inspection_event_id,
                        "inspection_serial_num": inspection_serial_num,
                        "business_id": business_id,
                        "business_name_official": business_name_official,
                        "business_name_alt": business_name_alt,
                        "search_name_norm": search_name_norm,
                        "full_address_clean": full_address_clean,
                        "city_canonical": city_canonical,
                        "zip_code": zip_code,
                        "latitude": safe_float_str(lat),
                        "longitude": safe_float_str(lon),
                        "inspection_date": to_iso_date(inspection_date_obj),
                        "inspection_type": inspection_type,
                        "inspection_result": inspection_result,
                        "inspection_score": safe_float_str(inspection_score),
                        "inspection_closed_business": inspection_closed_business,
                        "risk_description_raw": risk_description_raw,
                        "risk_level": risk_level,
                        "grade": grade,
                        "grade_label": row_grade_label,
                        "rating_not_available": "0" if clean_text(grade) else "1",
                        "red_points_total": 0,
                        "blue_points_total": 0,
                        "red_violation_count": 0,
                        "blue_violation_count": 0,
                        "violation_count_total": 0,
                        "source_row_count": 0,
                        "generated_from_missing_serial": generated_from_missing_serial,
                    }
                    event_map[inspection_event_id] = event

                for key in [
                    "business_name_official",
                    "business_name_alt",
                    "search_name_norm",
                    "full_address_clean",
                    "city_canonical",
                    "zip_code",
                    "inspection_date",
                    "inspection_type",
                    "inspection_result",
                    "inspection_score",
                    "risk_description_raw",
                    "risk_level",
                    "grade",
                    "grade_label",
                    "latitude",
                    "longitude",
                ]:
                    if not clean_text(str(event.get(key, ""))) and clean_text(clean_row.get(key, "")):
                        event[key] = clean_row[key]

                event["source_row_count"] = int(event["source_row_count"]) + 1
                if violation_type == "RED":
                    event["red_violation_count"] = int(event["red_violation_count"]) + 1
                    event["red_points_total"] = int(event["red_points_total"]) + (violation_points_int or 0)
                    event["violation_count_total"] = int(event["violation_count_total"]) + 1
                elif violation_type == "BLUE":
                    event["blue_violation_count"] = int(event["blue_violation_count"]) + 1
                    event["blue_points_total"] = int(event["blue_points_total"]) + (violation_points_int or 0)
                    event["violation_count_total"] = int(event["violation_count_total"]) + 1

                if violation_type:
                    key = (violation_type, violation_code, violation_desc_clean)
                    agg = violation_map.get(key)
                    if agg is None:
                        agg = {
                            "occurrences": 0,
                            "business_ids": set(),
                            "points_counter": Counter(),
                        }
                        violation_map[key] = agg
                    agg["occurrences"] = int(agg["occurrences"]) + 1
                    if business_id:
                        agg["business_ids"].add(business_id)
                    if violation_points_int is not None:
                        agg["points_counter"][violation_points_int] += 1

            print(
                f"[info] page={page_num} raw_rows={len(rows)} cumulative_raw={stats['rows_raw_written']}",
                flush=True,
            )

    events = list(event_map.values())
    events.sort(key=lambda item: (item.get("inspection_date", ""), item.get("inspection_event_id", "")))

    with paths["silver_event_csv"].open("w", newline="", encoding="utf-8") as f_event:
        writer = csv.DictWriter(f_event, fieldnames=SILVER_EVENT_FIELDS)
        writer.writeheader()
        for event in events:
            row = dict(event)
            for metric_key in [
                "red_points_total",
                "blue_points_total",
                "red_violation_count",
                "blue_violation_count",
                "violation_count_total",
                "source_row_count",
            ]:
                row[metric_key] = str(int(row[metric_key]))
            writer.writerow(row)

    event_grade_counter = Counter()
    event_dates: List[dt.date] = []
    for event in events:
        grade = clean_text(str(event.get("grade", "")))
        event_grade_counter[grade if grade else "__NULL__"] += 1
        event_date = parse_date(str(event.get("inspection_date", "")))
        if event_date:
            event_dates.append(event_date)

    max_event_date = max(event_dates) if event_dates else None
    min_event_date = min(event_dates) if event_dates else None
    today = dt.date.today()
    data_freshness_days = (today - max_event_date).days if max_event_date else None

    # Build business profile.
    business_map: Dict[str, Dict[str, object]] = {}
    latest_12m_cutoff = (
        max_event_date - dt.timedelta(days=365) if max_event_date is not None else dt.date.min
    )
    for event in events:
        business_id = clean_text(str(event.get("business_id", "")))
        if not business_id:
            continue
        profile = business_map.get(business_id)
        if profile is None:
            profile = {
                "business_id": business_id,
                "name_counter": Counter(),
                "alt_name_counter": Counter(),
                "search_name_norm": clean_text(str(event.get("search_name_norm", ""))),
                "latest_inspection_date": "",
                "latest_risk_level": "",
                "latest_grade": "",
                "latest_grade_label": "",
                "latest_inspection_score": "",
                "inspection_count": 0,
                "inspection_score_sum": 0.0,
                "inspection_score_count": 0,
                "red_points_total_all_time": 0,
                "blue_points_total_all_time": 0,
                "red_points_last_12m": 0,
                "blue_points_last_12m": 0,
                "full_address_clean": clean_text(str(event.get("full_address_clean", ""))),
                "city_canonical": clean_text(str(event.get("city_canonical", ""))),
                "zip_code": clean_text(str(event.get("zip_code", ""))),
                "latitude": clean_text(str(event.get("latitude", ""))),
                "longitude": clean_text(str(event.get("longitude", ""))),
            }
            business_map[business_id] = profile

        primary_name = clean_text(str(event.get("business_name_official", "")))
        if primary_name:
            profile["name_counter"][primary_name] += 1
        alt_name = clean_text(str(event.get("business_name_alt", "")))
        if alt_name:
            profile["alt_name_counter"][alt_name] += 1

            profile["inspection_count"] = int(profile["inspection_count"]) + 1
        red_points = int(event.get("red_points_total", 0))
        blue_points = int(event.get("blue_points_total", 0))
        profile["red_points_total_all_time"] = int(profile["red_points_total_all_time"]) + red_points
        profile["blue_points_total_all_time"] = int(profile["blue_points_total_all_time"]) + blue_points

        inspection_date = parse_date(clean_text(str(event.get("inspection_date", ""))))
        if inspection_date and inspection_date >= latest_12m_cutoff:
            profile["red_points_last_12m"] = int(profile["red_points_last_12m"]) + red_points
            profile["blue_points_last_12m"] = int(profile["blue_points_last_12m"]) + blue_points

        score = parse_float(clean_text(str(event.get("inspection_score", ""))))
        if score is not None:
            profile["inspection_score_sum"] = float(profile["inspection_score_sum"]) + score
            profile["inspection_score_count"] = int(profile["inspection_score_count"]) + 1

        current_latest = parse_date(clean_text(str(profile["latest_inspection_date"])))
        if inspection_date and (current_latest is None or inspection_date > current_latest):
            profile["latest_inspection_date"] = inspection_date.isoformat()
            profile["latest_risk_level"] = clean_text(str(event.get("risk_level", "")))
            profile["latest_grade"] = clean_text(str(event.get("grade", "")))
            profile["latest_grade_label"] = clean_text(str(event.get("grade_label", "")))
            profile["latest_inspection_score"] = clean_text(str(event.get("inspection_score", "")))
            profile["full_address_clean"] = clean_text(str(event.get("full_address_clean", "")))
            profile["city_canonical"] = clean_text(str(event.get("city_canonical", "")))
            profile["zip_code"] = clean_text(str(event.get("zip_code", "")))
            profile["latitude"] = clean_text(str(event.get("latitude", "")))
            profile["longitude"] = clean_text(str(event.get("longitude", "")))

    with paths["gold_business_csv"].open("w", newline="", encoding="utf-8") as f_business:
        writer = csv.DictWriter(f_business, fieldnames=GOLD_BUSINESS_FIELDS)
        writer.writeheader()
        for business_id in sorted(business_map.keys()):
            profile = business_map[business_id]
            name_counter: Counter = profile["name_counter"]
            alt_name_counter: Counter = profile["alt_name_counter"]
            primary_name = name_counter.most_common(1)[0][0] if name_counter else ""
            alt_names = [item[0] for item in alt_name_counter.most_common(3)]
            score_count = int(profile["inspection_score_count"])
            score_avg = (
                f"{float(profile['inspection_score_sum']) / score_count:.4f}" if score_count > 0 else ""
            )
            writer.writerow(
                {
                    "business_id": business_id,
                    "primary_name": primary_name,
                    "alt_names_top3": " | ".join(alt_names),
                    "search_name_norm": profile["search_name_norm"],
                    "latest_inspection_date": profile["latest_inspection_date"],
                    "latest_risk_level": profile["latest_risk_level"],
                    "latest_grade": profile["latest_grade"],
                    "latest_grade_label": profile["latest_grade_label"],
                    "latest_inspection_score": profile["latest_inspection_score"],
                    "inspection_count": str(profile["inspection_count"]),
                    "avg_inspection_score": score_avg,
                    "red_points_total_all_time": str(profile["red_points_total_all_time"]),
                    "blue_points_total_all_time": str(profile["blue_points_total_all_time"]),
                    "red_points_last_12m": str(profile["red_points_last_12m"]),
                    "blue_points_last_12m": str(profile["blue_points_last_12m"]),
                    "full_address_clean": profile["full_address_clean"],
                    "city_canonical": profile["city_canonical"],
                    "zip_code": profile["zip_code"],
                    "latitude": profile["latitude"],
                    "longitude": profile["longitude"],
                }
            )

    with paths["gold_violation_csv"].open("w", newline="", encoding="utf-8") as f_violation:
        writer = csv.DictWriter(f_violation, fieldnames=GOLD_VIOLATION_FIELDS)
        writer.writeheader()
        for (v_type, v_code, v_desc) in sorted(violation_map.keys()):
            agg = violation_map[(v_type, v_code, v_desc)]
            points_counter: Counter = agg["points_counter"]
            points_mode = points_counter.most_common(1)[0][0] if points_counter else ""
            writer.writerow(
                {
                    "violation_type": v_type,
                    "violation_code": v_code,
                    "violation_desc_clean": v_desc,
                    "occurrences": str(agg["occurrences"]),
                    "businesses_affected": str(len(agg["business_ids"])),
                    "points_mode": str(points_mode) if points_mode != "" else "",
                }
            )

    violation_dictionary_rows = build_violation_dictionary_rows(
        violation_map=violation_map,
        run_id=run_id,
        generated_at_utc=ingest_time,
    )
    if (
        extraction["mode_effective"] == "incremental"
        and paths["reference_violation_dictionary_csv"].exists()
    ):
        merged = {}
        with paths["reference_violation_dictionary_csv"].open("r", encoding="utf-8") as f_prev:
            reader = csv.DictReader(f_prev)
            for row in reader:
                key = (clean_text(row.get("violation_code", "")), clean_text(row.get("violation_type", "")))
                if key[0]:
                    merged[key] = row
        for row in violation_dictionary_rows:
            key = (row["violation_code"], row["violation_type"])
            merged[key] = row
        violation_dictionary_rows = [
            merged[key]
            for key in sorted(merged.keys(), key=lambda item: (item[0], item[1]))
        ]

    for dict_path in [
        paths["gold_violation_dictionary_csv"],
        paths["reference_violation_dictionary_csv"],
        paths["reference_violation_dictionary_run_csv"],
    ]:
        with dict_path.open("w", newline="", encoding="utf-8") as f_dict:
            writer = csv.DictWriter(f_dict, fieldnames=VIOLATION_DICTIONARY_FIELDS)
            writer.writeheader()
            for row in violation_dictionary_rows:
                writer.writerow(row)

    write_violation_dictionary_markdown(
        md_path=paths["docs_violation_dictionary_md"],
        rows=violation_dictionary_rows,
        run_id=run_id,
        generated_at_utc=ingest_time,
    )

    dashboard_violation_stats = write_dashboard_violation_explained_csv(
        silver_row_csv=paths["silver_row_csv"],
        output_csv=paths["gold_dashboard_violation_csv"],
        violation_dictionary_rows=violation_dictionary_rows,
    )

    def distribution_for(events_subset: List[Dict[str, object]]) -> Dict[str, float]:
        keys = ["1", "2", "3", "4", "__NULL__"]
        counter = Counter()
        for event in events_subset:
            grade = clean_text(str(event.get("grade", "")))
            counter[grade if grade else "__NULL__"] += 1
        total = sum(counter.values())
        if total == 0:
            return {key: 0.0 for key in keys}
        return {key: counter[key] / total for key in keys}

    recent_90_distribution = None
    prior_90_distribution = None
    grade_tvd_90d = None
    if max_event_date is not None:
        recent_start = max_event_date - dt.timedelta(days=89)
        prior_start = recent_start - dt.timedelta(days=90)
        recent_events = [
            event
            for event in events
            if (parse_date(clean_text(str(event.get("inspection_date", ""))) or "") is not None)
            and parse_date(clean_text(str(event.get("inspection_date", "")))) >= recent_start
        ]
        prior_events = [
            event
            for event in events
            if (parse_date(clean_text(str(event.get("inspection_date", ""))) or "") is not None)
            and prior_start <= parse_date(clean_text(str(event.get("inspection_date", "")))) < recent_start
        ]
        if recent_events and prior_events:
            recent_90_distribution = distribution_for(recent_events)
            prior_90_distribution = distribution_for(prior_events)
            grade_tvd_90d = (
                0.5
                * sum(
                    abs(recent_90_distribution[key] - prior_90_distribution[key])
                    for key in recent_90_distribution.keys()
                )
            )

    rows_after_dedup = int(stats["rows_after_dedup"])
    rows_raw_written = int(stats["rows_raw_written"])
    inspection_event_count = len(events)
    business_count = len(business_map)
    violation_fact_count = len(violation_map)
    violation_dictionary_entry_count = len(violation_dictionary_rows)
    violation_dictionary_unique_code_count = len(
        {clean_text(row.get("violation_code", "")) for row in violation_dictionary_rows if clean_text(row.get("violation_code", ""))}
    )
    dashboard_violation_row_count = int(dashboard_violation_stats["rows_written"])
    dashboard_violation_dictionary_matched_count = int(
        dashboard_violation_stats["dictionary_matched_count"]
    )
    dashboard_violation_fallback_rule_count = int(
        dashboard_violation_stats["fallback_rule_count"]
    )
    dashboard_violation_missing_code_count = int(
        dashboard_violation_stats["missing_violation_code_count"]
    )

    def rate(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    missing_serial_rate = rate(int(stats["missing_inspection_serial_num_count"]), rows_after_dedup)
    missing_violation_record_rate = rate(int(stats["missing_violation_record_id_count"]), rows_after_dedup)
    city_suspect_rate = rate(int(stats["city_suspect_count"]), rows_after_dedup)
    truncated_desc_rate = rate(int(stats["violation_desc_truncated_count"]), rows_after_dedup)
    invalid_date_rate = rate(int(stats["invalid_inspection_date_count"]), rows_after_dedup)
    grade_null_rate_event = rate(event_grade_counter["__NULL__"], inspection_event_count)
    dashboard_violation_dictionary_match_rate = rate(
        dashboard_violation_dictionary_matched_count,
        dashboard_violation_row_count,
    )

    blockers = []
    warnings = []
    no_new_rows = rows_after_dedup == 0

    if no_new_rows:
        stats["no_new_rows_count"] = 1
        if extraction["mode_effective"] == "incremental":
            warnings.append(
                {
                    "code": "incremental_no_new_rows",
                    "message": "Incremental window returned zero new rows.",
                    "value": 0,
                }
            )
        else:
            blockers.append(
                {
                    "code": "no_rows_after_dedup",
                    "message": "No rows available after deduplication.",
                    "value": 0,
                }
            )
    if missing_serial_rate > 0.05:
        blockers.append(
            {
                "code": "missing_inspection_serial_rate_high",
                "message": "Missing inspection_serial_num rate exceeds threshold.",
                "value": missing_serial_rate,
                "threshold": 0.05,
            }
        )
    if invalid_date_rate > 0.01:
        blockers.append(
            {
                "code": "invalid_inspection_date_rate_high",
                "message": "Invalid inspection_date parse rate exceeds threshold.",
                "value": invalid_date_rate,
                "threshold": 0.01,
            }
        )

    if city_suspect_rate > 0.005:
        warnings.append(
            {
                "code": "city_suspect_rate_high",
                "message": "Suspect city value rate exceeds warning threshold.",
                "value": city_suspect_rate,
                "threshold": 0.005,
            }
        )
    if int(stats["geo_out_of_bounds_count"]) > 0:
        warnings.append(
            {
                "code": "geo_out_of_bounds_present",
                "message": "Rows found with lat/lon outside expected King County bounds.",
                "value": int(stats["geo_out_of_bounds_count"]),
                "threshold": 0,
            }
        )
    if truncated_desc_rate > 0.05:
        warnings.append(
            {
                "code": "violation_desc_truncated_rate_high",
                "message": "Truncated violation descriptions are frequent.",
                "value": truncated_desc_rate,
                "threshold": 0.05,
            }
        )
    if data_freshness_days is not None and data_freshness_days > 30:
        warnings.append(
            {
                "code": "source_data_stale",
                "message": "Most recent inspection_date is older than 30 days.",
                "value": data_freshness_days,
                "threshold": 30,
            }
        )
    if grade_null_rate_event > 0.10:
        warnings.append(
            {
                "code": "grade_null_rate_high",
                "message": "Event-level null grade rate exceeds warning threshold.",
                "value": grade_null_rate_event,
                "threshold": 0.10,
            }
        )

    dq_report = {
        "run_id": run_id,
        "generated_at_utc": iso_utc(),
        "dataset_id": DATASET_ID,
        "source_url": SOCRATA_RESOURCE_URL,
        "extraction": extraction,
        "paths": {
            key: str(path)
            for key, path in paths.items()
            if isinstance(path, Path) and path.suffix in {".csv", ".json", ".md"}
        },
        "counts": {
            "rows_raw_written": rows_raw_written,
            "rows_after_dedup": rows_after_dedup,
            "inspection_event_count": inspection_event_count,
            "business_count": business_count,
            "violation_fact_count": violation_fact_count,
            "violation_dictionary_entry_count": violation_dictionary_entry_count,
            "violation_dictionary_unique_code_count": violation_dictionary_unique_code_count,
            "dashboard_violation_row_count": dashboard_violation_row_count,
            "dashboard_violation_dictionary_matched_count": dashboard_violation_dictionary_matched_count,
            "dashboard_violation_fallback_rule_count": dashboard_violation_fallback_rule_count,
            "dashboard_violation_missing_code_count": dashboard_violation_missing_code_count,
        },
        "metrics": {
            **stats,
            "missing_inspection_serial_num_rate": missing_serial_rate,
            "missing_violation_record_id_rate": missing_violation_record_rate,
            "city_suspect_rate": city_suspect_rate,
            "violation_desc_truncated_rate": truncated_desc_rate,
            "invalid_inspection_date_rate": invalid_date_rate,
            "grade_null_rate_event_level": grade_null_rate_event,
            "dashboard_violation_dictionary_match_rate": dashboard_violation_dictionary_match_rate,
            "min_inspection_date": to_iso_date(min_event_date),
            "max_inspection_date": to_iso_date(max_event_date),
            "data_freshness_days": data_freshness_days,
        },
        "grade_distribution_event_level": dict(event_grade_counter),
        "grade_distribution_row_level": dict(grade_counter_row),
        "grade_distribution_recent_90d": recent_90_distribution,
        "grade_distribution_prior_90d": prior_90_distribution,
        "grade_tvd_90d": grade_tvd_90d,
        "city_suspect_top20": city_suspect_counter.most_common(20),
        "issues": {
            "blockers": blockers,
            "warnings": warnings,
        },
        "status": "blocker" if blockers else ("warning" if warnings else "ok"),
    }

    with paths["dq_json"].open("w", encoding="utf-8") as f_json:
        json.dump(dq_report, f_json, ensure_ascii=False, indent=2)

    md_lines = [
        f"# DQ Report - {DATASET_ID}",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated (UTC): `{dq_report['generated_at_utc']}`",
        f"- Mode: `{extraction['mode_effective']}` (requested: `{extraction['mode_requested']}`)",
        f"- Source: `{SOCRATA_RESOURCE_URL}`",
        "",
        "## Summary Counts",
        "",
        f"- Raw rows written: `{rows_raw_written:,}`",
        f"- Rows after dedup: `{rows_after_dedup:,}`",
        f"- Inspection events: `{inspection_event_count:,}`",
        f"- Businesses: `{business_count:,}`",
        f"- Violation facts: `{violation_fact_count:,}`",
        f"- Violation dictionary entries (code+type): `{violation_dictionary_entry_count:,}`",
        f"- Violation dictionary unique codes: `{violation_dictionary_unique_code_count:,}`",
        f"- Dashboard violation rows: `{dashboard_violation_row_count:,}`",
        f"- Dashboard dictionary matched rows: `{dashboard_violation_dictionary_matched_count:,}`",
        f"- Dashboard fallback-rule rows: `{dashboard_violation_fallback_rule_count:,}`",
        f"- Dashboard missing-code rows: `{dashboard_violation_missing_code_count:,}`",
        f"- Extraction where clause: `{where_clause or 'N/A'}`",
        "",
        "## Key Metrics",
        "",
        f"- Missing `inspection_serial_num` rate: `{missing_serial_rate:.4%}`",
        f"- Missing `violation_record_id` rate: `{missing_violation_record_rate:.4%}`",
        f"- Suspect city rate: `{city_suspect_rate:.4%}`",
        f"- Truncated violation description rate: `{truncated_desc_rate:.4%}`",
        f"- Invalid inspection_date rate: `{invalid_date_rate:.4%}`",
        f"- Event-level null grade rate: `{grade_null_rate_event:.4%}`",
        f"- Dashboard dictionary match rate: `{dashboard_violation_dictionary_match_rate:.4%}`",
        f"- Inspection date range: `{to_iso_date(min_event_date)} -> {to_iso_date(max_event_date)}`",
        f"- Data freshness (days): `{data_freshness_days}`",
        "",
        "## Issues",
        "",
    ]

    if blockers:
        md_lines.append("### Blockers")
        for issue in blockers:
            md_lines.append(f"- `{issue['code']}`: {issue['message']} (value=`{issue['value']}`)")
    else:
        md_lines.append("- No blocker issues.")

    md_lines.append("")
    if warnings:
        md_lines.append("### Warnings")
        for issue in warnings:
            threshold = issue.get("threshold")
            if threshold is None:
                md_lines.append(f"- `{issue['code']}`: {issue['message']} (value=`{issue['value']}`)")
            else:
                md_lines.append(
                    f"- `{issue['code']}`: {issue['message']} "
                    f"(value=`{issue['value']}`, threshold=`{threshold}`)"
                )
    else:
        md_lines.append("- No warning issues.")

    md_lines.extend(
        [
            "",
            "## Top Suspect Cities",
            "",
        ]
    )
    if city_suspect_counter:
        for city, count in city_suspect_counter.most_common(20):
            md_lines.append(f"- `{city}`: `{count}`")
    else:
        md_lines.append("- None")

    md_lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Bronze raw: `{paths['bronze_raw_csv']}`",
            f"- Silver row: `{paths['silver_row_csv']}`",
            f"- Silver event: `{paths['silver_event_csv']}`",
            f"- Gold business profile: `{paths['gold_business_csv']}`",
            f"- Gold violation fact: `{paths['gold_violation_csv']}`",
            f"- Gold violation dictionary: `{paths['gold_violation_dictionary_csv']}`",
            f"- Gold dashboard violation explained: `{paths['gold_dashboard_violation_csv']}`",
            f"- Reference violation dictionary latest: `{paths['reference_violation_dictionary_csv']}`",
            f"- Violation remediation guide: `{paths['docs_violation_dictionary_md']}`",
            f"- DQ JSON: `{paths['dq_json']}`",
            f"- DQ Markdown: `{paths['dq_md']}`",
            "",
        ]
    )

    with paths["dq_md"].open("w", encoding="utf-8") as f_md:
        f_md.write("\n".join(md_lines))

    latest_run_payload = {
        "dataset_id": DATASET_ID,
        "run_id": run_id,
        "generated_at_utc": dq_report["generated_at_utc"],
        "status": dq_report["status"],
        "mode": extraction["mode_effective"],
        "where_clause": where_clause,
        "dq_json": str(paths["dq_json"]),
        "dq_md": str(paths["dq_md"]),
        "silver_event_csv": str(paths["silver_event_csv"]),
        "gold_business_csv": str(paths["gold_business_csv"]),
        "dashboard_violation_explained_csv": str(paths["gold_dashboard_violation_csv"]),
        "violation_dictionary_csv": str(paths["reference_violation_dictionary_csv"]),
        "violation_dictionary_md": str(paths["docs_violation_dictionary_md"]),
    }
    with paths["latest_run_json"].open("w", encoding="utf-8") as f_latest:
        json.dump(latest_run_payload, f_latest, ensure_ascii=False, indent=2)

    if paths["state_json"].exists():
        with paths["state_json"].open("r", encoding="utf-8") as f_state:
            pipeline_state = json.load(f_state)
    else:
        pipeline_state = {"dataset_id": DATASET_ID, "runs": []}

    pipeline_state["latest_run_id"] = run_id
    pipeline_state.setdefault("runs", []).append(
        {
            "run_id": run_id,
            "generated_at_utc": dq_report["generated_at_utc"],
            "status": dq_report["status"],
            "mode": extraction["mode_effective"],
            "where_clause": where_clause,
            "rows_after_dedup": rows_after_dedup,
            "inspection_event_count": inspection_event_count,
            "max_inspection_date": to_iso_date(max_event_date),
            "dq_json": str(paths["dq_json"]),
        }
    )
    # Keep state history bounded.
    pipeline_state["runs"] = pipeline_state["runs"][-50:]
    with paths["state_json"].open("w", encoding="utf-8") as f_state:
        json.dump(pipeline_state, f_state, ensure_ascii=False, indent=2)

    elapsed_seconds = (now_utc() - run_started_at).total_seconds()
    print(f"[info] finished in {elapsed_seconds:.1f}s status={dq_report['status']}")
    print(f"[info] dq_report={paths['dq_md']}")
    return 1 if blockers else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run King County food inspection cleaning + DQ pipeline."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root directory. Defaults to current working directory.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "incremental"],
        help="Extraction mode: full snapshot or incremental window by inspection_date.",
    )
    parser.add_argument(
        "--since-date",
        type=str,
        default="",
        help="Optional incremental start date in YYYY-MM-DD (overrides state checkpoint).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=3,
        help="For incremental mode without --since-date, include this many lookback days from latest checkpoint.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=50000,
        help="Socrata page size for each API request.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional max rows to process (for fast dry runs).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional max pages to fetch.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="HTTP timeout for each request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Max retries for request failures.",
    )
    parser.add_argument(
        "--app-token",
        type=str,
        default="",
        help="Optional Socrata app token (X-App-Token).",
    )
    parser.add_argument(
        "--fetcher",
        type=str,
        default="auto",
        choices=["auto", "urllib", "curl"],
        help="HTTP fetch backend: auto (urllib then curl fallback), urllib, or curl.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
