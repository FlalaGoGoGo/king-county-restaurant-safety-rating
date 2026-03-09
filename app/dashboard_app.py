#!/usr/bin/env python3
from __future__ import annotations

import json
import importlib.util
import os
import re
import time
from collections import defaultdict
from difflib import SequenceMatcher
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - fallback for non-UI test env
    st = None

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - fallback for non-Plotly env
    go = None
    PLOTLY_AVAILABLE = False

try:
    import joblib

    JOBLIB_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - fallback for non-ML env
    JOBLIB_AVAILABLE = False

SHAP_AVAILABLE = importlib.util.find_spec("shap") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - fallback for non-ML env
    SKLEARN_AVAILABLE = False

DATASET_ID = "f29f-zza5"
DEPLOY_BUNDLE_DIRNAME = "deploy_bundle"
SPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^A-Z0-9 ]+")
HIGH_RISK_GRADES = {"4"}
HIGH_RISK_RED_POINTS_THRESHOLD = 25
ROUTINE_INSPECTION_TYPE = "Routine Inspection/Field Review"
RETURN_INSPECTION_TYPE = "Return Inspection"
RATING_CLOSURE_LOOKBACK_DAYS = 90
MULTIPLE_RETURN_INSPECTION_THRESHOLD = 2
RISK_LEVEL_RE = re.compile(r"RISK\s+CATEGORY\s+(III|II|I)\b", re.IGNORECASE)
RISK_ROMAN_TO_LEVEL = {"I": "1", "II": "2", "III": "3"}
RISK_LEVEL_LABELS = {"1": "Risk I", "2": "Risk II", "3": "Risk III"}
RISK_LEVEL_TO_ROUTINE_WINDOW = {"1": 2, "2": 2, "3": 4}
OUTSIDE_KING_COUNTY_LABEL = "OUTSIDE KING COUNTY"
CITY_DIRECT_CORRECTIONS = {
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
KING_COUNTY_INCORPORATED_CITY_SET = {
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
    "FEDERAL WAY",
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
    "REDMOND",
    "RENTON",
    "SAMMAMISH",
    "SEATAC",
    "SEATTLE",
    "SHORELINE",
    "SKYKOMISH",
    "SNOQUALMIE",
    "TUKWILA",
    "WOODINVILLE",
    "YARROW POINT",
}
KING_COUNTY_UNINCORPORATED_LOCALITY_SET = {
    "FALL CITY",
    "HOBART",
    "PRESTON",
    "RAVENSDALE",
    "SNOQUALMIE PASS",
    "VASHON",
}
KING_COUNTY_LOCALITY_SET = KING_COUNTY_INCORPORATED_CITY_SET | KING_COUNTY_UNINCORPORATED_LOCALITY_SET
CITY_SCOPE_MAP = {
    **{city: "King County incorporated city" for city in KING_COUNTY_INCORPORATED_CITY_SET},
    **{city: "King County unincorporated locality" for city in KING_COUNTY_UNINCORPORATED_LOCALITY_SET},
}

GRADE_CODE_TO_LABEL = {
    "1": "Excellent",
    "2": "Good",
    "3": "Okay",
    "4": "Needs to Improve",
}
LABEL_NORMALIZATION_MAP = {
    "excellent": "Excellent",
    "good": "Good",
    "okay": "Okay",
    "needs to improve": "Needs to Improve",
    "needs to improve.": "Needs to Improve",
    "needs to improve ": "Needs to Improve",
    "needs to improve": "Needs to Improve",
}
GRADE_LABEL_OPTIONS = ["Excellent", "Good", "Okay", "Needs to Improve"]
GRADE_LABEL_TO_CODE = {v: k for k, v in GRADE_CODE_TO_LABEL.items()}
RATING_NOT_AVAILABLE_LABEL = "Rating not available"
GRADE_LABEL_OPTIONS_WITH_NA = GRADE_LABEL_OPTIONS + [RATING_NOT_AVAILABLE_LABEL]
SEARCH_MODE_OPTIONS: List[Tuple[str, str]] = [
    ("keyword", "Keyword Search"),
    ("map", "Map Search"),
    ("hybrid", "Keyword + Map"),
]
PUBLIC_NAV_ITEMS: List[Tuple[str, str]] = [
    ("overview", "Project Overview"),
    ("search", "Restaurant Search"),
    ("summary", "Historical Insights"),
]
ASSIGNMENT_NAV_ITEMS: List[Tuple[str, str]] = [
    ("executive", "Executive Summary"),
    ("descriptive", "Descriptive Analytics"),
    ("performance", "Model Performance"),
    ("explainability", "Explainability & Interactive Prediction"),
]
RATING_COLOR_MAP = {
    "Excellent": "#099268",
    "Good": "#82c91e",
    "Okay": "#c0d72f",
    "Needs to Improve": "#adb5bd",
    RATING_NOT_AVAILABLE_LABEL: "#ced4da",
}
DEFAULT_KING_COUNTY_MAP_CENTER = {"lat": 47.5480, "lon": -121.9836}
KING_COUNTY_BOUNDARY_COLOR = "#5f6c7b"
RATING_POSTER_IMAGE_FILES = {
    "Excellent": "excellent_poster.jpg",
    "Good": "good_poster.jpg",
    "Okay": "okay_poster.jpg",
    "Needs to Improve": "needs_to_improve_poster.jpg",
    RATING_NOT_AVAILABLE_LABEL: "rating_not_available_poster.jpg",
}
RATING_POSTER_DESCRIPTIONS = {
    "Excellent": "Consistently followed high standards for safe food handling.",
    "Good": "Exceeded the minimum requirements for safe food handling.",
    "Okay": "Met the minimum requirements for safe food handling.",
    "Needs to Improve": (
        "Was either closed within the last 90 days or needed multiple return inspections to correct unsafe food handling."
    ),
    RATING_NOT_AVAILABLE_LABEL: (
        "No official rating is currently available in the published record. "
        "This can happen for new or recently updated establishments."
    ),
}
OFFICIAL_RATING_EXPLANATIONS = {
    "Excellent": "Consistently followed high standards for safe food handling.",
    "Good": "Exceeded the minimum requirements for safe food handling.",
    "Okay": "Met the minimum requirements for safe food handling.",
    "Needs to Improve": (
        "Was either closed within the last 90 days or needed multiple return inspections to correct unsafe food handling."
    ),
}
RISK_CATEGORY_CARD_EXPLANATIONS = {
    "1": {
        "label": "Risk Category 1",
        "bullets": [
            "Scopes: Cold holding, limited food prep",
            "Examples: Coffee stands, hot dog stands",
            "Cook Step Exceptions: Commercially processed microwave dinners",
        ],
    },
    "2": {
        "label": "Risk Category 2",
        "bullets": [
            "Scopes: No Cook Step, Food Preparation",
            "Examples: Ice cream shop, grocery store, some bakeries",
            "Cook Step Exceptions: Pre-packed raw meat or seafood",
        ],
    },
    "3": {
        "label": "Risk Category 3",
        "bullets": [
            "Scopes: Same Day Service or Complex Food Preparation, Meat or Seafood Market, Overnight Cooking, Time as a Control, Approved HACCP",
            "Examples: Restaurant, meat or seafood markets",
        ],
    },
}

MODEL_NUMERIC_FEATURES = [
    "inspection_score",
    "red_points_total",
    "blue_points_total",
    "violation_count_total",
    "grade_num",
    "is_high_risk",
]
MODEL_CATEGORICAL_FEATURES = ["inspection_type", "inspection_result", "city_canonical"]
MODEL_ALL_FEATURES = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES
HOMEWORK_OWNER_QUESTION = (
    "Can a restaurant owner use the latest inspection profile to estimate whether the next inspection "
    "will be high risk, and which controllable signals should be fixed first to reduce that risk?"
)
HOMEWORK_FIELD_GROUPS: List[Tuple[str, List[Tuple[str, str]]]] = [
    (
        "Identifiers",
        [
            ("inspection_event_id", "Unique event-level identifier for one inspection visit."),
            ("inspection_serial_num", "County-published serial number for the inspection record."),
            ("business_id", "Stable establishment identifier used to link repeated inspections."),
            ("row_id", "Unique row identifier in the violation-level table."),
        ],
    ),
    (
        "Business / Location",
        [
            ("business_name_official", "Official restaurant name in the published source."),
            ("business_name_alt", "Alternate or legacy business name when available."),
            ("search_name_norm", "Normalized text field used to improve search matching."),
            ("full_address_clean", "Cleaned full street address for mapping and search."),
            ("city_canonical", "Standardized city/locality label after cleaning rules."),
            ("zip_code", "ZIP code associated with the establishment address."),
            ("latitude", "Latitude used for map display."),
            ("longitude", "Longitude used for map display."),
        ],
    ),
    (
        "Inspection Outcome",
        [
            ("inspection_date", "Date of the inspection event."),
            ("inspection_type", "Inspection category such as routine, return, or consultation."),
            ("inspection_result", "Published inspection outcome or status."),
            ("inspection_score", "Numerical score recorded for that inspection."),
            ("inspection_closed_business", "Flag indicating whether the business was closed during inspection."),
            ("source_row_count", "Number of raw rows consolidated into the event-level record."),
            ("generated_from_missing_serial", "Flag showing whether an event id was reconstructed from missing source metadata."),
        ],
    ),
    (
        "Rating / Risk",
        [
            ("grade", "Published county grade code."),
            ("grade_label", "Human-readable label for the published grade."),
            ("rating_not_available", "Flag indicating that no public rating was available."),
            ("red_points_total", "Total red-point severity recorded on the inspection."),
            ("blue_points_total", "Total blue-point severity recorded on the inspection."),
            ("red_violation_count", "Count of red violations on the inspection."),
            ("blue_violation_count", "Count of blue violations on the inspection."),
            ("violation_count_total", "Total number of violations on the inspection."),
        ],
    ),
    (
        "Violation / Remediation",
        [
            ("violation_type", "Red or blue violation category."),
            ("violation_code", "County violation code linked to the finding."),
            ("violation_points", "Points assigned to the specific violation."),
            ("violation_desc_clean", "Cleaned violation description for analysis."),
            ("violation_desc_raw", "Original raw violation description from source data."),
            ("dictionary_default_points_mode", "Dictionary rule describing how default points were interpreted."),
            ("dictionary_canonical_description", "Canonical description from the remediation dictionary."),
            ("action_category", "Grouped remediation theme for the violation."),
            ("action_priority", "Priority label for the recommended corrective action."),
            ("action_summary_zh", "Chinese remediation summary stored in the dictionary."),
            ("action_steps_zh", "Chinese action steps stored in the dictionary."),
            ("action_summary_en", "English remediation summary used in the dashboard."),
            ("safe_food_handling_refs", "Reference links or citations for safe-food-handling guidance."),
            ("action_source", "Source used to derive the remediation guidance."),
        ],
    ),
]

INSIGHT_QUESTIONS: Dict[str, List[Tuple[str, str]]] = {
    "Consumer": [
        ("C1", "I want to know which restaurants got better or worse this month."),
        ("C2", "I want to know which restaurants are consistently safer over time."),
        ("C3", "I want to know which restaurants had serious red-flag violations recently."),
        ("C4", "I want to know how one restaurant compares with similar restaurants nearby."),
        ("C5", "I want to know which hygiene problems are most common right now."),
        ("C6", "I want to know whether food safety risk is improving or worsening over time."),
    ],
    "Restaurant Owner": [
        ("O1", "I want to know which violations to fix first to reduce risk fastest."),
        ("O2", "I want to know which violations keep repeating at my restaurant."),
        ("O3", "I want to know how my restaurant compares with peer restaurants in my city."),
        ("O4", "I want to know what changed before my rating dropped."),
        ("O5", "I want to know which violation categories are increasing in my area."),
        ("O6", "I want to know whether my remediation actions are actually working."),
    ],
    "Regulator": [
        ("R1", "I want to know which cities or zip codes need attention right now."),
        ("R2", "I want to know where high-risk rates are rising the fastest."),
        ("R3", "I want to know which restaurants show persistent high-risk patterns."),
        ("R4", "I want to know which areas have both high risk and high inspection workload."),
        ("R5", "I want to know whether risk patterns are seasonal."),
        ("R6", "I want to know where data quality issues may affect decisions."),
    ],
    "King County": [
        (
            "K1",
            "I want to know which restaurants have differences between King County's published grade and this dashboard's calculated grade.",
        ),
        (
            "K2",
            "I want to know which records are outside King County scope or have locality ambiguity.",
        ),
        (
            "K3",
            "I want to know where city normalization quality is weak.",
        ),
        (
            "K4",
            "I want to know whether identifiers and dates are complete enough for reliable analytics.",
        ),
        (
            "K5",
            "I want to know where semantic nulls and text truncation could mislead users.",
        ),
        (
            "K6",
            "I want to know which quality issues are getting better or worse across data runs.",
        ),
    ],
}

SUPPORTED_LOCALES: Tuple[str, ...] = ("en", "zh-CN", "es", "fr", "ja")
LOCALE_LABELS: Dict[str, str] = {
    "en": "English",
    "zh-CN": "中文",
    "es": "Español",
    "fr": "Français",
    "ja": "日本語",
}
I18N_MESSAGES: Dict[str, Dict[str, str]] = {
    "en": {
        "app.title": "King County Restaurant Safety Dashboard",
        "app.caption": "Tabs: Project Overview / Restaurant Search / Historical Insights / Assignment Workflow",
        "locale.label": "Language",
        "tab.overview": "Project Overview",
        "tab.search": "Restaurant Search",
        "tab.summary": "Historical Insights",
        "tab.predict": "Assignment Workflow",
        "overview.header": "Project Overview",
        "search.step1": "Step 1. Please enter the restaurant details you want to search",
        "search.step1.hint": "Choose one search method. Only filters for the active method are shown.",
        "search.step2": "Step 2. Please click any restaurant to view detailed results",
        "search.step3": "Step 3. Please click one inspection to view violation details",
        "search.step4": "Step 4. Review violations and remediation recommendations",
        "search.clear_all": "Clear all filters",
        "search.map.hint": "Click one point on the map to choose a restaurant. The detailed inspection view still opens below.",
        "mode.keyword": "Keyword Search",
        "mode.map": "Map Search",
        "mode.hybrid": "Keyword + Map",
        "role.Consumer": "Consumer",
        "role.Restaurant Owner": "Restaurant Owner",
        "role.Regulator": "Regulator",
        "role.King County": "King County",
        "summary.header": "Historical Insights",
        "summary.step1": "Step 1. Please select your role",
        "summary.step2": "Step 2. Please select what you care about",
        "summary.question_label": "I want to know...",
        "summary.role_hint": "Choose your role, then choose one question. Only the relevant view is shown.",
        "summary.risk_hint": "High-risk definition: rating is 'Needs to Improve' or Red points >= 25.",
        "question.C1": "I want to know which restaurants got better or worse this month.",
        "question.C2": "I want to know which restaurants are consistently safer over time.",
        "question.C3": "I want to know which restaurants had serious red-flag violations recently.",
        "question.C4": "I want to know how one restaurant compares with similar restaurants nearby.",
        "question.C5": "I want to know which hygiene problems are most common right now.",
        "question.C6": "I want to know whether food safety risk is improving or worsening over time.",
        "question.O1": "I want to know which violations to fix first to reduce risk fastest.",
        "question.O2": "I want to know which violations keep repeating at my restaurant.",
        "question.O3": "I want to know how my restaurant compares with peer restaurants in my city.",
        "question.O4": "I want to know what changed before my rating dropped.",
        "question.O5": "I want to know which violation categories are increasing in my area.",
        "question.O6": "I want to know whether my remediation actions are actually working.",
        "question.R1": "I want to know which cities or zip codes need attention right now.",
        "question.R2": "I want to know where high-risk rates are rising the fastest.",
        "question.R3": "I want to know which restaurants show persistent high-risk patterns.",
        "question.R4": "I want to know which areas have both high risk and high inspection workload.",
        "question.R5": "I want to know whether risk patterns are seasonal.",
        "question.R6": "I want to know where data quality issues may affect decisions.",
        "question.K1": "I want to know which restaurants have differences between King County's published grade and this dashboard's calculated grade.",
        "question.K2": "I want to know which records are outside King County scope or have locality ambiguity.",
        "question.K3": "I want to know where city normalization quality is weak.",
        "question.K4": "I want to know whether identifiers and dates are complete enough for reliable analytics.",
        "question.K5": "I want to know where semantic nulls and text truncation could mislead users.",
        "question.K6": "I want to know which quality issues are getting better or worse across data runs.",
        "predict.header": "Assignment Workflow",
    },
    "zh-CN": {
        "app.title": "金县餐厅食品安全仪表盘",
        "app.caption": "标签页：项目总览 / 餐厅搜索 / 历史洞察 / 作业分析流程",
        "locale.label": "语言",
        "tab.overview": "项目总览",
        "tab.search": "餐厅搜索",
        "tab.summary": "历史洞察",
        "tab.predict": "作业分析流程",
        "overview.header": "项目总览",
        "search.step1": "步骤1：请输入要查询的餐厅信息",
        "search.step1.hint": "请选择一种搜索方式，仅显示当前方式对应的筛选项。",
        "search.step2": "步骤2：请点击任一餐厅查看详细结果",
        "search.step3": "步骤3：请点击一次检查记录查看违规详情",
        "search.step4": "步骤4：查看违规与整改建议",
        "search.clear_all": "清空全部筛选",
        "search.map.hint": "请在地图上点击一个点位选择餐厅，详细检查记录将在下方显示。",
        "mode.keyword": "关键词搜索",
        "mode.map": "地图搜索",
        "mode.hybrid": "关键词 + 地图",
        "role.Consumer": "消费者",
        "role.Restaurant Owner": "餐厅经营者",
        "role.Regulator": "监管者",
        "role.King County": "金县",
        "summary.header": "历史洞察",
        "summary.step1": "步骤1：请选择你的角色",
        "summary.step2": "步骤2：请选择你关心的问题",
        "summary.question_label": "我想了解...",
        "summary.role_hint": "先选择你的角色，再选择一个问题。页面只展示与该问题相关的视图。",
        "summary.risk_hint": "高风险定义：评级为“Needs to Improve”或红分 >= 25。",
        "question.C1": "我想知道本月哪些餐厅变好或变差了。",
        "question.C2": "我想知道哪些餐厅长期更安全稳定。",
        "question.C3": "我想知道近期哪些餐厅出现了严重红色违规。",
        "question.C4": "我想知道某家餐厅与附近同类餐厅相比如何。",
        "question.C5": "我想知道当前最常见的卫生问题是什么。",
        "question.C6": "我想知道食品安全风险总体是在改善还是恶化。",
        "question.O1": "我想知道优先整改哪些违规能最快降低风险。",
        "question.O2": "我想知道我店里哪些违规在反复出现。",
        "question.O3": "我想知道我店与同城同类餐厅相比如何。",
        "question.O4": "我想知道评级下降前发生了哪些变化。",
        "question.O5": "我想知道我所在区域哪些违规类别在上升。",
        "question.O6": "我想知道整改措施是否真的有效。",
        "question.R1": "我想知道当前哪些城市或邮编区域需要重点关注。",
        "question.R2": "我想知道哪些区域高风险率上升最快。",
        "question.R3": "我想知道哪些餐厅持续呈现高风险模式。",
        "question.R4": "我想知道哪些区域同时具备高风险与高检查工作量。",
        "question.R5": "我想知道风险模式是否具有季节性。",
        "question.R6": "我想知道哪些数据质量问题可能影响决策。",
        "question.K1": "我想知道哪些餐厅在县官方评级与本仪表盘计算评级之间存在差异。",
        "question.K2": "我想知道哪些记录超出金县范围或存在地理归属歧义。",
        "question.K3": "我想知道城市名称标准化质量薄弱在哪里。",
        "question.K4": "我想知道标识符和日期完整性是否足以支持可靠分析。",
        "question.K5": "我想知道语义空值或文本截断会在哪些地方误导用户。",
        "question.K6": "我想知道各类数据质量问题是在改善还是恶化。",
        "predict.header": "作业分析流程",
    },
    "es": {
        "app.title": "Panel de Seguridad Alimentaria del Condado de King",
        "app.caption": "Pestañas: Resumen del Proyecto / Búsqueda de Restaurantes / Información Histórica / Flujo Analítico de la Tarea",
        "locale.label": "Idioma",
        "tab.overview": "Resumen del Proyecto",
        "tab.search": "Búsqueda de Restaurantes",
        "tab.summary": "Información Histórica",
        "tab.predict": "Flujo Analítico de la Tarea",
        "overview.header": "Resumen del Proyecto",
        "search.step1": "Paso 1. Ingrese los datos del restaurante que desea buscar",
        "search.step1.hint": "Elija un método de búsqueda. Solo se muestran los filtros del método activo.",
        "search.step2": "Paso 2. Haga clic en cualquier restaurante para ver resultados detallados",
        "search.step3": "Paso 3. Haga clic en una inspección para ver los detalles de infracciones",
        "search.step4": "Paso 4. Revise infracciones y recomendaciones de corrección",
        "search.clear_all": "Borrar todos los filtros",
        "search.map.hint": "Haga clic en un punto del mapa para elegir un restaurante. La vista detallada de inspecciones se muestra abajo.",
        "mode.keyword": "Búsqueda por palabra clave",
        "mode.map": "Búsqueda por mapa",
        "mode.hybrid": "Palabra clave + mapa",
        "role.Consumer": "Consumidor",
        "role.Restaurant Owner": "Propietario del restaurante",
        "role.Regulator": "Regulador",
        "role.King County": "Condado de King",
        "summary.header": "Información Histórica",
        "summary.step1": "Paso 1. Seleccione su rol",
        "summary.step2": "Paso 2. Seleccione lo que le importa",
        "summary.question_label": "Quiero saber...",
        "summary.role_hint": "Elija su rol y luego una pregunta. Solo se mostrará la vista relevante.",
        "summary.risk_hint": "Definición de alto riesgo: la calificación es 'Needs to Improve' o puntos rojos >= 25.",
        "question.C1": "Quiero saber qué restaurantes mejoraron o empeoraron este mes.",
        "question.C2": "Quiero saber qué restaurantes son consistentemente más seguros con el tiempo.",
        "question.C3": "Quiero saber qué restaurantes tuvieron infracciones graves recientemente.",
        "question.C4": "Quiero saber cómo se compara un restaurante con restaurantes similares cercanos.",
        "question.C5": "Quiero saber qué problemas de higiene son más comunes ahora.",
        "question.C6": "Quiero saber si el riesgo de seguridad alimentaria está mejorando o empeorando con el tiempo.",
        "question.O1": "Quiero saber qué infracciones corregir primero para reducir el riesgo más rápido.",
        "question.O2": "Quiero saber qué infracciones se repiten en mi restaurante.",
        "question.O3": "Quiero saber cómo se compara mi restaurante con restaurantes pares en mi ciudad.",
        "question.O4": "Quiero saber qué cambió antes de que bajara mi calificación.",
        "question.O5": "Quiero saber qué categorías de infracciones están aumentando en mi zona.",
        "question.O6": "Quiero saber si mis acciones correctivas realmente están funcionando.",
        "question.R1": "Quiero saber qué ciudades o códigos postales necesitan atención ahora.",
        "question.R2": "Quiero saber dónde las tasas de alto riesgo están subiendo más rápido.",
        "question.R3": "Quiero saber qué restaurantes muestran patrones persistentes de alto riesgo.",
        "question.R4": "Quiero saber qué zonas tienen alto riesgo y alta carga de inspección.",
        "question.R5": "Quiero saber si los patrones de riesgo son estacionales.",
        "question.R6": "Quiero saber dónde los problemas de calidad de datos pueden afectar decisiones.",
        "question.K1": "Quiero saber qué restaurantes tienen diferencias entre la calificación publicada por el condado y la calculada por este panel.",
        "question.K2": "Quiero saber qué registros están fuera del alcance del condado o tienen ambigüedad de localidad.",
        "question.K3": "Quiero saber dónde es débil la normalización de ciudades.",
        "question.K4": "Quiero saber si identificadores y fechas están suficientemente completos para análisis confiables.",
        "question.K5": "Quiero saber dónde los nulos semánticos y truncamientos de texto pueden confundir a los usuarios.",
        "question.K6": "Quiero saber qué problemas de calidad están mejorando o empeorando entre ejecuciones.",
        "predict.header": "Flujo Analítico de la Tarea",
    },
    "fr": {
        "app.title": "Tableau de Bord de Sécurité Alimentaire du Comté de King",
        "app.caption": "Onglets : Aperçu du Projet / Recherche de Restaurants / Analyses Historiques / Workflow Analytique du Devoir",
        "locale.label": "Langue",
        "tab.overview": "Aperçu du Projet",
        "tab.search": "Recherche de Restaurants",
        "tab.summary": "Analyses Historiques",
        "tab.predict": "Workflow Analytique du Devoir",
        "overview.header": "Aperçu du Projet",
        "search.step1": "Étape 1. Saisissez les informations du restaurant à rechercher",
        "search.step1.hint": "Choisissez une méthode de recherche. Seuls les filtres de la méthode active sont affichés.",
        "search.step2": "Étape 2. Cliquez sur un restaurant pour voir les résultats détaillés",
        "search.step3": "Étape 3. Cliquez sur une inspection pour voir les détails des infractions",
        "search.step4": "Étape 4. Consultez les infractions et recommandations de correction",
        "search.clear_all": "Effacer tous les filtres",
        "search.map.hint": "Cliquez sur un point de la carte pour choisir un restaurant. La vue détaillée des inspections s'affiche ci-dessous.",
        "mode.keyword": "Recherche par mot-clé",
        "mode.map": "Recherche par carte",
        "mode.hybrid": "Mot-clé + carte",
        "role.Consumer": "Consommateur",
        "role.Restaurant Owner": "Propriétaire du restaurant",
        "role.Regulator": "Régulateur",
        "role.King County": "Comté de King",
        "summary.header": "Analyses Historiques",
        "summary.step1": "Étape 1. Sélectionnez votre rôle",
        "summary.step2": "Étape 2. Sélectionnez ce qui vous intéresse",
        "summary.question_label": "Je veux savoir...",
        "summary.role_hint": "Choisissez votre rôle puis une question. Seule la vue pertinente sera affichée.",
        "summary.risk_hint": "Définition du risque élevé : la note est 'Needs to Improve' ou points rouges >= 25.",
        "question.C1": "Je veux savoir quels restaurants se sont améliorés ou détériorés ce mois-ci.",
        "question.C2": "Je veux savoir quels restaurants sont régulièrement plus sûrs dans le temps.",
        "question.C3": "Je veux savoir quels restaurants ont eu récemment des infractions graves.",
        "question.C4": "Je veux savoir comment un restaurant se compare à des restaurants similaires à proximité.",
        "question.C5": "Je veux savoir quels problèmes d'hygiène sont les plus fréquents actuellement.",
        "question.C6": "Je veux savoir si le risque sanitaire alimentaire s'améliore ou se dégrade au fil du temps.",
        "question.O1": "Je veux savoir quelles infractions corriger en premier pour réduire le risque plus vite.",
        "question.O2": "Je veux savoir quelles infractions se répètent dans mon restaurant.",
        "question.O3": "Je veux savoir comment mon restaurant se compare à des restaurants similaires dans ma ville.",
        "question.O4": "Je veux savoir ce qui a changé avant la baisse de ma note.",
        "question.O5": "Je veux savoir quelles catégories d'infractions augmentent dans ma zone.",
        "question.O6": "Je veux savoir si mes actions correctives fonctionnent réellement.",
        "question.R1": "Je veux savoir quelles villes ou quels codes postaux nécessitent une attention immédiate.",
        "question.R2": "Je veux savoir où les taux de risque élevé augmentent le plus vite.",
        "question.R3": "Je veux savoir quels restaurants montrent des schémas persistants de risque élevé.",
        "question.R4": "Je veux savoir quelles zones cumulent risque élevé et forte charge d'inspection.",
        "question.R5": "Je veux savoir si les schémas de risque sont saisonniers.",
        "question.R6": "Je veux savoir où les problèmes de qualité des données peuvent affecter les décisions.",
        "question.K1": "Je veux savoir quels restaurants présentent des écarts entre la note publiée par le comté et la note calculée par ce tableau de bord.",
        "question.K2": "Je veux savoir quels enregistrements sont hors périmètre du comté ou présentent une ambiguïté de localité.",
        "question.K3": "Je veux savoir où la normalisation des villes est faible.",
        "question.K4": "Je veux savoir si les identifiants et les dates sont suffisamment complets pour une analyse fiable.",
        "question.K5": "Je veux savoir où les valeurs nulles sémantiques et les troncatures de texte peuvent induire les utilisateurs en erreur.",
        "question.K6": "Je veux savoir quels problèmes de qualité s'améliorent ou se détériorent entre les exécutions.",
        "predict.header": "Workflow Analytique du Devoir",
    },
    "ja": {
        "app.title": "キング郡レストラン食品安全ダッシュボード",
        "app.caption": "タブ：プロジェクト概要 / レストラン検索 / 履歴インサイト / 課題分析ワークフロー",
        "locale.label": "言語",
        "tab.overview": "プロジェクト概要",
        "tab.search": "レストラン検索",
        "tab.summary": "履歴インサイト",
        "tab.predict": "課題分析ワークフロー",
        "overview.header": "プロジェクト概要",
        "search.step1": "ステップ1：検索したいレストラン情報を入力してください",
        "search.step1.hint": "検索方法を1つ選択してください。選択中の方法に対応するフィルターのみ表示されます。",
        "search.step2": "ステップ2：任意のレストランをクリックして詳細結果を表示してください",
        "search.step3": "ステップ3：1件の検査をクリックして違反詳細を表示してください",
        "search.step4": "ステップ4：違反内容と改善提案を確認してください",
        "search.clear_all": "すべてのフィルターをクリア",
        "search.map.hint": "地図上のポイントをクリックしてレストランを選択します。詳細検査ビューは下部に表示されます。",
        "mode.keyword": "キーワード検索",
        "mode.map": "地図検索",
        "mode.hybrid": "キーワード + 地図",
        "role.Consumer": "消費者",
        "role.Restaurant Owner": "店舗オーナー",
        "role.Regulator": "規制担当",
        "role.King County": "キング郡",
        "summary.header": "履歴インサイト",
        "summary.step1": "ステップ1：役割を選択してください",
        "summary.step2": "ステップ2：知りたい内容を選択してください",
        "summary.question_label": "知りたいこと...",
        "summary.role_hint": "まず役割を選択し、次に質問を選択してください。該当するビューのみ表示されます。",
        "summary.risk_hint": "高リスク定義：評価が「Needs to Improve」または Red points >= 25。",
        "question.C1": "今月、どのレストランの評価が改善または悪化したかを知りたい。",
        "question.C2": "長期的により安全性が高いレストランを知りたい。",
        "question.C3": "最近、重大な赤色違反があったレストランを知りたい。",
        "question.C4": "あるレストランが近隣の類似店と比べてどうか知りたい。",
        "question.C5": "現在、最も多い衛生問題を知りたい。",
        "question.C6": "食品安全リスクが時間とともに改善しているか悪化しているか知りたい。",
        "question.O1": "リスクを最速で下げるために、どの違反から優先的に改善すべきか知りたい。",
        "question.O2": "自店で繰り返し発生している違反を知りたい。",
        "question.O3": "自店が同じ市内の同業店と比べてどうか知りたい。",
        "question.O4": "評価が下がる前に何が変わったか知りたい。",
        "question.O5": "地域で増加している違反カテゴリを知りたい。",
        "question.O6": "改善施策が実際に効果を出しているか知りたい。",
        "question.R1": "今、どの都市または郵便番号エリアに注意が必要か知りたい。",
        "question.R2": "高リスク率が最も速く上昇している場所を知りたい。",
        "question.R3": "継続的な高リスク傾向を示すレストランを知りたい。",
        "question.R4": "高リスクかつ検査負荷が高いエリアを知りたい。",
        "question.R5": "リスクパターンに季節性があるか知りたい。",
        "question.R6": "意思決定に影響し得るデータ品質問題の場所を知りたい。",
        "question.K1": "郡の公表評価と本ダッシュボード計算評価に差異があるレストランを知りたい。",
        "question.K2": "キング郡の対象外、または地域帰属が曖昧なレコードを知りたい。",
        "question.K3": "都市名正規化の品質が弱い箇所を知りたい。",
        "question.K4": "識別子と日付の完全性が信頼できる分析に十分か知りたい。",
        "question.K5": "意味的な欠損やテキスト切り捨てがユーザーを誤解させる箇所を知りたい。",
        "question.K6": "データ品質問題が実行間で改善しているか悪化しているか知りたい。",
        "predict.header": "課題分析ワークフロー",
    },
}
I18N_DEBUG = os.getenv("KC_I18N_DEBUG", "").strip().lower() in {"1", "true", "yes"}
I18N_MISSING_KEYS: Dict[str, Set[str]] = defaultdict(set)


def tr(key: str, locale: Optional[str] = None, **kwargs: Any) -> str:
    loc = clean_text(locale or st.session_state.get("locale", "en") if st else "en")
    if loc not in SUPPORTED_LOCALES:
        loc = "en"
    locale_map = I18N_MESSAGES.get(loc, {})
    if key in locale_map:
        text = locale_map[key]
    else:
        if loc != "en" and key in I18N_MESSAGES["en"]:
            I18N_MISSING_KEYS[loc].add(key)
        text = I18N_MESSAGES["en"].get(key) or key
    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:
            pass
    return text


def render_i18n_debug_panel() -> None:
    if st is None or not I18N_DEBUG:
        return
    loc = clean_text(st.session_state.get("locale", "en"))
    if loc not in SUPPORTED_LOCALES:
        loc = "en"
    missing_keys = sorted(I18N_MISSING_KEYS.get(loc, set()))
    if missing_keys:
        st.warning(f"i18n debug: missing {len(missing_keys)} keys for locale '{loc}'.")
        st.code("\n".join(missing_keys), language="text")
    else:
        st.caption(f"i18n debug: no missing keys for locale '{loc}'.")


def normalize_search_text(value: str) -> str:
    text = str(value or "").upper()
    text = NON_ALNUM_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def format_city_name(value: object) -> str:
    city = clean_text(value)
    if not city:
        return ""
    if city == OUTSIDE_KING_COUNTY_LABEL:
        return "Outside King County"
    if city == city.upper() or city == city.lower():
        return city.title()
    return city


def add_city_display_column(df: pd.DataFrame) -> pd.DataFrame:
    df["city_display"] = df["city_canonical"].map(format_city_name)
    return df


def clean_city_token(value: object) -> str:
    city = clean_text(value).upper().rstrip(",")
    city = re.sub(r"\bWA\b$", "", city).strip()
    city = SPACE_RE.sub(" ", city)
    if city in CITY_DIRECT_CORRECTIONS:
        return CITY_DIRECT_CORRECTIONS[city]
    return city


def build_zip_to_locality_lookup(df: pd.DataFrame) -> Dict[str, str]:
    if df.empty or "zip_code" not in df.columns or "city_canonical" not in df.columns:
        return {}
    base = df.copy()
    base["zip_code"] = base["zip_code"].astype(str).str.strip()
    base["city_base"] = base["city_canonical"].map(clean_city_token)
    valid = base[
        base["zip_code"].ne("")
        & base["city_base"].isin(KING_COUNTY_LOCALITY_SET)
    ].copy()
    if valid.empty:
        return {}
    zip_counts = valid.groupby("zip_code", dropna=False).size().rename("zip_total").reset_index()
    city_counts = (
        valid.groupby(["zip_code", "city_base"], dropna=False)
        .size()
        .rename("city_count")
        .reset_index()
        .sort_values(["zip_code", "city_count", "city_base"], ascending=[True, False, True])
    )
    top = city_counts.drop_duplicates("zip_code", keep="first").merge(zip_counts, on="zip_code", how="left")
    top["share"] = top["city_count"] / top["zip_total"]
    top = top[(top["city_count"] >= 5) & (top["share"] >= 0.70)].copy()
    return dict(zip(top["zip_code"], top["city_base"]))


def apply_city_quality_rules(df: pd.DataFrame, zip_lookup: Dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    raw_series = out.get("city_canonical", pd.Series("", index=out.index)).astype(str).str.strip()
    base_city = raw_series.map(clean_city_token)
    zip_code = out.get("zip_code", pd.Series("", index=out.index)).astype(str).str.strip()
    mapped_city = zip_code.map(zip_lookup).fillna("")

    out["city_raw_original"] = raw_series
    out["city_cleaning_reason"] = "exact"
    out["city_canonical"] = base_city

    direct_mapped_mask = raw_series.ne("") & raw_series.str.upper().map(lambda x: CITY_DIRECT_CORRECTIONS.get(x, "")).ne("")
    out.loc[direct_mapped_mask, "city_cleaning_reason"] = (
        "mapped:" + raw_series[direct_mapped_mask].str.upper() + "->" + out.loc[direct_mapped_mask, "city_canonical"]
    )

    zip_fix_mask = (
        out["city_canonical"].ne("")
        & ~out["city_canonical"].isin(KING_COUNTY_LOCALITY_SET)
        & mapped_city.ne("")
    )
    out.loc[zip_fix_mask, "city_cleaning_reason"] = (
        "zip_majority_fix:" + zip_code[zip_fix_mask] + "->" + mapped_city[zip_fix_mask]
    )
    out.loc[zip_fix_mask, "city_canonical"] = mapped_city[zip_fix_mask]

    outside_mask = out["city_canonical"].ne("") & ~out["city_canonical"].isin(KING_COUNTY_LOCALITY_SET)
    out.loc[outside_mask, "city_cleaning_reason"] = (
        "outside_king_county_or_nonlocality:" + out.loc[outside_mask, "city_canonical"]
    )
    out.loc[outside_mask, "city_canonical"] = OUTSIDE_KING_COUNTY_LABEL

    empty_mask = out["city_canonical"].eq("")
    out.loc[empty_mask, "city_cleaning_reason"] = "empty"
    out["city_scope"] = out["city_canonical"].map(CITY_SCOPE_MAP)
    outside_scope_mask = out["city_canonical"].eq(OUTSIDE_KING_COUNTY_LABEL)
    out.loc[outside_scope_mask, "city_scope"] = "Outside King County or unresolved"
    out["city_scope"] = out["city_scope"].fillna("")
    out["city_in_king_county"] = out["city_canonical"].isin(KING_COUNTY_LOCALITY_SET).astype(int)
    return add_city_display_column(out)


def rating_label_from_values(grade_code: str, grade_label: str) -> str:
    code = str(grade_code or "").strip()
    if code in GRADE_CODE_TO_LABEL:
        return GRADE_CODE_TO_LABEL[code]

    label_raw = str(grade_label or "").strip()
    label_norm = LABEL_NORMALIZATION_MAP.get(label_raw.lower(), "")
    if label_norm:
        return label_norm
    return label_raw


def parse_risk_level_from_description(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    match = RISK_LEVEL_RE.search(text)
    if not match:
        return ""
    return RISK_ROMAN_TO_LEVEL.get(match.group(1).upper(), "")


def format_risk_level_label(value: object) -> str:
    level = clean_text(value)
    return RISK_LEVEL_LABELS.get(level, "")


def format_rating_source_label(value: object) -> str:
    mapping = {
        "dashboard_recent_routine_average": "Dashboard calculation: recent routine-inspection average",
        "dashboard_recent_closure_or_multiple_returns": (
            "Dashboard calculation: recent closure or multiple return inspections"
        ),
        "rating_not_available": RATING_NOT_AVAILABLE_LABEL,
    }
    return mapping.get(clean_text(value), clean_text(value))


def rating_label_from_avg_red_points(risk_level: str, avg_red_points: float) -> str:
    if pd.isna(avg_red_points):
        return ""
    risk = clean_text(risk_level)
    avg_val = float(avg_red_points)
    if risk == "3":
        if avg_val <= 3.75:
            return "Excellent"
        if avg_val <= 16.25:
            return "Good"
        return "Okay"
    if risk in {"1", "2"}:
        if avg_val <= 0:
            return "Excellent"
        if avg_val <= 5:
            return "Good"
        return "Okay"
    return ""


def pick_mode_text(series: pd.Series) -> str:
    text_series = series.astype(str).str.strip()
    text_series = text_series[text_series != ""]
    if text_series.empty:
        return ""
    modes = text_series.mode()
    if not modes.empty:
        return str(modes.iloc[0]).strip()
    return str(text_series.iloc[0]).strip()


def app_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_columns(df: pd.DataFrame, defaults: Dict[str, object]) -> pd.DataFrame:
    for col, default_value in defaults.items():
        if col not in df.columns:
            df[col] = default_value
    return df


def path_with_gzip_variants(path: Path) -> List[Path]:
    variants = [path]
    if path.suffix != ".gz":
        variants.append(Path(f"{path}.gz"))
    return variants


def resolve_first_existing_path(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def dataset_path_candidates(root: Path, *parts: str) -> List[Path]:
    candidates: List[Path] = []
    for base_dir_name in ("Data", DEPLOY_BUNDLE_DIRNAME):
        candidates.extend(path_with_gzip_variants(root / base_dir_name / Path(*parts)))
    return candidates


def payload_path_candidates(root: Path, raw_value: object) -> List[Path]:
    value = clean_text(raw_value)
    if not value:
        return []
    payload_path = Path(value)
    if not payload_path.is_absolute():
        payload_path = root / payload_path
    return path_with_gzip_variants(payload_path)


def resolve_repo_path(root: Path, raw_value: object) -> str:
    value = clean_text(raw_value)
    if not value:
        return ""

    raw_path = Path(value)
    candidates: List[Path] = []
    if raw_path.is_absolute():
        for anchor in ("models", "outputs", "images", "docs", "deploy_bundle", "Data"):
            if anchor in raw_path.parts:
                anchor_index = raw_path.parts.index(anchor)
                candidates.extend(path_with_gzip_variants(root.joinpath(*raw_path.parts[anchor_index:])))
                break
        candidates.extend(path_with_gzip_variants(raw_path))
    else:
        candidates.extend(path_with_gzip_variants(root / raw_path))
        candidates.extend(path_with_gzip_variants(raw_path))

    resolved = resolve_first_existing_path(candidates)
    return str(resolved or raw_path)


def resolve_nested_paths(root: Path, value: Any) -> Any:
    if isinstance(value, dict):
        return {k: resolve_nested_paths(root, v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_nested_paths(root, item) for item in value]
    if isinstance(value, str):
        return resolve_repo_path(root, value)
    return value


def load_latest_run_payload(root: Path) -> Dict[str, str]:
    state_candidates = [
        root / "Data" / "state" / f"{DATASET_ID}_latest_run.json",
        root / DEPLOY_BUNDLE_DIRNAME / "state" / f"{DATASET_ID}_latest_run.json",
    ]
    state_path = resolve_first_existing_path(state_candidates)
    if state_path is None:
        raise FileNotFoundError(f"latest run state not found: {state_candidates[0]}")
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_data_paths(root: Path, payload: Dict[str, str]) -> Tuple[Path, Path]:
    run_id = str(payload.get("run_id", "")).strip()

    silver_candidates = payload_path_candidates(root, payload.get("silver_event_csv", ""))
    silver_candidates.extend(
        dataset_path_candidates(root, "silver", DATASET_ID, run_id, "inspection_event.csv")
    )
    silver_event_csv = resolve_first_existing_path(silver_candidates)

    violation_candidates = payload_path_candidates(root, payload.get("dashboard_violation_explained_csv", ""))
    violation_candidates.extend(
        dataset_path_candidates(root, "gold", DATASET_ID, run_id, "dashboard_violation_explained.csv")
    )
    dashboard_violation_csv = resolve_first_existing_path(violation_candidates)

    if silver_event_csv is None:
        raise FileNotFoundError(f"silver event csv not found: {silver_candidates[0]}")
    if dashboard_violation_csv is None:
        raise FileNotFoundError(f"dashboard violation csv not found: {violation_candidates[0]}")

    return silver_event_csv, dashboard_violation_csv


def resolve_bronze_raw_csv_path(root: Path, payload: Dict[str, str]) -> Path:
    run_id = str(payload.get("run_id", "")).strip()
    raw_candidates = dataset_path_candidates(root, "bronze", DATASET_ID, run_id, "raw.csv")
    return resolve_first_existing_path(raw_candidates) or raw_candidates[0]


def cache_data(func):
    if st is None:
        return func
    return st.cache_data(show_spinner=False)(func)


def cache_resource(func):
    if st is None:
        return func
    return st.cache_resource(show_spinner=False)(func)


def import_shap_module() -> Any:
    if not SHAP_AVAILABLE:
        raise ModuleNotFoundError("shap is not installed.")
    import shap

    return shap


def import_torch_modules() -> Tuple[Any, Any]:
    if not TORCH_AVAILABLE:
        raise ModuleNotFoundError("torch is not installed.")
    import torch
    from torch import nn

    return torch, nn


@cache_data
def load_rating_poster_catalog(root_str: str) -> Dict[str, Dict[str, str]]:
    root = Path(root_str)
    img_dir = root / "images"
    catalog: Dict[str, Dict[str, str]] = {}
    for rating, file_name in RATING_POSTER_IMAGE_FILES.items():
        poster_path = img_dir / file_name
        catalog[rating] = {
            "path": str(poster_path) if poster_path.exists() else "",
            "description": RATING_POSTER_DESCRIPTIONS.get(rating, ""),
        }
    return catalog


def _append_boundary_ring_coords(
    ring: List[Any],
    lon_values: List[Optional[float]],
    lat_values: List[Optional[float]],
) -> None:
    for pair in ring:
        if not isinstance(pair, list) or len(pair) < 2:
            continue
        lon = pair[0]
        lat = pair[1]
        try:
            lon_values.append(round(float(lon), 5))
            lat_values.append(round(float(lat), 5))
        except (TypeError, ValueError):
            continue
    if lon_values and lat_values:
        lon_values.append(None)
        lat_values.append(None)


@cache_data
def load_king_county_boundary_line_coords(root_str: str) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    root = Path(root_str)
    boundary_candidates = [
        root / "Data" / "reference" / "king_county_boundary.geojson",
        root / DEPLOY_BUNDLE_DIRNAME / "reference" / "king_county_boundary.geojson",
        root / "data" / "reference" / "king_county_boundary.geojson",
    ]
    boundary_path = resolve_first_existing_path(boundary_candidates)
    if boundary_path is None:
        return [], []

    with boundary_path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    lon_values: List[Optional[float]] = []
    lat_values: List[Optional[float]] = []
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry", {})
        geom_type = clean_text(geometry.get("type"))
        coords = geometry.get("coordinates", [])
        if geom_type == "Polygon":
            for ring in coords:
                _append_boundary_ring_coords(ring, lon_values, lat_values)
        elif geom_type == "MultiPolygon":
            for polygon in coords:
                for ring in polygon:
                    _append_boundary_ring_coords(ring, lon_values, lat_values)

    return lon_values, lat_values


def apply_global_styles() -> None:
    st.markdown(
        """
<style>
html, body, [class*="css"] {
  font-family: "Avenir Next", "IBM Plex Sans", "Trebuchet MS", "Segoe UI", sans-serif;
  background: #ffffff;
}
h1, h2, h3, h4, h5, h6 {
  font-weight: 700 !important;
  letter-spacing: 0.01em;
}
div[data-testid="stAppViewContainer"] {
  background: #ffffff;
}
section.main {
  background: #ffffff;
}
div[data-testid="stHeader"] {
  background: rgba(255, 255, 255, 0.92);
  border-bottom: 1px solid rgba(15, 45, 54, 0.06);
}
div[data-testid="stToolbar"] {
  right: 0.75rem;
}
.block-container {
  max-width: 1360px;
  padding-top: 1.35rem;
  padding-bottom: 3rem;
}
div[data-testid="stMetricLabel"] > div {
  font-weight: 600 !important;
  color: #5d7d85 !important;
}
div[data-testid="stMetricValue"] {
  font-weight: 800 !important;
  color: #0f2d36 !important;
}
[data-testid="stMarkdownContainer"] p {
  line-height: 1.5;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 0.75rem;
  background: transparent;
  border: none;
  border-bottom: 1px solid #d9e2e6;
  border-radius: 0;
  padding: 0 0 0.35rem;
  flex-wrap: nowrap;
  overflow-x: auto;
}
[data-testid="stTabs"] button[role="tab"] {
  padding: 0.3rem 0.05rem 0.75rem !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  font-weight: 650 !important;
  color: #67808a !important;
  background: transparent !important;
  box-shadow: none !important;
  white-space: nowrap !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: #0f2d36 !important;
  border-bottom-color: #0d8a98 !important;
  background: transparent !important;
}
[data-testid="stTabs"] button[role="tab"]:hover {
  color: #0f2d36 !important;
}
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input {
  border-radius: 10px !important;
  border: 1px solid #cae0e4 !important;
  background: #ffffff !important;
}
[data-testid="stButton"] button {
  border-radius: 8px !important;
  border: 1px solid #cbd7dc !important;
  font-weight: 650 !important;
  min-height: 2.5rem;
  padding: 0.55rem 0.9rem !important;
  white-space: normal !important;
  line-height: 1.2 !important;
  background: #ffffff !important;
  box-shadow: none !important;
}
[data-testid="stButton"] button[kind="primary"] {
  background: #0d8a98 !important;
  border-color: #0d8a98 !important;
  color: #ffffff !important;
}
[data-testid="stButton"] button > div {
  text-align: left;
}
[data-testid="stPlotlyChart"],
[data-testid="stImage"],
[data-testid="stPyplot"] {
  border: 1px solid #e2e8eb;
  border-radius: 12px;
  background: #ffffff;
  padding: 0.35rem;
  box-shadow: none;
}
[data-testid="stImage"] img {
  border-radius: 8px;
}
[data-testid="stExpander"] {
  border: 1px solid #e2e8eb !important;
  border-radius: 10px !important;
  background: #ffffff;
}
.stCaptionContainer,
[data-testid="stCaptionContainer"] {
  color: #607f86;
}
.stDataFrame {
  border: 1px solid #e2e8eb;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: none;
}
.detail-info-card {
  background: #ffffff;
  border:1px solid #e2e8eb;
  border-radius:10px;
  padding:12px 14px;
  min-height:116px;
  box-shadow: none;
}
.detail-info-title {
  color:#5d7d85;
  font-size:12px;
  font-weight:600;
}
.detail-info-value {
  color:#0f2d36;
  font-size:18px;
  font-weight:800;
  margin-top:4px;
}
.detail-info-note {
  color:#5d7d85;
  font-size:12px;
  line-height:1.45;
  margin-top:8px;
}
.detail-info-list {
  color:#5d7d85;
  font-size:12px;
  line-height:1.45;
  margin:8px 0 0 18px;
  padding:0;
}
.detail-info-list li {
  margin:0 0 4px;
}
.essay-card {
  background: #ffffff;
  border: 1px solid #e2e8eb;
  border-radius: 10px;
  padding: 14px 16px;
  box-shadow: none;
  margin-bottom: 12px;
}
.essay-card h4 {
  margin: 0 0 8px;
  color: #0f2d36;
}
.essay-card p {
  margin: 0;
  color: #36545c;
}
.takeaway-box {
  border-left: 3px solid #0d8a98;
  background: #f6fafb;
  padding: 10px 12px;
  border-radius: 8px;
  margin: 8px 0 18px;
}
.takeaway-box strong {
  color: #0f2d36;
}
.dictionary-note {
  color: #5d7d85;
  font-size: 12px;
}
.section-hero {
  margin: 0 0 1rem;
  padding: 0 0 0.9rem;
  border: none;
  border-bottom: 1px solid #e7edf0;
  border-radius: 0;
  background: transparent;
  box-shadow: none;
}
.section-kicker {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #6c838c;
  margin-bottom: 0.22rem;
}
.section-title {
  margin: 0;
  color: #0f2d36;
  font-size: 1.75rem;
  font-weight: 750;
  letter-spacing: -0.01em;
}
.section-subtitle {
  margin-top: 0.35rem;
  color: #60757d;
  line-height: 1.5;
  font-size: 0.96rem;
}
.subsection-label {
  margin: 0.75rem 0 0.4rem;
  color: #0f2d36;
  font-size: 1rem;
  font-weight: 700;
}
.panel-hint {
  margin: 0.15rem 0 0.55rem;
  color: #72858d;
  font-size: 0.88rem;
}
.glass-panel {
  border: 1px solid #e2e8eb;
  border-radius: 10px;
  background: #ffffff;
  padding: 0.9rem 1rem;
  box-shadow: none;
}
.compact-rule {
  margin: 0.9rem 0 1rem;
  border-top: 1px solid #e4edef;
}
.media-frame {
  border: 1px solid #dde8ea;
  border-radius: 18px;
  background: #ffffff;
  padding: 0.8rem;
  box-shadow: 0 6px 18px rgba(15, 45, 54, 0.05);
}
.nav-shell {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.9rem;
  margin: 0 0 0.7rem;
}
.nav-shell-title {
  font-size: 0.78rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: #5d7d85;
}
.nav-shell-note {
  font-size: 0.9rem;
  color: #49676f;
}
.nav-group-label {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  margin: 0 0 0.62rem;
  padding: 0.28rem 0.72rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.nav-group-public-label {
  color: #0d8a98;
  background: rgba(13, 138, 152, 0.10);
  border: 1px solid rgba(13, 138, 152, 0.18);
}
.nav-group-assignment-label {
  color: #0f2d36;
  background: rgba(15, 45, 54, 0.08);
  border: 1px solid rgba(15, 45, 54, 0.12);
}
div[data-testid="column"]:has(.nav-group-public-label),
div[data-testid="column"]:has(.nav-group-assignment-label) {
  background: linear-gradient(180deg, rgba(255,255,255,0.88) 0%, rgba(248,252,252,0.96) 100%);
  border: 1px solid #cae0e4;
  border-radius: 18px;
  padding: 0.88rem 0.95rem 0.84rem;
  box-shadow: 0 8px 24px rgba(15, 45, 54, 0.08);
}
div[data-testid="column"]:has(.nav-group-public-label) [data-testid="stButton"] button,
div[data-testid="column"]:has(.nav-group-assignment-label) [data-testid="stButton"] button {
  min-height: 4.1rem;
  border-radius: 14px !important;
  padding: 0.88rem 1rem !important;
  white-space: normal !important;
  line-height: 1.18 !important;
  font-size: 0.92rem !important;
  justify-content: flex-start !important;
  align-items: center !important;
  text-align: left !important;
  box-shadow: none !important;
}
div[data-testid="column"]:has(.nav-group-public-label) [data-testid="stButton"] button {
  border-color: #b6dfe4 !important;
  background: linear-gradient(180deg, #ffffff 0%, #f4fcfd 100%) !important;
}
div[data-testid="column"]:has(.nav-group-public-label) [data-testid="stButton"] button[kind="primary"] {
  background: linear-gradient(135deg, #16a2af, #0d8a98) !important;
  border-color: #0d8a98 !important;
  color: #f5feff !important;
}
div[data-testid="column"]:has(.nav-group-assignment-label) [data-testid="stButton"] button {
  border-color: #c8d8db !important;
  background: linear-gradient(180deg, #ffffff 0%, #f7fafb 100%) !important;
}
div[data-testid="column"]:has(.nav-group-assignment-label) [data-testid="stButton"] button[kind="primary"] {
  background: linear-gradient(135deg, #0f2d36, #1c5664) !important;
  border-color: #0f2d36 !important;
  color: #f5feff !important;
}
@media (max-width: 768px) {
  .section-hero {
    padding: 0 0 0.8rem;
    border-radius: 0;
  }
  .section-title {
    font-size: 1.35rem;
  }
  .section-subtitle {
    font-size: 0.9rem;
  }
  .nav-shell {
    display: block;
  }
  .nav-shell-note {
    margin-top: 0.3rem;
  }
  div[data-testid="column"]:has(.nav-group-public-label),
  div[data-testid="column"]:has(.nav-group-assignment-label) {
    padding: 0.72rem 0.78rem;
    border-radius: 14px;
  }
  div[data-testid="column"]:has(.nav-group-public-label) [data-testid="stButton"] button,
  div[data-testid="column"]:has(.nav-group-assignment-label) [data-testid="stButton"] button {
    min-height: 3.15rem;
    font-size: 0.82rem !important;
    padding: 0.66rem 0.56rem !important;
  }
  [data-testid="stButton"] button {
    width: 100%;
  }
  .stDataFrame {
    border-radius: 10px;
  }
  .detail-info-card {
    min-height: 98px;
    padding: 10px 11px;
  }
  .detail-info-value {
    font-size: 16px;
  }
  .essay-card {
    padding: 12px 13px;
    border-radius: 12px;
  }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_step_separator() -> None:
    st.markdown(
        "<div style='margin: 22px 0; border-top: 1px dashed #9fc8cf;'></div>",
        unsafe_allow_html=True,
    )


def format_detail_card_avg_red_title(risk_level: object) -> str:
    level = clean_text(risk_level)
    if level == "3":
        return "Avg Red Points in Recent 4 Inspections"
    if level in {"1", "2"}:
        return "Avg Red Points in Recent 2 Inspections"
    return "Avg Red Points in Recent Inspections"


def format_risk_category_card_value(risk_level: object) -> str:
    meta = RISK_CATEGORY_CARD_EXPLANATIONS.get(clean_text(risk_level), {})
    return clean_text(meta.get("label", "")) or "-"


def render_detail_info_card(title: str, value: str, note: str = "", bullets: Optional[List[str]] = None) -> None:
    note_html = (
        f"<div class='detail-info-note'>{escape(note)}</div>"
        if clean_text(note)
        else ""
    )
    bullet_items = bullets or []
    bullets_html = ""
    if bullet_items:
        bullets_html = "<ul class='detail-info-list'>" + "".join(
            f"<li>{escape(item)}</li>" for item in bullet_items if clean_text(item)
        ) + "</ul>"
    st.markdown(
        (
            "<div class='detail-info-card'>"
            f"<div class='detail-info-title'>{escape(title)}</div>"
            f"<div class='detail-info-value'>{escape(value)}</div>"
            f"{note_html}"
            f"{bullets_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_search_mode_buttons() -> str:
    current_mode = clean_text(st.session_state.get("search_mode", "keyword")).lower() or "keyword"
    valid_modes = {key for key, _ in SEARCH_MODE_OPTIONS}
    if current_mode not in valid_modes:
        current_mode = "keyword"
        st.session_state["search_mode"] = current_mode

    mode_weights = [max(1.0, min(1.25, len(mode_label) / 12.5)) for _, mode_label in SEARCH_MODE_OPTIONS]
    mode_cols = st.columns(mode_weights, gap="small")
    for col, (mode_key, mode_label) in zip(mode_cols, SEARCH_MODE_OPTIONS):
        translated_mode_label = tr(f"mode.{mode_key}") if st is not None else mode_label
        button_type = "primary" if current_mode == mode_key else "secondary"
        if col.button(
            translated_mode_label,
            key=f"search_mode_btn_{mode_key}",
            use_container_width=True,
            type=button_type,
        ):
            if current_mode != mode_key:
                st.session_state["search_mode"] = mode_key
                st.session_state["search_map_nonce"] = int(st.session_state.get("search_map_nonce", 0)) + 1
                st.session_state["search_results_table_nonce"] = int(
                    st.session_state.get("search_results_table_nonce", 0)
                ) + 1
                st.rerun()
    return clean_text(st.session_state.get("search_mode", current_mode)).lower() or "keyword"


def compute_map_center_zoom(
    map_df: pd.DataFrame,
    selected_business_id: str = "",
) -> Tuple[float, float, float]:
    if map_df.empty:
        return (
            float(DEFAULT_KING_COUNTY_MAP_CENTER["lat"]),
            float(DEFAULT_KING_COUNTY_MAP_CENTER["lon"]),
            8.8,
        )

    lat_series = pd.to_numeric(map_df["latitude"], errors="coerce").dropna()
    lon_series = pd.to_numeric(map_df["longitude"], errors="coerce").dropna()
    if lat_series.empty or lon_series.empty:
        return (
            float(DEFAULT_KING_COUNTY_MAP_CENTER["lat"]),
            float(DEFAULT_KING_COUNTY_MAP_CENTER["lon"]),
            8.8,
        )

    center_lat = float(lat_series.mean())
    center_lon = float(lon_series.mean())
    zoom = 9.2

    if selected_business_id:
        selected_mask = map_df["business_id"].astype(str).str.strip() == selected_business_id
        if selected_mask.any():
            selected_row = map_df[selected_mask].iloc[0]
            sel_lat = pd.to_numeric(pd.Series([selected_row.get("latitude")]), errors="coerce").iloc[0]
            sel_lon = pd.to_numeric(pd.Series([selected_row.get("longitude")]), errors="coerce").iloc[0]
            if not pd.isna(sel_lat) and not pd.isna(sel_lon):
                center_lat = float(sel_lat)
                center_lon = float(sel_lon)
                if len(map_df) == 1:
                    return center_lat, center_lon, 13.8
                zoom = 12.2

    lat_span = float(lat_series.max() - lat_series.min()) if len(lat_series) > 1 else 0.0
    lon_span = float(lon_series.max() - lon_series.min()) if len(lon_series) > 1 else 0.0
    span = max(lat_span, lon_span)
    if span > 0.9:
        zoom = min(zoom, 8.4)
    elif span > 0.45:
        zoom = min(zoom, 9.0)
    elif span > 0.22:
        zoom = min(zoom, 10.0)
    elif span > 0.08:
        zoom = min(zoom, 11.2)
    elif span > 0.03:
        zoom = min(zoom, 12.4)
    else:
        zoom = max(zoom, 13.0)
    return center_lat, center_lon, zoom


def build_search_map_figure(map_df: pd.DataFrame, selected_business_id: str = "") -> Any:
    center_lat, center_lon, zoom = compute_map_center_zoom(map_df, selected_business_id)
    selected_indices: List[int] = []
    if selected_business_id:
        selected_indices = (
            map_df.index[map_df["business_id"].astype(str).str.strip() == selected_business_id]
            .tolist()
        )

    colors = [RATING_COLOR_MAP.get(clean_text(value), "#ced4da") for value in map_df["latest_rating"]]
    customdata = np.array(
        [
            map_df["business_id"].astype(str).tolist(),
            map_df["display_name"].astype(str).tolist(),
            map_df["full_address_clean"].astype(str).tolist(),
            map_df["latest_rating"].astype(str).tolist(),
        ]
    ).T
    boundary_lon, boundary_lat = load_king_county_boundary_line_coords(str(app_root()))
    marker_trace = go.Scattermapbox(
        lat=pd.to_numeric(map_df["latitude"], errors="coerce"),
        lon=pd.to_numeric(map_df["longitude"], errors="coerce"),
        mode="markers",
        customdata=customdata,
        marker={
            "size": 10,
            "color": colors,
            "opacity": 0.82,
        },
        selectedpoints=selected_indices if selected_indices else None,
        selected={"marker": {"size": 16, "opacity": 1.0}},
        unselected={"marker": {"opacity": 0.36}},
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "%{customdata[2]}<br>"
            "Rating: %{customdata[3]}<br>"
            "Lat: %{lat:.4f} | Lon: %{lon:.4f}<extra></extra>"
        ),
    )
    fig = go.Figure()
    if boundary_lon and boundary_lat:
        fig.add_trace(
            go.Scattermapbox(
                lat=boundary_lat,
                lon=boundary_lon,
                mode="lines",
                hoverinfo="skip",
                line={"color": KING_COUNTY_BOUNDARY_COLOR, "width": 2.2},
                opacity=0.95,
                showlegend=False,
            )
        )
    fig.add_trace(marker_trace)
    fig.update_layout(
        mapbox={
            "style": "carto-positron",
            "center": {"lat": center_lat, "lon": center_lon},
            "zoom": zoom,
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=382,
        clickmode="event+select",
        dragmode="pan",
        showlegend=False,
        uirevision="search-map",
    )
    return fig


@cache_data
def load_risk_description_lookups(
    root_str: str, run_id: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root_str)
    raw_csv = resolve_first_existing_path(
        dataset_path_candidates(root, "bronze", DATASET_ID, run_id, "raw.csv")
    )
    if raw_csv is None:
        return (
            pd.DataFrame(columns=["inspection_serial_num", "risk_description_raw"]),
            pd.DataFrame(columns=["business_id", "inspection_date", "inspection_type", "risk_description_raw"]),
        )

    raw_df = pd.read_csv(raw_csv, dtype=str).fillna("")
    if "description" not in raw_df.columns:
        return (
            pd.DataFrame(columns=["inspection_serial_num", "risk_description_raw"]),
            pd.DataFrame(columns=["business_id", "inspection_date", "inspection_type", "risk_description_raw"]),
        )

    raw_df["risk_description_raw"] = raw_df["description"].astype(str).str.strip()
    serial_lookup = (
        raw_df[raw_df["inspection_serial_num"].astype(str).str.strip() != ""]
        .groupby("inspection_serial_num", dropna=False)["risk_description_raw"]
        .agg(pick_mode_text)
        .reset_index()
    )
    group_lookup = (
        raw_df[
            ["business_id", "inspection_date", "inspection_type", "risk_description_raw"]
        ]
        .groupby(["business_id", "inspection_date", "inspection_type"], dropna=False)["risk_description_raw"]
        .agg(pick_mode_text)
        .reset_index()
    )
    return serial_lookup, group_lookup


def attach_risk_metadata(
    events_df: pd.DataFrame, root_str: str, payload: Dict[str, str]
) -> pd.DataFrame:
    run_id = clean_text(payload.get("run_id", ""))
    if not run_id:
        events_df["risk_description_raw"] = events_df.get("risk_description_raw", "")
        events_df["risk_level"] = events_df.get("risk_level", "")
        events_df["risk_level_label"] = events_df["risk_level"].map(format_risk_level_label)
        return events_df

    serial_lookup, group_lookup = load_risk_description_lookups(root_str, run_id)
    out = events_df.copy()
    out["risk_description_raw"] = out.get("risk_description_raw", "").astype(str)

    if not serial_lookup.empty:
        out = out.merge(
            serial_lookup.rename(columns={"risk_description_raw": "risk_description_by_serial"}),
            on="inspection_serial_num",
            how="left",
        )
    else:
        out["risk_description_by_serial"] = ""

    if not group_lookup.empty:
        out = out.merge(
            group_lookup.rename(columns={"risk_description_raw": "risk_description_by_group"}),
            on=["business_id", "inspection_date", "inspection_type"],
            how="left",
        )
    else:
        out["risk_description_by_group"] = ""

    out["risk_description_raw"] = out["risk_description_raw"].where(
        out["risk_description_raw"].astype(str).str.strip() != "",
        out["risk_description_by_serial"],
    )
    out["risk_description_raw"] = out["risk_description_raw"].where(
        out["risk_description_raw"].astype(str).str.strip() != "",
        out["risk_description_by_group"],
    )
    out["risk_level"] = out.get("risk_level", "").astype(str)
    parsed_level = out["risk_description_raw"].map(parse_risk_level_from_description)
    out["risk_level"] = out["risk_level"].where(
        out["risk_level"].astype(str).str.strip() != "",
        parsed_level,
    )

    out = out.sort_values(
        by=["business_id", "inspection_date_dt", "inspection_event_id"],
        ascending=[True, True, True],
        na_position="first",
    ).copy()
    risk_series = out["risk_level"].replace("", pd.NA)
    risk_series = risk_series.groupby(out["business_id"]).ffill()
    risk_series = risk_series.groupby(out["business_id"]).bfill()
    out["risk_level"] = risk_series.fillna("").astype(str)
    out["risk_level_label"] = out["risk_level"].map(format_risk_level_label)
    out = out.drop(columns=["risk_description_by_serial", "risk_description_by_group"], errors="ignore")
    return out


def append_effective_rating_columns(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return events_df

    groups: List[pd.DataFrame] = []
    for _, group in events_df.groupby("business_id", dropna=False, sort=False):
        g = group.sort_values(
            by=["inspection_date_dt", "inspection_event_id"],
            ascending=[True, True],
            na_position="first",
        ).copy()
        routine_red_history: List[float] = []
        avg_values: List[float] = []
        window_targets: List[float] = []
        used_counts: List[float] = []
        has_required_window_flags: List[float] = []
        recent_closed_flags: List[float] = []
        recent_return_counts: List[float] = []
        recalculated_nums: List[str] = []
        recalculated_labels: List[str] = []
        official_nums: List[str] = []
        official_labels: List[str] = []
        effective_nums: List[str] = []
        effective_labels: List[str] = []
        rating_sources: List[str] = []
        closure_dates: List[pd.Timestamp] = []
        return_dates: List[pd.Timestamp] = []

        for row in g.itertuples(index=False):
            risk_level = clean_text(getattr(row, "risk_level", ""))
            window_target = RISK_LEVEL_TO_ROUTINE_WINDOW.get(risk_level)
            inspection_type = clean_text(getattr(row, "inspection_type", ""))
            red_points = getattr(row, "red_points_total", np.nan)
            inspection_date_dt = getattr(row, "inspection_date_dt", pd.NaT)
            closed_flag = clean_text(getattr(row, "inspection_closed_business", "")).lower() in {"1", "true", "yes"}
            if inspection_type == ROUTINE_INSPECTION_TYPE and not pd.isna(red_points):
                routine_red_history.append(float(red_points))
            if pd.notna(inspection_date_dt):
                if closed_flag:
                    closure_dates.append(inspection_date_dt)
                if inspection_type == RETURN_INSPECTION_TYPE:
                    return_dates.append(inspection_date_dt)

            recent_values: List[float] = []
            if window_target and routine_red_history:
                recent_values = routine_red_history[-window_target:]
                avg_red = float(sum(recent_values) / len(recent_values))
                used_count = len(recent_values)
            else:
                avg_red = float("nan")
                used_count = 0
            has_required_window = bool(window_target and used_count >= int(window_target))
            candidate_label = rating_label_from_avg_red_points(risk_level, avg_red) if used_count > 0 else ""

            recent_closed = False
            recent_return_count = 0
            if pd.notna(inspection_date_dt):
                cutoff = inspection_date_dt - pd.Timedelta(days=RATING_CLOSURE_LOOKBACK_DAYS)
                recent_closed = any(date >= cutoff for date in closure_dates)
                recent_return_count = sum(1 for date in return_dates if date >= cutoff)

            if recent_closed or recent_return_count >= MULTIPLE_RETURN_INSPECTION_THRESHOLD:
                calculated_label = "Needs to Improve"
                rating_source = "dashboard_recent_closure_or_multiple_returns"
            elif candidate_label:
                calculated_label = candidate_label
                rating_source = "dashboard_recent_routine_average"
            else:
                calculated_label = ""
                rating_source = "rating_not_available"

            recalculated_num = GRADE_LABEL_TO_CODE.get(calculated_label, "") if calculated_label else ""
            official_label = rating_label_from_values(
                getattr(row, "grade", ""), getattr(row, "grade_label", "")
            )
            official_num = clean_text(getattr(row, "grade", ""))
            if official_num not in GRADE_CODE_TO_LABEL:
                official_num = GRADE_LABEL_TO_CODE.get(official_label, "")

            if calculated_label:
                effective_num = recalculated_num
                effective_label = calculated_label
            else:
                effective_num = ""
                effective_label = RATING_NOT_AVAILABLE_LABEL

            avg_values.append(avg_red)
            window_targets.append(float(window_target) if window_target else np.nan)
            used_counts.append(float(used_count) if used_count else np.nan)
            has_required_window_flags.append(1.0 if has_required_window else 0.0)
            recent_closed_flags.append(1.0 if recent_closed else 0.0)
            recent_return_counts.append(float(recent_return_count) if recent_return_count else np.nan)
            recalculated_nums.append(recalculated_num)
            recalculated_labels.append(calculated_label)
            official_nums.append(official_num)
            official_labels.append(official_label)
            effective_nums.append(effective_num)
            effective_labels.append(effective_label)
            rating_sources.append(rating_source)

        g["rating_avg_red_points_recent"] = avg_values
        g["rating_window_target"] = window_targets
        g["rating_recent_routine_count_used"] = used_counts
        g["rating_has_required_routine_window"] = has_required_window_flags
        g["rating_recent_closure_90d_flag"] = recent_closed_flags
        g["rating_recent_return_inspection_count_90d"] = recent_return_counts
        g["recalculated_rating_num"] = recalculated_nums
        g["recalculated_rating_label"] = recalculated_labels
        g["official_rating_num"] = official_nums
        g["official_rating_label"] = official_labels
        g["effective_rating_num"] = effective_nums
        g["effective_rating_label"] = effective_labels
        g["effective_rating_source"] = rating_sources
        groups.append(g)

    return pd.concat(groups, ignore_index=True)


def resolve_prepared_app_data_paths(root: Path) -> Tuple[Path, Path, Path]:
    prepared_dir = root / DEPLOY_BUNDLE_DIRNAME / "prepared"
    return (
        prepared_dir / "events_app.parquet",
        prepared_dir / "violations_app.parquet",
        prepared_dir / "summary_app.parquet",
    )


@cache_data
def load_data(root_str: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    root = Path(root_str)
    payload = load_latest_run_payload(root)
    prepared_events_path, prepared_violations_path, _ = resolve_prepared_app_data_paths(root)
    if prepared_events_path.exists() and prepared_violations_path.exists():
        events_df = pd.read_parquet(prepared_events_path)
        violations_df = pd.read_parquet(prepared_violations_path)
        if "inspection_date_dt" in events_df.columns:
            events_df["inspection_date_dt"] = pd.to_datetime(events_df["inspection_date_dt"], errors="coerce")
        if "inspection_date_dt" in violations_df.columns:
            violations_df["inspection_date_dt"] = pd.to_datetime(
                violations_df["inspection_date_dt"], errors="coerce"
            )
        return events_df, violations_df, payload

    silver_event_csv, dashboard_violation_csv = resolve_data_paths(root, payload)

    events_df = pd.read_csv(silver_event_csv, dtype=str).fillna("")
    violations_df = pd.read_csv(dashboard_violation_csv, dtype=str).fillna("")

    ensure_columns(
        events_df,
        {
            "business_id": "",
            "business_name_official": "",
            "business_name_alt": "",
            "search_name_norm": "",
            "full_address_clean": "",
            "city_canonical": "",
            "city_raw_original": "",
            "city_scope": "",
            "city_cleaning_reason": "",
            "zip_code": "",
            "inspection_date": "",
            "inspection_type": "",
            "inspection_result": "",
            "inspection_score": "",
            "grade": "",
            "grade_label": "",
            "risk_description_raw": "",
            "risk_level": "",
            "red_points_total": "",
            "blue_points_total": "",
            "violation_count_total": "",
        },
    )

    ensure_columns(
        violations_df,
        {
            "row_id": "",
            "business_id": "",
            "business_name_official": "",
            "business_name_alt": "",
            "search_name_norm": "",
            "full_address_clean": "",
            "city_canonical": "",
            "city_raw_original": "",
            "city_scope": "",
            "city_cleaning_reason": "",
            "zip_code": "",
            "inspection_date": "",
            "violation_type": "",
            "violation_code": "",
            "violation_points": "",
            "violation_desc_clean": "",
            "action_priority": "",
            "action_category": "",
            "action_summary_en": "",
            "action_summary_zh": "",
            "action_steps_zh": "",
            "safe_food_handling_refs": "",
            "action_source": "",
        },
    )

    for col in ["inspection_score", "red_points_total", "blue_points_total", "violation_count_total"]:
        events_df[col] = pd.to_numeric(events_df[col], errors="coerce")

    city_zip_lookup = build_zip_to_locality_lookup(events_df)
    events_df = apply_city_quality_rules(events_df, city_zip_lookup)
    violations_df = apply_city_quality_rules(violations_df, city_zip_lookup)

    events_df["inspection_date_dt"] = pd.to_datetime(events_df["inspection_date"], errors="coerce")
    events_df["grade_num"] = pd.to_numeric(events_df["grade"], errors="coerce")

    events_df["grade_label_standard"] = events_df["grade"].map(GRADE_CODE_TO_LABEL)
    fallback_labels = events_df["grade_label"].astype(str).str.strip()
    fallback_labels = fallback_labels.str.lower().map(LABEL_NORMALIZATION_MAP).fillna(fallback_labels)
    events_df["grade_label_standard"] = events_df["grade_label_standard"].fillna(fallback_labels)
    events_df = attach_risk_metadata(events_df, root_str, payload)
    events_df = append_effective_rating_columns(events_df)

    grade_high_risk = events_df["grade"].astype(str).str.strip().isin(HIGH_RISK_GRADES)
    red_points_high_risk = events_df["red_points_total"].fillna(0) >= HIGH_RISK_RED_POINTS_THRESHOLD
    events_df["is_high_risk"] = (grade_high_risk | red_points_high_risk).astype(int)

    violations_df["inspection_date_dt"] = pd.to_datetime(
        violations_df["inspection_date"], errors="coerce"
    )
    violations_df["violation_points_num"] = pd.to_numeric(
        violations_df["violation_points"], errors="coerce"
    )

    return events_df, violations_df, payload


@cache_data
def load_prepared_business_summary(root_str: str) -> pd.DataFrame:
    _, _, prepared_summary_path = resolve_prepared_app_data_paths(Path(root_str))
    if prepared_summary_path.exists():
        return pd.read_parquet(prepared_summary_path)
    return pd.DataFrame()


def display_name_frame(df: pd.DataFrame) -> pd.Series:
    name = df["business_name_official"].where(
        df["business_name_official"].astype(str).str.strip() != "", df["business_name_alt"]
    )
    return name.where(name.astype(str).str.strip() != "", df["business_id"])


def build_business_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    ordered = events_df.sort_values(by=["inspection_date_dt"], ascending=True, na_position="first")
    latest = ordered.drop_duplicates(subset=["business_id"], keep="last").copy()

    counts = events_df.groupby("business_id", dropna=False).size().rename("inspection_count")
    avg_score = (
        events_df.groupby("business_id", dropna=False)["inspection_score"].mean().round(2).rename("avg_inspection_score")
    )

    summary = latest.merge(counts, left_on="business_id", right_index=True, how="left")
    summary = summary.merge(avg_score, left_on="business_id", right_index=True, how="left")

    summary["inspection_count"] = summary["inspection_count"].fillna(0).astype(int)
    summary["display_name"] = display_name_frame(summary)
    summary["city_display"] = summary["city_canonical"].map(format_city_name)

    summary["latest_inspection_date"] = summary["inspection_date"]
    summary["latest_score"] = summary["inspection_score"]
    summary["latest_grade"] = summary["grade"]
    summary["latest_rating"] = summary["effective_rating_label"].astype(str).str.strip()
    summary["latest_rating"] = summary["latest_rating"].replace("", RATING_NOT_AVAILABLE_LABEL)
    summary["latest_risk_level"] = summary["risk_level"].astype(str).str.strip()
    summary["latest_risk_level_label"] = summary["risk_level_label"].astype(str).str.strip()
    summary["latest_rating_avg_red_points"] = pd.to_numeric(
        summary["rating_avg_red_points_recent"], errors="coerce"
    )
    summary["latest_rating_window_target"] = pd.to_numeric(
        summary["rating_window_target"], errors="coerce"
    )
    summary["latest_rating_routine_n"] = pd.to_numeric(
        summary["rating_recent_routine_count_used"], errors="coerce"
    )
    summary["latest_rating_source"] = summary["effective_rating_source"].astype(str).str.strip()
    summary["latest_rating_source_label"] = summary["latest_rating_source"].map(format_rating_source_label)
    summary["latest_recalculated_rating"] = summary["recalculated_rating_label"].astype(str).str.strip()
    summary["latest_official_rating"] = summary["official_rating_label"].astype(str).str.strip()

    summary["display_name_norm"] = summary["display_name"].map(normalize_search_text)
    summary["address_norm"] = summary["full_address_clean"].map(normalize_search_text)
    summary["city_norm"] = summary["city_display"].map(normalize_search_text)
    summary["search_blob_norm"] = (
        summary["display_name"].astype(str)
        + " "
        + summary["business_name_alt"].astype(str)
        + " "
        + summary["full_address_clean"].astype(str)
        + " "
        + summary["city_display"].astype(str)
        + " "
        + summary["zip_code"].astype(str)
        + " "
        + summary["search_name_norm"].astype(str)
    ).map(normalize_search_text)

    return summary


def fuzzy_match_value(query_norm: str, text_norm: str, threshold: float = 0.86) -> bool:
    if not query_norm or not text_norm:
        return False
    if query_norm in text_norm:
        return True

    ratio = SequenceMatcher(None, query_norm, text_norm).ratio()
    if ratio >= threshold:
        return True

    q_tokens = [token for token in query_norm.split() if len(token) >= 4]
    if not q_tokens:
        return False
    t_tokens = text_norm.split()
    for q in q_tokens:
        for t in t_tokens:
            if abs(len(t) - len(q)) > 2:
                continue
            if SequenceMatcher(None, q, t).ratio() >= 0.88:
                return True
    return False


def filter_businesses(
    summary_df: pd.DataFrame,
    query: str,
    city: str,
    zip_code: str,
    rating_labels: List[str],
) -> pd.DataFrame:
    df = summary_df.copy()

    if city != "All":
        df = df[df["city_display"] == city]

    zip_code = zip_code.strip()
    if zip_code:
        df = df[df["zip_code"].str.contains(zip_code, na=False)]

    if rating_labels:
        df = df[df["latest_rating"].isin(rating_labels)]

    query = query.strip()
    if query:
        query_lower = query.lower()
        query_norm = normalize_search_text(query)

        direct_mask = (
            df["display_name"].str.lower().str.contains(query_lower, na=False)
            | df["business_name_alt"].str.lower().str.contains(query_lower, na=False)
            | df["full_address_clean"].str.lower().str.contains(query_lower, na=False)
            | df["city_display"].str.lower().str.contains(query_lower, na=False)
            | df["city_canonical"].str.lower().str.contains(query_lower, na=False)
            | df["zip_code"].str.contains(query, na=False)
            | df["search_name_norm"].str.contains(query_norm, na=False)
        )

        matched_count = int(direct_mask.sum())
        if matched_count < 60:
            fuzzy_threshold = 0.9 if len(query_norm) <= 4 else 0.86
            fuzzy_mask = df["search_blob_norm"].map(
                lambda text: fuzzy_match_value(query_norm, str(text), threshold=fuzzy_threshold)
            )
            direct_mask = direct_mask | fuzzy_mask

        df = df[direct_mask]

    return df.sort_values(
        by=["inspection_date_dt", "inspection_count"],
        ascending=[False, False],
        na_position="last",
    )


def format_source_label(value: str) -> str:
    mapping = {
        "dictionary_code_type": "Dictionary match",
        "fallback_rule": "Rule fallback",
        "missing_violation_code": "Missing violation code",
    }
    return mapping.get(value, value)


def rating_explanation_markdown(active_rating: str, official_rating: str) -> str:
    lines: List[str] = []
    for rating in ["Excellent", "Good", "Okay", "Needs to Improve"]:
        prefix = "-> " if rating == active_rating else "- "
        label = rating
        if official_rating and rating == official_rating and rating != active_rating:
            label += " (official county grade)"
        lines.append(f"{prefix}**{label}**: {OFFICIAL_RATING_EXPLANATIONS[rating]}")
    if active_rating == RATING_NOT_AVAILABLE_LABEL:
        lines.append(
            f"- **{RATING_NOT_AVAILABLE_LABEL}**: No county rating is currently available in the published record."
        )
    return "\n".join(lines)


def summary_filters(events_df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp, List[str]]:
    valid_dates = events_df["inspection_date_dt"].dropna()
    if valid_dates.empty:
        today = pd.Timestamp.today().normalize()
        return today, today, []

    min_date = valid_dates.min().normalize()
    max_date = valid_dates.max().normalize()
    default_start = max(min_date, max_date - pd.Timedelta(days=365))

    c1, c2, c3 = st.columns([1, 1, 2])
    start_date = c1.date_input(
        "Start date",
        value=default_start.date(),
        min_value=min_date.date(),
        max_value=max_date.date(),
        key="summary_start_date",
    )
    end_date = c2.date_input(
        "End date",
        value=max_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date(),
        key="summary_end_date",
    )

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    city_options = sorted(
        [city for city in events_df["city_display"].dropna().unique().tolist() if str(city).strip()]
    )
    selected_cities = c3.multiselect("Cities (optional)", options=city_options, default=[])

    return pd.Timestamp(start_date), pd.Timestamp(end_date), selected_cities


def apply_summary_filters(
    events_df: pd.DataFrame,
    violations_df: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    selected_cities: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    events_mask = (
        events_df["inspection_date_dt"].notna()
        & (events_df["inspection_date_dt"] >= start_ts)
        & (events_df["inspection_date_dt"] <= (end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)))
    )
    filtered_events = events_df[events_mask].copy()

    violations_mask = (
        violations_df["inspection_date_dt"].notna()
        & (violations_df["inspection_date_dt"] >= start_ts)
        & (violations_df["inspection_date_dt"] <= (end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)))
    )
    filtered_violations = violations_df[violations_mask].copy()

    if selected_cities:
        filtered_events = filtered_events[filtered_events["city_display"].isin(selected_cities)]
        filtered_violations = filtered_violations[filtered_violations["city_display"].isin(selected_cities)]

    return filtered_events, filtered_violations


def score_to_rating_band(score_value: float) -> str:
    if pd.isna(score_value):
        return ""
    score = float(score_value)
    if score <= 10:
        return "Excellent"
    if score <= 25:
        return "Good"
    if score <= 45:
        return "Okay"
    return "Needs to Improve"


def compute_monthly_rating_changes(
    events_df: pd.DataFrame, month_period: pd.Period, source: str = "official_grade"
) -> pd.DataFrame:
    grade_df = events_df.copy()
    grade_df = grade_df[grade_df["inspection_date_dt"].notna()].copy()

    if source == "official_grade":
        grade_df["rating_num"] = pd.to_numeric(grade_df["effective_rating_num"], errors="coerce")
        grade_df["rating_label"] = grade_df["effective_rating_label"].astype(str).str.strip()
    else:
        grade_df["rating_label"] = grade_df["inspection_score"].map(score_to_rating_band)
        grade_df["rating_num"] = grade_df["rating_label"].map(GRADE_LABEL_TO_CODE)
        grade_df["rating_num"] = pd.to_numeric(grade_df["rating_num"], errors="coerce")

    grade_df = grade_df[grade_df["rating_num"].isin([1, 2, 3, 4])].copy()
    if grade_df.empty:
        return pd.DataFrame()

    grade_df = grade_df.sort_values(["business_id", "inspection_date_dt"])
    grade_df["prev_rating_num"] = grade_df.groupby("business_id")["rating_num"].shift(1)
    grade_df["prev_date"] = grade_df.groupby("business_id")["inspection_date_dt"].shift(1)
    grade_df["display_name"] = display_name_frame(grade_df)
    grade_df["city_display"] = grade_df["city_canonical"].map(format_city_name)

    month_start = month_period.to_timestamp()
    month_end = (month_period + 1).to_timestamp() - pd.Timedelta(seconds=1)

    in_month = grade_df[
        (grade_df["inspection_date_dt"] >= month_start)
        & (grade_df["inspection_date_dt"] <= month_end)
    ].copy()
    if in_month.empty:
        return pd.DataFrame()

    in_month_latest = in_month.sort_values("inspection_date_dt").drop_duplicates(
        subset=["business_id"], keep="last"
    )
    changed = in_month_latest[
        in_month_latest["prev_rating_num"].notna()
        & (in_month_latest["rating_num"] != in_month_latest["prev_rating_num"])
    ].copy()
    if changed.empty:
        return pd.DataFrame()

    changed["from_rating"] = changed["prev_rating_num"].astype(int).astype(str).map(
        GRADE_CODE_TO_LABEL
    )
    changed["to_rating"] = changed["rating_num"].astype(int).astype(str).map(
        GRADE_CODE_TO_LABEL
    )
    changed["change_type"] = changed.apply(
        lambda row: "Improved" if row["rating_num"] < row["prev_rating_num"] else "Declined",
        axis=1,
    )
    changed["transition"] = changed["from_rating"] + " -> " + changed["to_rating"]

    return changed[
        [
            "display_name",
            "city_display",
            "full_address_clean",
            "inspection_date",
            "prev_date",
            "from_rating",
            "to_rating",
            "transition",
            "change_type",
        ]
    ]


def latest_events_by_business(events_df: pd.DataFrame) -> pd.DataFrame:
    latest = (
        events_df.sort_values("inspection_date_dt", ascending=True, na_position="first")
        .drop_duplicates(subset=["business_id"], keep="last")
        .copy()
    )
    latest["display_name"] = display_name_frame(latest)
    latest["latest_rating"] = latest["effective_rating_label"].astype(str).str.strip().replace(
        "", RATING_NOT_AVAILABLE_LABEL
    )
    return latest


def business_profile_stats(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    agg = (
        events_df.groupby("business_id", dropna=False)
        .agg(
            inspection_count=("business_id", "size"),
            high_risk_rate=("is_high_risk", "mean"),
            avg_inspection_score=("inspection_score", "mean"),
            avg_red_points=("red_points_total", "mean"),
            avg_blue_points=("blue_points_total", "mean"),
        )
        .reset_index()
    )
    latest = latest_events_by_business(events_df)[
        [
            "business_id",
            "display_name",
            "city_display",
            "full_address_clean",
            "inspection_date",
            "latest_rating",
            "inspection_score",
            "red_points_total",
            "blue_points_total",
            "violation_count_total",
            "is_high_risk",
        ]
    ].copy()
    prof = agg.merge(latest, on="business_id", how="left")
    prof["safety_score"] = (
        (1.0 - prof["high_risk_rate"].fillna(0.0)) * 100.0
        - prof["avg_red_points"].fillna(0.0) * 1.2
        - prof["avg_blue_points"].fillna(0.0) * 0.35
    )
    return prof


def build_official_grade_gap_df(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    latest = (
        events_df.sort_values(["inspection_date_dt", "inspection_event_id"], ascending=[True, True], na_position="first")
        .drop_duplicates(subset=["business_id"], keep="last")
        .copy()
    )
    latest["display_name"] = display_name_frame(latest)
    latest["official_rating_label"] = latest["official_rating_label"].astype(str).str.strip()
    latest["dashboard_rating_label"] = latest["effective_rating_label"].astype(str).str.strip()
    latest = latest[
        latest["official_rating_label"].ne("")
        & latest["dashboard_rating_label"].ne("")
        & (latest["official_rating_label"] != latest["dashboard_rating_label"])
    ].copy()
    if latest.empty:
        return latest
    latest["city_display"] = latest["city_canonical"].map(format_city_name)
    latest["risk_category"] = latest["risk_level_label"].astype(str).str.strip()
    latest["dashboard_vs_official"] = (
        latest["dashboard_rating_label"] + " vs " + latest["official_rating_label"]
    )
    latest["avg_red_points_used"] = pd.to_numeric(latest["rating_avg_red_points_recent"], errors="coerce").round(2)
    latest["routine_inspections_used"] = pd.to_numeric(
        latest["rating_recent_routine_count_used"], errors="coerce"
    )
    return latest.sort_values(
        by=["inspection_date_dt", "display_name"], ascending=[False, True], na_position="last"
    )


def sanitize_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_json_like(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json_like(v) for v in value]
    if isinstance(value, float):
        if pd.isna(value) or np.isinf(value):
            return ""
    return value


@cache_data
def load_king_county_quality_bundle(root_str: str, run_id: str) -> Dict[str, Any]:
    root = Path(root_str)
    analysis_dir = root / "outputs" / "analysis"
    run_stamp = clean_text(run_id)[:8]

    issue_columns = [
        "category",
        "issue",
        "count",
        "denominator",
        "share_pct",
        "severity",
        "why_it_matters",
        "suggested_action",
        "owner",
    ]

    audit_snapshot: Dict[str, Any] = {}
    if run_stamp:
        audit_path = analysis_dir / f"db_quality_audit_{run_stamp}.json"
        if audit_path.exists():
            try:
                audit_snapshot = sanitize_json_like(json.loads(audit_path.read_text(encoding="utf-8")))
            except Exception:
                audit_snapshot = {}

    samples: Dict[str, Any] = {}
    if run_stamp:
        samples_path = analysis_dir / f"db_quality_samples_{run_stamp}.json"
        if samples_path.exists():
            try:
                samples = sanitize_json_like(json.loads(samples_path.read_text(encoding="utf-8")))
            except Exception:
                samples = {}

    issue_df = pd.DataFrame(columns=issue_columns)
    if run_stamp:
        issue_path = analysis_dir / f"king_county_issue_catalog_{run_stamp}.csv"
        if issue_path.exists():
            try:
                issue_df = pd.read_csv(issue_path)
            except Exception:
                issue_df = pd.DataFrame(columns=issue_columns)
    issue_df = ensure_columns(issue_df, {col: "" for col in issue_columns})
    issue_df = issue_df[issue_columns].copy()

    def to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def safe_rate(num_value: Any, den_value: Any) -> float:
        num = to_float(num_value)
        den = to_float(den_value)
        if pd.isna(num) or pd.isna(den) or den <= 0:
            return float("nan")
        return float(num / den)

    history_rows: List[Dict[str, Any]] = []
    for audit_file in sorted(analysis_dir.glob("db_quality_audit_*.json")):
        try:
            snap = json.loads(audit_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        snap_run_id = clean_text(snap.get("run_id", ""))
        rows_raw = to_float(snap.get("rows_raw", ""))
        run_stamp_i = clean_text(audit_file.stem.split("_")[-1])[:8]

        true_mismatch_rate = float("nan")
        if run_stamp_i:
            issue_path_i = analysis_dir / f"king_county_issue_catalog_{run_stamp_i}.csv"
            if issue_path_i.exists():
                try:
                    issue_i = pd.read_csv(issue_path_i)
                    issue_text = issue_i.get("issue", pd.Series(dtype=str)).astype(str)
                    mask = issue_text.str.contains("true mismatch excluding", case=False, na=False)
                    if mask.any():
                        row = issue_i[mask].iloc[0]
                        true_mismatch_rate = safe_rate(row.get("count", np.nan), row.get("denominator", np.nan))
                except Exception:
                    true_mismatch_rate = float("nan")

        history_rows.append(
            {
                "run_id": snap_run_id or run_stamp_i,
                "rows_raw": to_float(snap.get("rows_raw", np.nan)),
                "city_outside_rate": safe_rate(
                    snap.get("city_known_outside_rows", np.nan), rows_raw
                ),
                "city_unknown_rate": safe_rate(
                    snap.get("city_unknown_rows", np.nan), rows_raw
                ),
                "date_parse_fail_rate": safe_rate(
                    snap.get("raw_date_parse_fail_rows", np.nan), rows_raw
                ),
                "violation_truncated_rate": safe_rate(
                    snap.get("violation_desc_truncated_rows", np.nan), rows_raw
                ),
                "true_mismatch_rate": true_mismatch_rate,
            }
        )

    history_df = pd.DataFrame(history_rows)
    if not history_df.empty:
        history_df = history_df.sort_values("run_id").reset_index(drop=True)
        for col in [
            "city_outside_rate",
            "city_unknown_rate",
            "date_parse_fail_rate",
            "violation_truncated_rate",
            "true_mismatch_rate",
        ]:
            history_df[col] = pd.to_numeric(history_df[col], errors="coerce")

    return {
        "run_id": run_id,
        "run_stamp": run_stamp,
        "audit_snapshot": audit_snapshot,
        "issue_catalog": issue_df.to_dict(orient="records"),
        "samples": samples,
        "history": history_df.to_dict(orient="records"),
    }


def safer_percentile(series: pd.Series, value: float, lower_is_better: bool = True) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    if lower_is_better:
        return float((s >= value).mean())
    return float((s <= value).mean())


def build_next_inspection_dataset(events_df: pd.DataFrame) -> pd.DataFrame:
    dataset = events_df[
        [
            "business_id",
            "inspection_date_dt",
            "inspection_score",
            "red_points_total",
            "blue_points_total",
            "violation_count_total",
            "inspection_type",
            "inspection_result",
            "city_canonical",
            "grade_num",
            "is_high_risk",
        ]
    ].copy()

    dataset = dataset[dataset["inspection_date_dt"].notna()].copy()
    dataset = dataset.sort_values(["business_id", "inspection_date_dt"])
    dataset["target_next_high_risk"] = dataset.groupby("business_id")["is_high_risk"].shift(-1)
    dataset = dataset[dataset["target_next_high_risk"].notna()].copy()
    dataset["target_next_high_risk"] = dataset["target_next_high_risk"].astype(int)
    return dataset


def train_predict_models(events_df: pd.DataFrame) -> Dict[str, Any]:
    if not SKLEARN_AVAILABLE:
        return {
            "available": False,
            "message": "scikit-learn is not installed. Run: pip install -r requirements.txt",
        }

    model_df = build_next_inspection_dataset(events_df)
    if len(model_df) < 1200:
        return {
            "available": False,
            "message": f"Not enough training rows ({len(model_df)}).",
        }

    if model_df["target_next_high_risk"].nunique() < 2:
        return {
            "available": False,
            "message": "Target has only one class in current data.",
        }

    model_df = model_df.sort_values("inspection_date_dt")
    split_idx = int(len(model_df) * 0.8)
    train_df = model_df.iloc[:split_idx].copy()
    test_df = model_df.iloc[split_idx:].copy()

    if test_df.empty or train_df.empty:
        train_df, test_df = train_test_split(
            model_df,
            test_size=0.2,
            random_state=42,
            stratify=model_df["target_next_high_risk"],
        )

    if train_df["target_next_high_risk"].nunique() < 2 or test_df["target_next_high_risk"].nunique() < 2:
        train_df, test_df = train_test_split(
            model_df,
            test_size=0.2,
            random_state=42,
            stratify=model_df["target_next_high_risk"],
        )

    X_train = train_df[MODEL_ALL_FEATURES].copy()
    y_train = train_df["target_next_high_risk"].copy()
    X_test = test_df[MODEL_ALL_FEATURES].copy()
    y_test = test_df["target_next_high_risk"].copy()

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                MODEL_NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                MODEL_CATEGORICAL_FEATURES,
            ),
        ]
    )

    model_specs = {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }

    trained_models: Dict[str, Pipeline] = {}
    metric_rows = []

    for model_name, estimator in model_specs.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocess),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred.astype(float)

        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = float("nan")

        metric_rows.append(
            {
                "Model": model_name,
                "Accuracy": float(accuracy_score(y_test, y_pred)),
                "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "F1": float(f1_score(y_test, y_pred, zero_division=0)),
                "ROC_AUC": float(auc),
            }
        )
        trained_models[model_name] = pipeline

    metrics_df = pd.DataFrame(metric_rows).sort_values(by=["ROC_AUC", "F1"], ascending=False)
    best_model_name = str(metrics_df.iloc[0]["Model"])

    return {
        "available": True,
        "models": trained_models,
        "metrics": metrics_df,
        "best_model_name": best_model_name,
        "best_model": trained_models[best_model_name],
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
    }


def extract_top_feature_importance(model_pipeline: Pipeline, top_n: int = 12) -> pd.DataFrame:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return pd.DataFrame()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim == 1:
            importances = abs(coef)
        else:
            importances = abs(coef[0])
    else:
        return pd.DataFrame()

    feature_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    feature_df = feature_df.sort_values(by="importance", ascending=False).head(top_n)
    return feature_df


def build_overview_tab(
    events_df: pd.DataFrame,
    violations_df: pd.DataFrame,
    payload: Dict[str, str],
    root: Path,
) -> None:
    render_section_header(
        tr("overview.header"),
        "A public-facing guide to King County restaurant hygiene inspection data, rating rules, and public-use pathways.",
        "Public Release",
    )

    red_count = int(
        violations_df["violation_type"].astype(str).str.upper().str.strip().eq("RED").sum()
    )
    blue_count = int(
        violations_df["violation_type"].astype(str).str.upper().str.strip().eq("BLUE").sum()
    )
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Inspection rows", f"{len(events_df):,}")
    o2.metric("Restaurants", f"{events_df['business_id'].nunique():,}")
    o3.metric("Red violation records", f"{red_count:,}")
    o4.metric("Blue violation records", f"{blue_count:,}")

    render_subsection_label(
        "What this dashboard covers",
        "The public side should explain the inspection system quickly before users dive into search and homework analytics.",
    )
    intro_cols = st.columns(3)
    with intro_cols[0]:
        render_essay_card(
            "Why this project exists",
            "Many people can find a restaurant but still struggle to interpret inspection records. This dashboard translates county inspection data into clearer search, trend, and risk views.",
        )
    with intro_cols[1]:
        render_essay_card(
            "Who it serves",
            "The project was built as an MSIS data science deliverable, but it is structured for public use by consumers, restaurant owners, regulators, and a King County audit view.",
        )
    with intro_cols[2]:
        render_essay_card(
            "How to navigate it",
            "Use Restaurant Search for one venue, Historical Insights for cross-restaurant patterns, and the homework tabs for predictive analytics, model comparison, and owner-facing explanation.",
        )

    render_subsection_label(
        "Official rating levels and posters",
        "This table mirrors the county-facing rating logic and keeps the poster reference next to the rule explanation.",
    )
    rating_table = pd.DataFrame(
        [
            {
                "Risk Level": "3",
                "Rating": "Excellent",
                "Min Avg Red Points": "0.00",
                "Max Avg Red Points": "3.75",
                "Rule": "Average of last 4 routine inspections",
            },
            {
                "Risk Level": "3",
                "Rating": "Good",
                "Min Avg Red Points": "3.76",
                "Max Avg Red Points": "16.25",
                "Rule": "Average of last 4 routine inspections",
            },
            {
                "Risk Level": "3",
                "Rating": "Okay",
                "Min Avg Red Points": "16.26",
                "Max Avg Red Points": "max",
                "Rule": "Average of last 4 routine inspections",
            },
            {
                "Risk Level": "1-2",
                "Rating": "Excellent",
                "Min Avg Red Points": "0.00",
                "Max Avg Red Points": "0.00",
                "Rule": "Average of up to last 2 routine inspections",
            },
            {
                "Risk Level": "1-2",
                "Rating": "Good",
                "Min Avg Red Points": "0.01",
                "Max Avg Red Points": "5.00",
                "Rule": "Average of up to last 2 routine inspections",
            },
            {
                "Risk Level": "1-2",
                "Rating": "Okay",
                "Min Avg Red Points": "5.01",
                "Max Avg Red Points": "max",
                "Rule": "Average of up to last 2 routine inspections",
            },
            {
                "Risk Level": "Any",
                "Rating": "Needs to Improve",
                "Min Avg Red Points": "-",
                "Max Avg Red Points": "-",
                "Rule": "Closed within last 90 days or needed multiple return inspections",
            },
            {
                "Risk Level": "-",
                "Rating": RATING_NOT_AVAILABLE_LABEL,
                "Min Avg Red Points": "-",
                "Max Avg Red Points": "-",
                "Rule": "No routine-based rating is currently available",
            },
        ]
    )
    poster_catalog = load_rating_poster_catalog(str(root))
    rating_col, poster_col = st.columns([1.45, 0.95], gap="large")
    with rating_col:
        st.dataframe(rating_table, use_container_width=True, hide_index=True, height=314)
        render_takeaway_box(
            "The public dashboard keeps the county poster logic visible so users can connect a familiar grade label to the inspection severity signals behind it."
        )
    with poster_col:
        selected_overview_rating = st.selectbox(
            "Poster reference",
            GRADE_LABEL_OPTIONS_WITH_NA,
            key="overview_rating_poster_select",
        )
        poster_item = poster_catalog.get(selected_overview_rating, {})
        poster_path = clean_text(poster_item.get("path", ""))
        poster_desc = clean_text(poster_item.get("description", ""))
        if poster_path and Path(poster_path).exists():
            render_static_image(str(poster_path), caption=f"{selected_overview_rating} poster", width=190)
        else:
            st.info("Poster image not found in the local `images` directory.")
        st.markdown(f"**{selected_overview_rating}**")
        st.write(poster_desc)

    info_cols = st.columns(2, gap="large")
    with info_cols[0]:
        render_subsection_label(
            "How risk is classified in this project",
            "The dashboard recalculates a comparable risk signal instead of relying only on the published label.",
        )
        st.write(
            "This dashboard recalculates food safety ratings from risk level, the most recent routine inspections "
            "available for that restaurant (up to the official window cap), and the recent closure / return-inspection "
            "rule shown by King County. The county's published grade is retained separately for audit comparison."
        )
    with info_cols[1]:
        render_subsection_label(
            "How the public can participate",
            "The goal is to turn inspection data into usable decisions instead of a raw archive.",
        )
        st.markdown(
            "1. Check a restaurant before visiting and compare recent inspection history.\n"
            "2. Restaurant owners can use recurring violation patterns to prioritize fixes.\n"
            "3. For official details and county resources, use the links below."
        )
        st.markdown(
            "- King County search portal: https://kingcounty.gov/en/dept/dph/health-safety/food-safety/search-restaurant-safety-ratings\n"
            "- Open data dataset: https://data.kingcounty.gov/Health-Wellness/Food-Establishment-Inspection-Data/f29f-zza5\n"
            "- API endpoint: https://data.kingcounty.gov/api/v3/views/f29f-zza5/query.json"
        )

    render_subsection_label(
        "Most frequent violation codes (Top 12)",
        "This is the fastest way to see which issues recur across the county and therefore deserve more public attention.",
    )
    top_codes = (
        violations_df[
            violations_df["violation_code"].astype(str).str.strip().ne("")
            & violations_df["violation_type"].astype(str).str.strip().ne("")
        ]
        .groupby(["violation_code", "violation_type", "violation_desc_clean"], dropna=False)
        .size()
        .reset_index(name="Occurrences")
        .sort_values("Occurrences", ascending=False)
        .head(12)
    )
    if top_codes.empty:
        st.info("No violation-code data available.")
    else:
        st.dataframe(
            top_codes.rename(
                columns={
                    "violation_code": "Code",
                    "violation_type": "Type",
                    "violation_desc_clean": "Description",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    run_id = clean_text(payload.get("run_id", ""))
    generated_at = clean_text(payload.get("generated_at_utc", ""))
    st.caption(
        f"Data batch run_id: {run_id or '-'} | Generated (UTC): {generated_at or '-'}"
    )


@cache_data
def load_predict_manifest(root_str: str) -> Dict[str, Any]:
    root = Path(root_str)
    latest_pointer = root / "models" / "hw1_predict" / "latest_manifest.json"
    default_manifest = root / "models" / "hw1_predict" / "manifest.json"

    manifest_path = default_manifest
    if latest_pointer.exists():
        try:
            payload = json.loads(latest_pointer.read_text(encoding="utf-8"))
            maybe_path_str = resolve_repo_path(root, payload.get("manifest_path", ""))
            maybe_path = Path(maybe_path_str) if maybe_path_str else Path()
            if maybe_path_str and maybe_path.exists():
                manifest_path = maybe_path
        except Exception:
            manifest_path = default_manifest

    if not manifest_path.exists():
        return {
            "available": False,
            "message": (
                "Predict manifest not found. Run: "
                "python3 scripts/train_predict_models.py --root . --output-dir models/hw1_predict"
            ),
            "manifest": {},
            "manifest_path": str(manifest_path),
        }

    manifest = resolve_nested_paths(root, json.loads(manifest_path.read_text(encoding="utf-8")))
    return {
        "available": True,
        "message": "",
        "manifest": manifest,
        "manifest_path": str(manifest_path),
    }


@cache_data
def load_csv_safe(path_str: str) -> pd.DataFrame:
    path_value = resolve_repo_path(app_root(), path_str)
    path = Path(path_value)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@cache_data
def load_json_safe(path_str: str) -> Dict[str, Any]:
    path_value = resolve_repo_path(app_root(), path_str)
    path = Path(path_value)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def retry_io_operation(loader: Any, attempts: int = 3, delay_seconds: float = 0.25) -> Any:
    last_error: Exception | None = None
    for attempt_idx in range(attempts):
        try:
            return loader()
        except (OSError, TimeoutError) as exc:
            last_error = exc
            if attempt_idx == attempts - 1:
                break
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError("I/O operation failed without a captured exception.")


@cache_resource
def load_joblib_resource(path_str: str) -> Any:
    if not JOBLIB_AVAILABLE:
        raise RuntimeError("joblib is required for loading trained models.")
    resolved_path = resolve_repo_path(app_root(), path_str)
    return retry_io_operation(lambda: joblib.load(resolved_path))


@cache_resource
def load_pil_image_resource(path_str: str) -> Any:
    from PIL import Image

    resolved_path = resolve_repo_path(app_root(), path_str)

    def _load_image() -> Any:
        with Image.open(resolved_path) as img:
            return img.copy()

    return retry_io_operation(_load_image)


def render_static_image(path_value: str | Path, caption: str = "", width: str | int = "stretch") -> bool:
    path_str = clean_text(path_value)
    if not path_str:
        return False
    resolved_path = resolve_repo_path(app_root(), path_str)
    if not Path(resolved_path).exists():
        return False
    try:
        image = load_pil_image_resource(resolved_path)
        st.image(image, caption=caption, width=width)
        return True
    except Exception as exc:
        fallback_label = caption or Path(resolved_path).name
        st.caption(f"Image unavailable for {fallback_label}: {exc}")
        return False


def build_mlp_binary_classifier(nn_module: Any) -> Any:
    class MLPBinaryClassifier(nn_module.Module):
        def __init__(self, input_dim: int, hidden_layers: Tuple[int, ...], dropout: float) -> None:
            super().__init__()
            layers: List[Any] = []
            prev_dim = input_dim
            for h in hidden_layers:
                layers.append(nn_module.Linear(prev_dim, int(h)))
                layers.append(nn_module.ReLU())
                if dropout > 0:
                    layers.append(nn_module.Dropout(float(dropout)))
                prev_dim = int(h)
            layers.append(nn_module.Linear(prev_dim, 1))
            layers.append(nn_module.Sigmoid())
            self.net = nn_module.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            return self.net(x)

    return MLPBinaryClassifier


@cache_resource
def load_torch_mlp_bundle(model_path_str: str, preprocessor_path_str: str) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for MLP inference.")
    if not JOBLIB_AVAILABLE:
        raise RuntimeError("joblib is required for MLP preprocessing.")

    torch, nn = import_torch_modules()
    model_path = resolve_repo_path(app_root(), model_path_str)
    preprocessor_path = resolve_repo_path(app_root(), preprocessor_path_str)

    checkpoint = retry_io_operation(lambda: torch.load(model_path, map_location="cpu"))
    input_dim = int(checkpoint.get("input_dim", 0))
    if input_dim <= 0:
        raise RuntimeError("Invalid PyTorch checkpoint: input_dim missing or invalid.")

    hidden_layers = checkpoint.get("hidden_layers", [128, 128])
    hidden_layers = tuple(int(x) for x in hidden_layers)
    dropout = float(checkpoint.get("dropout", 0.0))
    MLPBinaryClassifier = build_mlp_binary_classifier(nn)
    model = MLPBinaryClassifier(input_dim=input_dim, hidden_layers=hidden_layers, dropout=dropout)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    preprocessor = retry_io_operation(lambda: joblib.load(preprocessor_path))
    return {
        "model": model,
        "preprocessor": preprocessor,
        "torch": torch,
    }


def probability_to_band(prob: float) -> str:
    if prob >= 0.5:
        return "High"
    if prob >= 0.25:
        return "Medium"
    return "Low"


def predict_probability_with_manifest_model(model_info: Dict[str, Any], features: pd.DataFrame) -> float:
    kind = clean_text(model_info.get("kind", ""))
    model_path = clean_text(model_info.get("model_path", ""))
    if not model_path:
        raise RuntimeError("Model path is missing in manifest.")

    if kind == "sklearn_pipeline":
        pipeline = load_joblib_resource(model_path)
        def _predict_with_pipeline() -> float:
            if hasattr(pipeline, "predict_proba"):
                return float(pipeline.predict_proba(features)[:, 1][0])
            pred = float(pipeline.predict(features)[0])
            return min(max(pred, 0.0), 1.0)

        proba = retry_io_operation(_predict_with_pipeline)
        return proba

    if kind == "pytorch_mlp":
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        pre_path = clean_text(model_info.get("preprocessor_path", ""))
        if not pre_path:
            raise RuntimeError("MLP preprocessor path is missing in manifest.")
        bundle = load_torch_mlp_bundle(model_path, pre_path)
        pre = bundle["preprocessor"]
        model = bundle["model"]
        torch = bundle["torch"]

        def _predict_with_mlp() -> float:
            arr = np.asarray(pre.transform(features), dtype=np.float32)
            tensor = torch.from_numpy(arr)
            with torch.no_grad():
                return float(model(tensor).numpy().reshape(-1)[0])

        proba = retry_io_operation(_predict_with_mlp)
        return min(max(proba, 0.0), 1.0)

    raise RuntimeError(f"Unsupported model kind: {kind}")


def build_predict_defaults_for_business(
    events_df: pd.DataFrame,
    business_id: str,
    global_defaults: Dict[str, Any],
) -> Dict[str, Any]:
    defaults = {
        **(global_defaults.get("numeric_defaults", {}) or {}),
        **(global_defaults.get("categorical_defaults", {}) or {}),
    }
    latest_row = (
        events_df[events_df["business_id"] == business_id]
        .sort_values("inspection_date_dt", ascending=False, na_position="last")
        .head(1)
    )
    if latest_row.empty:
        return defaults

    row = latest_row.iloc[0]
    for col in MODEL_NUMERIC_FEATURES:
        value = row.get(col)
        if pd.notna(value):
            try:
                if col == "grade_num":
                    defaults[col] = int(min(max(round(float(value)), 1), 4))
                elif col == "is_high_risk":
                    defaults[col] = int(round(float(value)))
                else:
                    defaults[col] = float(value)
            except Exception:
                pass
    for col in MODEL_CATEGORICAL_FEATURES:
        value = clean_text(row.get(col, ""))
        if value:
            defaults[col] = value
    return defaults


def get_best_shap_model_name(manifest: Dict[str, Any]) -> str:
    shap_meta = manifest.get("shap", {}) or {}
    return (
        clean_text(manifest.get("best_shap_tree_model_name", ""))
        or clean_text(shap_meta.get("model_name", ""))
        or clean_text(shap_meta.get("best_tree_model_name", ""))
        or clean_text(manifest.get("best_tree_model_name", ""))
    )


def maybe_render_shap_waterfall(
    manifest: Dict[str, Any],
    selected_model_name: str,
    features_df: pd.DataFrame,
) -> None:
    if not SHAP_AVAILABLE:
        st.info("SHAP is not installed in this environment.")
        return
    if not JOBLIB_AVAILABLE:
        st.info("joblib is required for SHAP runtime rendering.")
        return
    shap = import_shap_module()

    models = manifest.get("models", {})
    fallback_model_name = get_best_shap_model_name(manifest)
    model_info = models.get(selected_model_name) or models.get(fallback_model_name)
    if not model_info:
        st.info("No tree model metadata found for SHAP waterfall.")
        return
    if clean_text(model_info.get("kind", "")) != "sklearn_pipeline":
        st.info("Waterfall is available for tree-based sklearn models.")
        return

    model_path = clean_text(model_info.get("model_path", ""))
    if not model_path:
        st.info("Tree model path missing for SHAP waterfall.")
        return

    pipeline = load_joblib_resource(model_path)
    preprocessor = pipeline.named_steps.get("preprocessor")
    model = pipeline.named_steps.get("model")
    transformed = preprocessor.transform(features_df)
    transformed_dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    transformed_dense = np.asarray(transformed_dense, dtype=np.float32)

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(transformed_dense)
    if isinstance(shap_values_raw, list):
        shap_values = np.asarray(shap_values_raw[-1], dtype=np.float32)
    else:
        shap_values = np.asarray(shap_values_raw, dtype=np.float32)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_value = float(np.asarray(expected_value).reshape(-1)[-1])
    else:
        expected_value = float(expected_value)

    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=expected_value,
        data=transformed_dense[0],
        feature_names=feature_names,
    )

    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close()


def build_search_tab(summary_df: pd.DataFrame, events_df: pd.DataFrame, violations_df: pd.DataFrame) -> None:
    render_section_header(
        "Restaurant Search",
        "Filter restaurants, compare matched venues, and inspect one establishment's inspection history and remediation detail.",
        "Public Tool",
    )
    city_values = ["All"] + sorted(
        [city for city in summary_df["city_display"].dropna().unique() if str(city).strip()]
    )
    render_subsection_label(
        tr("search.step1"),
        "Choose a search mode, apply compact filters, and keep the selection stable across reruns.",
    )
    search_mode = render_search_mode_buttons()
    select_restaurant_message = "Click one restaurant row in Search Results to open detailed results."
    if search_mode in {"map", "hybrid"}:
        select_restaurant_message = (
            "Click one restaurant row in Search Results or one point on the map to open detailed results."
        )

    query = ""
    city = "All"
    zip_code = ""
    rating_labels: List[str] = []

    if search_mode == "keyword":
        f1, f2, f3, f4 = st.columns([2.2, 1.0, 0.9, 1.6], gap="small")
        query = f1.text_input("Restaurant / Address / City / Zip", key="search_filter_query")
        city = f2.selectbox("City", city_values, index=0, key="search_filter_city")
        zip_code = f3.text_input("Zip", key="search_filter_zip")
        rating_labels = f4.multiselect(
            "Latest Rating",
            options=GRADE_LABEL_OPTIONS_WITH_NA,
            key="search_filter_latest_rating",
        )
    elif search_mode == "map":
        f1, f2 = st.columns([1.1, 1.9], gap="small")
        city = f1.selectbox("City", city_values, index=0, key="search_filter_map_city")
        rating_labels = f2.multiselect(
            "Latest Rating",
            options=GRADE_LABEL_OPTIONS_WITH_NA,
            key="search_filter_map_latest_rating",
        )
    else:
        f1, f2 = st.columns([1.7, 1.4], gap="small")
        query = f1.text_input("Restaurant / Address", key="search_filter_combo_query")
        rating_labels = f2.multiselect(
            "Latest Rating",
            options=GRADE_LABEL_OPTIONS_WITH_NA,
            key="search_filter_combo_latest_rating",
        )

    controls_meta, controls_action = st.columns([4.6, 1.0], gap="small")
    with controls_meta:
        st.caption(tr("search.step1.hint"))
    with controls_action:
        if st.button(tr("search.clear_all"), key="search_clear_all_filters", use_container_width=True):
            reset_values = {
                "search_filter_query": "",
                "search_filter_city": "All",
                "search_filter_zip": "",
                "search_filter_latest_rating": [],
                "search_filter_map_city": "All",
                "search_filter_map_latest_rating": [],
                "search_filter_combo_query": "",
                "search_filter_combo_latest_rating": [],
                "search_selected_business_id": "",
            }
            for key, value in reset_values.items():
                st.session_state[key] = value
            st.session_state["search_map_nonce"] = int(st.session_state.get("search_map_nonce", 0)) + 1
            st.session_state["search_results_table_nonce"] = int(
                st.session_state.get("search_results_table_nonce", 0)
            ) + 1
            for session_key in list(st.session_state.keys()):
                if (
                    session_key.startswith("search_selected_event_id_")
                    or session_key.startswith("inspection_history_")
                    or session_key.startswith("search_history_")
                ):
                    st.session_state.pop(session_key, None)
            st.rerun()

    filtered = filter_businesses(summary_df, query, city, zip_code, rating_labels)
    filter_signature = "|".join(
        [
            search_mode,
            clean_text(query),
            clean_text(city),
            clean_text(zip_code),
            ",".join(sorted([clean_text(item) for item in rating_labels])),
            str(len(filtered)),
        ]
    )
    filter_signature_hash = str(abs(hash(filter_signature)))

    sort_options = [
        ("Restaurant", "display_name"),
        ("Address", "full_address_clean"),
        ("City", "city_display"),
        ("Zip", "zip_code"),
    ]
    label_to_field = {label: field for label, field in sort_options}
    field_to_label = {field: label for label, field in sort_options}
    current_sort_field = clean_text(st.session_state.get("search_sort_field", "display_name"))
    if current_sort_field not in field_to_label:
        current_sort_field = "display_name"
        st.session_state["search_sort_field"] = current_sort_field
    current_sort_label = field_to_label.get(current_sort_field, "Restaurant")
    current_sort_order = "Ascending" if bool(st.session_state.get("search_sort_asc", True)) else "Descending"

    render_step_separator()
    render_subsection_label(
        tr("search.step2"),
        "Results and map selection stay in the same stage so the user can scan, sort, and pick one venue quickly.",
    )
    top_meta_left, top_meta_mid, top_meta_right = st.columns([1.0, 1.0, 2.4], gap="small")
    selected_sort_label = top_meta_left.selectbox(
        "Sort field",
        [label for label, _ in sort_options],
        index=[label for label, _ in sort_options].index(current_sort_label),
        key="search_sort_field_select",
    )
    selected_sort_order = top_meta_mid.selectbox(
        "Order",
        ["Ascending", "Descending"],
        index=0 if current_sort_order == "Ascending" else 1,
        key="search_sort_order_select",
    )
    with top_meta_right:
        st.caption(
            f"Matched restaurants: {len(filtered):,}. "
            "Select a row to pin it to the top and unlock the detail workflow below."
        )
    st.session_state["search_sort_field"] = label_to_field[selected_sort_label]
    st.session_state["search_sort_asc"] = selected_sort_order == "Ascending"
    sort_field = st.session_state["search_sort_field"]
    sort_asc = bool(st.session_state["search_sort_asc"])
    if sort_field in filtered.columns:
        filtered = filtered.sort_values(by=[sort_field], ascending=[sort_asc], na_position="last")

    pinned_selected_business_id = clean_text(st.session_state.get("search_selected_business_id", ""))
    if pinned_selected_business_id:
        pinned_mask = filtered["business_id"].astype(str).str.strip() == pinned_selected_business_id
        if pinned_mask.any():
            filtered = pd.concat(
                [filtered[pinned_mask], filtered[~pinned_mask]],
                ignore_index=True,
            )

    if filtered.empty:
        st.info("No matches found. Try a different keyword or relax filters.")
        return

    if search_mode in {"map", "hybrid"}:
        map_df = filtered.copy()
        map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
        map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
        map_df = map_df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
        if not PLOTLY_AVAILABLE:
            st.info("Plotly is not installed in this environment, so map search is unavailable.")
        elif map_df.empty:
            st.info("No matched restaurants with valid coordinates are available for map search.")
        else:
            st.caption(tr("search.map.hint"))
            figure = build_search_map_figure(
                map_df,
                selected_business_id=clean_text(st.session_state.get("search_selected_business_id", "")),
            )
            map_nonce = int(st.session_state.get("search_map_nonce", 0))
            map_event = st.plotly_chart(
                figure,
                key=f"search_map_plot_{search_mode}_{filter_signature_hash}_{map_nonce}",
                width="stretch",
                on_select="rerun",
                selection_mode=("points",),
                config={"scrollZoom": True, "displaylogo": False},
                theme=None,
            )
            map_selected_business_id = ""
            selection = getattr(map_event, "selection", None)
            if selection and getattr(selection, "points", None):
                selected_point = selection.points[0]
                custom_data = selected_point.get("customdata", []) if isinstance(selected_point, dict) else []
                if custom_data:
                    map_selected_business_id = clean_text(custom_data[0])
                elif getattr(selection, "point_indices", None):
                    point_index = int(selection.point_indices[0])
                    if 0 <= point_index < len(map_df):
                        map_selected_business_id = clean_text(map_df.iloc[point_index]["business_id"])
            if map_selected_business_id and map_selected_business_id != clean_text(
                st.session_state.get("search_selected_business_id", "")
            ):
                st.session_state["search_selected_business_id"] = map_selected_business_id
                st.session_state["search_results_table_nonce"] = int(
                    st.session_state.get("search_results_table_nonce", 0)
                ) + 1

    filtered_reset = filtered.reset_index(drop=True).copy()
    pinned_feedback = ""
    if pinned_selected_business_id and not filtered_reset.empty:
        first_business_id = clean_text(filtered_reset.iloc[0]["business_id"])
        if first_business_id == pinned_selected_business_id:
            pinned_feedback = (
                f"Selected restaurant: {clean_text(filtered_reset.iloc[0]['display_name'])} | "
                "Pinned to the top of the list."
            )
    if pinned_feedback:
        st.caption(pinned_feedback)

    search_view = filtered_reset[
        ["display_name", "full_address_clean", "city_display", "zip_code"]
    ].rename(
        columns={
            "display_name": "Restaurant",
            "full_address_clean": "Address",
            "city_display": "City",
            "zip_code": "Zip",
        }
    )
    if pinned_feedback:
        def highlight_selected_row(row: pd.Series) -> List[str]:
            if int(row.name) == 0:
                styles = ["background-color: #eefafc; font-weight: 700;" for _ in row]
                if styles:
                    styles[0] += " border-left: 4px solid #0b7285;"
                return styles
            return ["" for _ in row]

        search_view_display: Any = search_view.style.apply(highlight_selected_row, axis=1)
    else:
        search_view_display = search_view

    results_table_nonce = int(st.session_state.get("search_results_table_nonce", 0))
    result_selection = st.dataframe(
        search_view_display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"search_results_table_select_{filter_signature_hash}_{results_table_nonce}",
        height=315,
    )

    selected_idx: Optional[int] = None
    if hasattr(result_selection, "selection") and result_selection.selection and result_selection.selection.rows:
        selected_idx = int(result_selection.selection.rows[0])
    else:
        previous_id = clean_text(st.session_state.get("search_selected_business_id", ""))
        if previous_id:
            candidates = filtered_reset.index[
                filtered_reset["business_id"].astype(str).str.strip() == previous_id
            ].tolist()
            if candidates:
                selected_idx = int(candidates[0])

    if selected_idx is None or selected_idx >= len(filtered_reset):
        st.session_state["search_selected_business_id"] = ""
        st.info(select_restaurant_message)
        return

    selected_business_id = clean_text(filtered_reset.iloc[selected_idx]["business_id"])
    if not selected_business_id:
        st.session_state["search_selected_business_id"] = ""
        st.info(select_restaurant_message)
        return
    st.session_state["search_selected_business_id"] = selected_business_id
    selected_summary_row = filtered_reset.iloc[selected_idx]

    biz_events = events_df[events_df["business_id"] == selected_business_id].copy()
    biz_events = biz_events.sort_values(by="inspection_date_dt", ascending=False, na_position="last")
    if biz_events.empty:
        st.info("No inspection history is available for the selected restaurant.")
        return

    render_step_separator()
    render_subsection_label(
        "Selected restaurant snapshot",
        "The top card keeps the current poster, latest inspection signals, and score trend in one compact block.",
    )

    biz_violations = violations_df[violations_df["business_id"] == selected_business_id].copy()
    has_violation = (
        biz_violations["violation_code"].str.strip().ne("")
        | biz_violations["violation_desc_clean"].str.strip().ne("")
    )
    biz_violations = biz_violations[has_violation].copy()
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    biz_violations["priority_rank"] = biz_violations["action_priority"].map(priority_rank).fillna(3)
    biz_violations = biz_violations.sort_values(
        by=["inspection_date_dt", "priority_rank"],
        ascending=[False, True],
        na_position="last",
    )

    latest_row = biz_events.iloc[0]
    latest_rating = clean_text(selected_summary_row.get("latest_rating", "")) or RATING_NOT_AVAILABLE_LABEL
    latest_score = latest_row.get("inspection_score")
    latest_risk_level = clean_text(selected_summary_row.get("latest_risk_level", ""))
    latest_avg_red = pd.to_numeric(
        pd.Series([selected_summary_row.get("latest_rating_avg_red_points")]), errors="coerce"
    ).iloc[0]
    poster_catalog = load_rating_poster_catalog(str(app_root()))
    selected_poster = poster_catalog.get(latest_rating, {})
    selected_poster_path = clean_text(selected_poster.get("path", ""))
    rating_note = OFFICIAL_RATING_EXPLANATIONS.get(
        latest_rating, clean_text(selected_poster.get("description", ""))
    )
    risk_meta = RISK_CATEGORY_CARD_EXPLANATIONS.get(latest_risk_level, {})
    latest_risk_card_value = format_risk_category_card_value(latest_risk_level)
    avg_red_title = format_detail_card_avg_red_title(latest_risk_level)
    latest_score_text = "-" if pd.isna(latest_score) else f"{float(latest_score):.0f}"
    latest_avg_red_text = "-" if pd.isna(latest_avg_red) else f"{float(latest_avg_red):.2f}"

    snapshot_left, snapshot_right = st.columns([0.7, 1.8], gap="large")
    with snapshot_left:
        if selected_poster_path and Path(selected_poster_path).exists():
            render_static_image(str(selected_poster_path), caption=f"{latest_rating} poster", width=176)
        else:
            st.info("Poster image not found in the local `images` directory.")
        st.caption(
            f"Selected venue: {clean_text(selected_summary_row.get('display_name', '')) or '-'} | "
            f"Sorted by {field_to_label.get(sort_field, sort_field)} ({'ASC' if sort_asc else 'DESC'})"
        )
    with snapshot_right:
        row0_col1, row0_col2, row0_col3 = st.columns(3, gap="small")
        with row0_col1:
            render_detail_info_card(
                "Restaurant Name",
                clean_text(selected_summary_row.get("display_name", "")) or "-",
            )
        with row0_col2:
            render_detail_info_card(
                "Latest Inspection Date",
                clean_text(latest_row.get("inspection_date", "")) or "-",
            )
        with row0_col3:
            render_detail_info_card("Latest Inspection Score", latest_score_text)

        row1_col1, row1_col2, row1_col3 = st.columns(3, gap="small")
        with row1_col1:
            render_detail_info_card(
                "Restaurant Address",
                clean_text(selected_summary_row.get("full_address_clean", "")) or "-",
            )
        with row1_col2:
            render_detail_info_card("Inspection Count", f"{len(biz_events):,}")
        with row1_col3:
            render_detail_info_card(avg_red_title, latest_avg_red_text)

        row2_col1, row2_col2 = st.columns(2, gap="small")
        with row2_col1:
            render_detail_info_card(
                "Risk Category",
                latest_risk_card_value,
                bullets=risk_meta.get("bullets", []),
            )
        with row2_col2:
            render_detail_info_card("Food Safety Rating", latest_rating, note=rating_note)

    score_plot_df = biz_events[["inspection_date_dt", "inspection_score"]].dropna(
        subset=["inspection_date_dt", "inspection_score"]
    )
    score_plot_df = score_plot_df.sort_values("inspection_date_dt")
    if not score_plot_df.empty:
        st.markdown("**Inspection score trend**")
        st.line_chart(
            score_plot_df.set_index("inspection_date_dt")["inspection_score"],
            use_container_width=True,
        )
    else:
        st.info("No score history available for plotting.")

    render_step_separator()
    render_subsection_label(
        tr("search.step3"),
        "Filter the selected restaurant's inspection history, then click one row to open that day's violations and remediation guidance.",
    )
    h1, h2, h3, h4 = st.columns([1.05, 1.05, 1.45, 0.85], gap="small")
    history_valid_dates = biz_events["inspection_date_dt"].dropna()
    min_hist_date = history_valid_dates.min().date() if not history_valid_dates.empty else None
    max_hist_date = history_valid_dates.max().date() if not history_valid_dates.empty else None
    history_start = h1.date_input(
        "Start date",
        value=min_hist_date,
        min_value=min_hist_date,
        max_value=max_hist_date,
        key=f"search_history_start_{selected_business_id}",
    ) if min_hist_date and max_hist_date else None
    history_end = h2.date_input(
        "End date",
        value=max_hist_date,
        min_value=min_hist_date,
        max_value=max_hist_date,
        key=f"search_history_end_{selected_business_id}",
    ) if min_hist_date and max_hist_date else None
    history_type_options = sorted(
        [v for v in biz_events["inspection_type"].astype(str).str.strip().unique().tolist() if v]
    )
    selected_history_types = h3.multiselect(
        "Inspection type",
        options=history_type_options,
        default=[],
        key=f"search_history_type_{selected_business_id}",
    )
    h4.write("")
    if h4.button(
        "Clear filters",
        key=f"search_history_clear_{selected_business_id}",
        use_container_width=True,
    ):
        if min_hist_date and max_hist_date:
            st.session_state[f"search_history_start_{selected_business_id}"] = min_hist_date
            st.session_state[f"search_history_end_{selected_business_id}"] = max_hist_date
        st.session_state[f"search_history_type_{selected_business_id}"] = []
        st.rerun()
    filtered_biz_events = biz_events.copy()
    if history_start is not None:
        filtered_biz_events = filtered_biz_events[
            filtered_biz_events["inspection_date_dt"] >= pd.Timestamp(history_start)
        ]
    if history_end is not None:
        filtered_biz_events = filtered_biz_events[
            filtered_biz_events["inspection_date_dt"] <= pd.Timestamp(history_end)
        ]
    if selected_history_types:
        filtered_biz_events = filtered_biz_events[
            filtered_biz_events["inspection_type"].isin(selected_history_types)
        ]
    if filtered_biz_events.empty:
        st.info("No inspections match the current Step 3 filters.")
        return

    history_view = filtered_biz_events[
        [
            "inspection_event_id",
            "inspection_date",
            "inspection_type",
            "inspection_result",
            "inspection_score",
            "effective_rating_label",
            "violation_count_total",
            "red_points_total",
            "blue_points_total",
        ]
    ].rename(
        columns={
            "inspection_event_id": "Inspection ID",
            "inspection_date": "Inspection date",
            "inspection_type": "Inspection type",
            "inspection_result": "Result",
            "inspection_score": "Score",
            "effective_rating_label": "Rating",
            "violation_count_total": "Violation count",
            "red_points_total": "Red Points",
            "blue_points_total": "Blue Points",
        }
    )
    inspection_selection = st.dataframe(
        history_view,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"inspection_history_{selected_business_id}",
        height=286,
    )

    selected_row_idx: Optional[int] = None
    if hasattr(inspection_selection, "selection") and inspection_selection.selection and inspection_selection.selection.rows:
        selected_row_idx = int(inspection_selection.selection.rows[0])
    else:
        selected_event_state_key = f"search_selected_event_id_{selected_business_id}"
        previous_event_id = clean_text(st.session_state.get(selected_event_state_key, ""))
        if previous_event_id:
            matched_rows = history_view.index[
                history_view["Inspection ID"].astype(str).str.strip() == previous_event_id
            ].tolist()
            if matched_rows:
                selected_row_idx = int(matched_rows[0])

    if selected_row_idx is None or selected_row_idx >= len(history_view):
        st.info("Click one inspection row to view violations and remediation for that day.")
        return

    selected_event_id = clean_text(history_view.iloc[selected_row_idx]["Inspection ID"])
    selected_event_date = clean_text(history_view.iloc[selected_row_idx]["Inspection date"])
    st.session_state[f"search_selected_event_id_{selected_business_id}"] = selected_event_id

    render_step_separator()
    render_subsection_label(
        tr("search.step4"),
        f"Selected inspection: {selected_event_date} | ID: {selected_event_id}",
    )
    selected_violations = biz_violations[
        biz_violations["inspection_event_id"].astype(str).str.strip() == selected_event_id
    ].copy()
    if selected_violations.empty:
        selected_violations = biz_violations[
            biz_violations["inspection_date"].astype(str).str.strip() == selected_event_date
        ].copy()

    if selected_violations.empty:
        st.info("No violation details found for the selected inspection.")
    else:
        st.dataframe(
            selected_violations[
                [
                    "violation_type",
                    "violation_code",
                    "violation_points",
                    "violation_desc_clean",
                    "action_priority",
                    "action_category",
                    "action_summary_en",
                ]
            ].rename(
                columns={
                    "violation_type": "Type",
                    "violation_code": "Code",
                    "violation_points": "Points",
                    "violation_desc_clean": "Violation",
                    "action_priority": "Priority",
                    "action_category": "Category",
                    "action_summary_en": "Remediation summary",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=290,
        )


def build_summary_tab(
    events_df: pd.DataFrame,
    violations_df: pd.DataFrame,
    payload: Dict[str, str],
    root: Path,
) -> None:
    render_section_header(
        tr("summary.header"),
        "Multi-role historical views for consumers, owners, regulators, and county-quality auditing.",
        "Public Release",
    )
    st.caption(tr("summary.role_hint"))
    st.caption(tr("summary.risk_hint"))

    filtered_events = events_df[events_df["inspection_date_dt"].notna()].copy()
    filtered_violations = violations_df.copy()
    if filtered_events.empty:
        st.info("No inspections found in current data.")
        return

    profiles_df = business_profile_stats(filtered_events)
    if profiles_df.empty:
        st.info("No business profile data is available.")
        return

    role_options = list(INSIGHT_QUESTIONS.keys())
    st.markdown(f"**{tr('summary.step1')}**")
    selected_role = st.radio(
        "Role",
        role_options,
        horizontal=True,
        format_func=lambda x: tr(f"role.{x}"),
        label_visibility="collapsed",
        key="insight_role",
    )
    role_questions = INSIGHT_QUESTIONS[selected_role]
    question_label_to_id: Dict[str, str] = {}
    question_options: List[str] = []
    for qid, fallback_label in role_questions:
        translated_label = tr(f"question.{qid}")
        if translated_label == f"question.{qid}":
            translated_label = fallback_label
        question_label_to_id[translated_label] = qid
        question_options.append(translated_label)
    render_step_separator()
    st.markdown(f"**{tr('summary.step2')}**")
    selected_question_label = st.selectbox(
        tr("summary.question_label"),
        question_options,
        key=f"insight_question_{selected_role.lower().replace(' ', '_')}",
    )
    selected_question_id = question_label_to_id[selected_question_label]

    option_map = {
        (
            f"{row.display_name} | {row.full_address_clean} | "
            f"{row.city_display} ({row.business_id})"
        ): row.business_id
        for row in profiles_df.sort_values(
            by=["display_name", "city_display"], ascending=[True, True]
        ).itertuples(index=False)
    }
    city_options = sorted(
        [city for city in profiles_df["city_display"].dropna().unique().tolist() if str(city).strip()]
    )

    if selected_role == "Consumer":
        if selected_question_id == "C1":
            month_periods = sorted(
                filtered_events["inspection_date_dt"].dropna().dt.to_period("M").unique().tolist()
            )
            if not month_periods:
                st.info("No monthly period data found.")
                return
            month_labels = [period.strftime("%Y-%m") for period in month_periods]
            selected_month_label = st.selectbox(
                "Select month",
                month_labels,
                index=len(month_labels) - 1,
                key="c1_month_select",
            )
            selected_period = pd.Period(selected_month_label, freq="M")
            movement_df = compute_monthly_rating_changes(filtered_events, selected_period, source="official_grade")
            movement_source = "Dashboard calculated rating"
            if movement_df.empty:
                movement_df = compute_monthly_rating_changes(filtered_events, selected_period, source="score_band")
                movement_source = "Score-band proxy rating"

            st.caption(f"Movement source: {movement_source}")
            if movement_df.empty:
                st.info("No rating changes detected for this month.")
                return

            improved_df = movement_df[movement_df["change_type"] == "Improved"].copy()
            declined_df = movement_df[movement_df["change_type"] == "Declined"].copy()
            c_up, c_down = st.columns(2)
            c_up.metric("Improved restaurants", f"{len(improved_df):,}")
            c_down.metric("Declined restaurants", f"{len(declined_df):,}")

            st.markdown("**Improved restaurants**")
            if improved_df.empty:
                st.write("No improvements for this month.")
            else:
                st.dataframe(
                    improved_df[
                        [
                            "display_name",
                            "city_display",
                            "full_address_clean",
                            "prev_date",
                            "inspection_date",
                            "transition",
                        ]
                    ]
                    .rename(
                        columns={
                            "display_name": "Restaurant",
                            "city_display": "City",
                            "full_address_clean": "Address",
                            "prev_date": "Previous date",
                            "inspection_date": "Current date",
                            "transition": "Rating change",
                        }
                    )
                    .sort_values(by=["Current date"], ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    height=320,
                )

            st.markdown("**Declined restaurants**")
            if declined_df.empty:
                st.write("No declines for this month.")
            else:
                st.dataframe(
                    declined_df[
                        [
                            "display_name",
                            "city_display",
                            "full_address_clean",
                            "prev_date",
                            "inspection_date",
                            "transition",
                        ]
                    ]
                    .rename(
                        columns={
                            "display_name": "Restaurant",
                            "city_display": "City",
                            "full_address_clean": "Address",
                            "prev_date": "Previous date",
                            "inspection_date": "Current date",
                            "transition": "Rating change",
                        }
                    )
                    .sort_values(by=["Current date"], ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    height=320,
                )

        elif selected_question_id == "C2":
            f1, f2 = st.columns([2, 1])
            selected_city = f1.selectbox(
                "Area (city)",
                ["All"] + city_options,
                index=0,
                key="c2_city_select",
            )
            min_inspections = int(f2.number_input("Min inspections", min_value=1, max_value=50, value=6, step=1))

            safer_df = profiles_df[profiles_df["inspection_count"] >= min_inspections].copy()
            if selected_city != "All":
                safer_df = safer_df[safer_df["city_display"] == selected_city]
            safer_df = safer_df.sort_values(
                by=["safety_score", "inspection_count"],
                ascending=[False, False],
                na_position="last",
            )
            if safer_df.empty:
                st.info("No restaurants match the current area and minimum-inspection filter.")
                return

            st.dataframe(
                safer_df[
                    [
                        "display_name",
                        "city_display",
                        "full_address_clean",
                        "inspection_count",
                        "high_risk_rate",
                        "avg_red_points",
                        "latest_rating",
                        "safety_score",
                    ]
                ]
                .head(25)
                .rename(
                    columns={
                        "display_name": "Restaurant",
                        "city_display": "City",
                        "full_address_clean": "Address",
                        "inspection_count": "Inspections",
                        "high_risk_rate": "High-risk rate",
                        "avg_red_points": "Avg red points",
                        "latest_rating": "Latest rating",
                        "safety_score": "Safety score",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=380,
            )
            bar_df = safer_df.head(15).set_index("display_name")[["safety_score"]]
            st.bar_chart(bar_df, use_container_width=True)

        elif selected_question_id == "C3":
            days = st.selectbox("Recent window (days)", [30, 60, 90, 180], index=2, key="c3_days")
            max_date = filtered_events["inspection_date_dt"].max()
            if pd.isna(max_date):
                st.info("Date information is not available.")
                return
            cutoff = max_date - pd.Timedelta(days=int(days))
            recent_red = filtered_violations[
                (filtered_violations["inspection_date_dt"].notna())
                & (filtered_violations["inspection_date_dt"] >= cutoff)
                & (filtered_violations["violation_type"].astype(str).str.upper().str.strip() == "RED")
            ].copy()
            if recent_red.empty:
                st.info("No red-violation records in the selected window.")
                return

            recent_red["display_name"] = recent_red["business_name_official"].where(
                recent_red["business_name_official"].astype(str).str.strip() != "",
                recent_red["business_name_alt"],
            )
            recent_red["display_name"] = recent_red["display_name"].where(
                recent_red["display_name"].astype(str).str.strip() != "",
                recent_red["business_id"],
            )
            red_summary = (
                recent_red.groupby(
                    ["business_id", "display_name", "city_display", "full_address_clean"],
                    dropna=False,
                )
                .agg(
                    red_records=("row_id", "count"),
                    total_red_points=("violation_points_num", "sum"),
                    latest_red_date=("inspection_date_dt", "max"),
                )
                .reset_index()
            )
            red_summary["latest_red_date"] = red_summary["latest_red_date"].dt.strftime("%Y-%m-%d")
            red_summary = red_summary.sort_values(
                by=["total_red_points", "red_records"],
                ascending=[False, False],
            )
            st.dataframe(
                red_summary[
                    [
                        "display_name",
                        "city_display",
                        "full_address_clean",
                        "red_records",
                        "total_red_points",
                        "latest_red_date",
                    ]
                ]
                .head(25)
                .rename(
                    columns={
                        "display_name": "Restaurant",
                        "city_display": "City",
                        "full_address_clean": "Address",
                        "red_records": "Red records",
                        "total_red_points": "Total red points",
                        "latest_red_date": "Most recent red date",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=380,
            )
            st.bar_chart(
                red_summary.head(15).set_index("display_name")[["red_records"]],
                use_container_width=True,
            )

        elif selected_question_id == "C4":
            if not option_map:
                st.info("No restaurants are available for peer comparison.")
                return
            selected_restaurant_label = st.selectbox(
                "Select restaurant",
                list(option_map.keys()),
                index=0,
                key="c4_restaurant",
            )
            selected_business_id = option_map[selected_restaurant_label]
            selected_row_df = profiles_df[
                profiles_df["business_id"].astype(str).str.strip() == selected_business_id
            ]
            if selected_row_df.empty:
                st.info("Selected restaurant profile was not found.")
                return
            selected_row = selected_row_df.iloc[0]
            peer_df = profiles_df[
                (profiles_df["city_display"] == selected_row["city_display"])
                & (profiles_df["inspection_count"] >= 3)
            ].copy()
            if peer_df.empty:
                st.info("No peer group is available in this city.")
                return

            score_pct = safer_percentile(peer_df["avg_inspection_score"], selected_row["avg_inspection_score"], True)
            red_pct = safer_percentile(peer_df["avg_red_points"], selected_row["avg_red_points"], True)
            risk_pct = safer_percentile(peer_df["high_risk_rate"], selected_row["high_risk_rate"], True)
            p1, p2, p3 = st.columns(3)
            p1.metric("Safer than peers (score)", "-" if pd.isna(score_pct) else f"{score_pct:.1%}")
            p2.metric("Safer than peers (red points)", "-" if pd.isna(red_pct) else f"{red_pct:.1%}")
            p3.metric("Safer than peers (high-risk rate)", "-" if pd.isna(risk_pct) else f"{risk_pct:.1%}")

            compare_df = pd.DataFrame(
                [
                    ["Average inspection score", selected_row["avg_inspection_score"], peer_df["avg_inspection_score"].median()],
                    ["Average red points", selected_row["avg_red_points"], peer_df["avg_red_points"].median()],
                    ["High-risk rate", selected_row["high_risk_rate"], peer_df["high_risk_rate"].median()],
                    ["Inspection count", selected_row["inspection_count"], peer_df["inspection_count"].median()],
                ],
                columns=["Metric", "Selected restaurant", "City median"],
            )
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

            st.markdown("**Safer peers in the same city (Top 15 by safety score)**")
            peer_top = peer_df.sort_values(by=["safety_score", "inspection_count"], ascending=[False, False]).head(15)
            st.dataframe(
                peer_top[
                    [
                        "display_name",
                        "full_address_clean",
                        "inspection_count",
                        "high_risk_rate",
                        "avg_red_points",
                        "safety_score",
                    ]
                ].rename(
                    columns={
                        "display_name": "Restaurant",
                        "full_address_clean": "Address",
                        "inspection_count": "Inspections",
                        "high_risk_rate": "High-risk rate",
                        "avg_red_points": "Avg red points",
                        "safety_score": "Safety score",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=320,
            )

        elif selected_question_id == "C5":
            days = st.selectbox("Recent window (days)", [30, 60, 90, 180], index=2, key="c5_days")
            max_date = filtered_events["inspection_date_dt"].max()
            if pd.isna(max_date):
                st.info("Date information is not available.")
                return
            cutoff = max_date - pd.Timedelta(days=int(days))
            recent_codes = filtered_violations[
                (filtered_violations["inspection_date_dt"].notna())
                & (filtered_violations["inspection_date_dt"] >= cutoff)
                & (filtered_violations["violation_code"].astype(str).str.strip() != "")
            ].copy()
            if recent_codes.empty:
                st.info("No coded violations in the selected window.")
                return

            top_issue_df = (
                recent_codes.groupby(
                    ["violation_code", "violation_type", "violation_desc_clean", "action_category"],
                    dropna=False,
                )
                .agg(
                    occurrences=("row_id", "count"),
                    affected_restaurants=("business_id", "nunique"),
                    avg_points=("violation_points_num", "mean"),
                )
                .reset_index()
                .sort_values(by=["occurrences", "affected_restaurants"], ascending=[False, False])
            )
            top_issue_df["avg_points"] = top_issue_df["avg_points"].round(2)
            st.dataframe(
                top_issue_df.head(25).rename(
                    columns={
                        "violation_code": "Code",
                        "violation_type": "Type",
                        "violation_desc_clean": "Violation",
                        "action_category": "Category",
                        "occurrences": "Occurrences",
                        "affected_restaurants": "Restaurants affected",
                        "avg_points": "Avg points",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=380,
            )
            top_bar = (
                top_issue_df.head(12)
                .assign(code_type=lambda d: d["violation_code"] + "-" + d["violation_type"])
                .set_index("code_type")[["occurrences"]]
            )
            st.bar_chart(top_bar, use_container_width=True)

        elif selected_question_id == "C6":
            selected_city = st.selectbox(
                "Area (city)",
                ["All"] + city_options,
                index=0,
                key="c6_city",
            )
            trend_df = filtered_events.copy()
            if selected_city != "All":
                trend_df = trend_df[trend_df["city_display"] == selected_city]
            if trend_df.empty:
                st.info("No records in the selected area.")
                return

            trend_df["month"] = trend_df["inspection_date_dt"].dt.to_period("M").dt.to_timestamp()
            monthly = (
                trend_df.groupby("month", dropna=False)
                .agg(
                    inspections=("business_id", "size"),
                    high_risk_rate=("is_high_risk", "mean"),
                    avg_score=("inspection_score", "mean"),
                )
                .reset_index()
                .sort_values("month")
            )
            monthly["high_risk_rate_rolling3"] = monthly["high_risk_rate"].rolling(3, min_periods=1).mean()
            monthly["avg_score_rolling3"] = monthly["avg_score"].rolling(3, min_periods=1).mean()

            st.markdown("**Monthly high-risk rate**")
            st.line_chart(
                monthly.set_index("month")[["high_risk_rate", "high_risk_rate_rolling3"]],
                use_container_width=True,
            )
            st.markdown("**Monthly average score**")
            st.line_chart(
                monthly.set_index("month")[["avg_score", "avg_score_rolling3"]],
                use_container_width=True,
            )
            st.dataframe(
                monthly.tail(24).rename(
                    columns={
                        "month": "Month",
                        "inspections": "Inspections",
                        "high_risk_rate": "High-risk rate",
                        "avg_score": "Avg score",
                        "high_risk_rate_rolling3": "High-risk rate (3M avg)",
                        "avg_score_rolling3": "Avg score (3M avg)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=320,
            )

    elif selected_role == "Restaurant Owner":
        if selected_question_id == "O1":
            code_rows = filtered_violations[
                filtered_violations["violation_code"].astype(str).str.strip().ne("")
            ].copy()
            if code_rows.empty:
                st.info("No coded violation records found.")
                return
            priority_weight = {"high": 3.0, "medium": 2.0, "low": 1.0}
            ranking = (
                code_rows.groupby(
                    [
                        "violation_code",
                        "violation_type",
                        "action_category",
                        "action_priority",
                        "action_summary_en",
                    ],
                    dropna=False,
                )
                .agg(
                    occurrences=("row_id", "count"),
                    affected_restaurants=("business_id", "nunique"),
                    avg_points=("violation_points_num", "mean"),
                )
                .reset_index()
            )
            ranking["avg_points"] = ranking["avg_points"].fillna(0.0)
            ranking["priority_weight"] = ranking["action_priority"].map(priority_weight).fillna(1.0)
            ranking["priority_score"] = (
                ranking["occurrences"] * ranking["avg_points"] + ranking["occurrences"] * ranking["priority_weight"]
            )
            ranking = ranking.sort_values(by=["priority_score", "occurrences"], ascending=[False, False])
            st.dataframe(
                ranking.head(25).rename(
                    columns={
                        "violation_code": "Code",
                        "violation_type": "Type",
                        "action_category": "Category",
                        "action_priority": "Priority",
                        "action_summary_en": "Remediation summary",
                        "occurrences": "Occurrences",
                        "affected_restaurants": "Restaurants affected",
                        "avg_points": "Avg points",
                        "priority_score": "Priority score",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=380,
            )
            st.bar_chart(
                ranking.head(12)
                .assign(code_type=lambda d: d["violation_code"] + "-" + d["violation_type"])
                .set_index("code_type")[["priority_score"]],
                use_container_width=True,
            )

        elif selected_question_id == "O2":
            if not option_map:
                st.info("No restaurants available.")
                return
            selected_restaurant_label = st.selectbox(
                "Select restaurant",
                list(option_map.keys()),
                index=0,
                key="o2_restaurant",
            )
            selected_business_id = option_map[selected_restaurant_label]
            biz_viol = filtered_violations[
                (filtered_violations["business_id"].astype(str).str.strip() == selected_business_id)
                & (filtered_violations["violation_code"].astype(str).str.strip() != "")
                & (filtered_violations["inspection_date_dt"].notna())
            ].copy()
            if biz_viol.empty:
                st.info("No coded violations for this restaurant.")
                return
            biz_viol["month"] = biz_viol["inspection_date_dt"].dt.to_period("M").astype(str)
            top_codes = biz_viol.groupby("violation_code", dropna=False).size().sort_values(ascending=False).head(12).index.tolist()
            pivot = (
                biz_viol[biz_viol["violation_code"].isin(top_codes)]
                .groupby(["violation_code", "month"], dropna=False)
                .size()
                .unstack(fill_value=0)
            )
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)
            if pivot.shape[1] > 12:
                pivot = pivot.iloc[:, -12:]
            st.markdown("**Monthly repeat heat table (last 12 months max)**")
            st.dataframe(pivot, use_container_width=True, height=360)

            repeated = (
                biz_viol.groupby("violation_code", dropna=False)
                .agg(months_with_issue=("month", "nunique"), total_records=("row_id", "count"))
                .reset_index()
            )
            repeated = repeated[repeated["months_with_issue"] >= 3].sort_values(
                by=["months_with_issue", "total_records"], ascending=[False, False]
            )
            st.markdown("**Repeated issues (appeared in 3+ months)**")
            if repeated.empty:
                st.write("No repeated issues crossing 3 months.")
            else:
                st.dataframe(
                    repeated.rename(
                        columns={
                            "violation_code": "Code",
                            "months_with_issue": "Months with issue",
                            "total_records": "Total records",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=300,
                )

        elif selected_question_id == "O3":
            if not option_map:
                st.info("No restaurants are available for peer comparison.")
                return
            selected_restaurant_label = st.selectbox(
                "Select restaurant",
                list(option_map.keys()),
                index=0,
                key="o3_restaurant",
            )
            selected_business_id = option_map[selected_restaurant_label]
            selected_row_df = profiles_df[
                profiles_df["business_id"].astype(str).str.strip() == selected_business_id
            ]
            if selected_row_df.empty:
                st.info("Selected restaurant profile was not found.")
                return
            selected_row = selected_row_df.iloc[0]
            peer_df = profiles_df[
                (profiles_df["city_display"] == selected_row["city_display"])
                & (profiles_df["inspection_count"] >= 3)
            ].copy()
            if peer_df.empty:
                st.info("No peer group is available in this city.")
                return

            score_pct = safer_percentile(peer_df["avg_inspection_score"], selected_row["avg_inspection_score"], True)
            red_pct = safer_percentile(peer_df["avg_red_points"], selected_row["avg_red_points"], True)
            risk_pct = safer_percentile(peer_df["high_risk_rate"], selected_row["high_risk_rate"], True)
            p1, p2, p3 = st.columns(3)
            p1.metric("Score safety percentile", "-" if pd.isna(score_pct) else f"{score_pct:.1%}")
            p2.metric("Red-point safety percentile", "-" if pd.isna(red_pct) else f"{red_pct:.1%}")
            p3.metric("Risk-rate safety percentile", "-" if pd.isna(risk_pct) else f"{risk_pct:.1%}")

            owner_compare_df = pd.DataFrame(
                [
                    ["Average inspection score", selected_row["avg_inspection_score"], peer_df["avg_inspection_score"].median()],
                    ["Average red points", selected_row["avg_red_points"], peer_df["avg_red_points"].median()],
                    ["High-risk rate", selected_row["high_risk_rate"], peer_df["high_risk_rate"].median()],
                    ["Inspection count", selected_row["inspection_count"], peer_df["inspection_count"].median()],
                ],
                columns=["Metric", "My restaurant", "City median"],
            )
            st.dataframe(owner_compare_df, use_container_width=True, hide_index=True)

        elif selected_question_id == "O4":
            if not option_map:
                st.info("No restaurants available.")
                return
            selected_restaurant_label = st.selectbox(
                "Select restaurant",
                list(option_map.keys()),
                index=0,
                key="o4_restaurant",
            )
            selected_business_id = option_map[selected_restaurant_label]
            biz_events = filtered_events[
                filtered_events["business_id"].astype(str).str.strip() == selected_business_id
            ].sort_values("inspection_date_dt", ascending=True).copy()
            if biz_events.empty:
                st.info("No inspection records for this restaurant.")
                return
            biz_events["rating_num"] = pd.to_numeric(biz_events["effective_rating_num"], errors="coerce")
            biz_events["prev_rating_num"] = biz_events["rating_num"].shift(1)
            biz_events["prev_score"] = biz_events["inspection_score"].shift(1)
            biz_events["prev_red"] = biz_events["red_points_total"].shift(1)
            biz_events["prev_blue"] = biz_events["blue_points_total"].shift(1)
            biz_events["prev_violations"] = biz_events["violation_count_total"].shift(1)

            drops = biz_events[
                biz_events["prev_rating_num"].notna()
                & biz_events["rating_num"].notna()
                & (biz_events["rating_num"] > biz_events["prev_rating_num"])
            ].copy()
            if drops.empty:
                st.info("No rating-drop event was detected for this restaurant.")
                return

            drops["prev_rating_label"] = (
                drops["prev_rating_num"].astype(int).astype(str).map(GRADE_CODE_TO_LABEL)
            )
            drops["current_rating_label"] = (
                drops["rating_num"].astype(int).astype(str).map(GRADE_CODE_TO_LABEL)
            )
            drops["rating_change"] = drops["prev_rating_label"] + " -> " + drops["current_rating_label"]
            drops["score_delta"] = drops["inspection_score"] - drops["prev_score"]
            drops["red_delta"] = drops["red_points_total"] - drops["prev_red"]
            drops["blue_delta"] = drops["blue_points_total"] - drops["prev_blue"]
            drops["violation_delta"] = drops["violation_count_total"] - drops["prev_violations"]

            st.dataframe(
                drops[
                    [
                        "inspection_date",
                        "rating_change",
                        "prev_score",
                        "inspection_score",
                        "score_delta",
                        "prev_red",
                        "red_points_total",
                        "red_delta",
                        "prev_blue",
                        "blue_points_total",
                        "blue_delta",
                        "prev_violations",
                        "violation_count_total",
                        "violation_delta",
                    ]
                ].rename(
                    columns={
                        "inspection_date": "Drop date",
                        "prev_score": "Prev score",
                        "inspection_score": "Current score",
                        "score_delta": "Score delta",
                        "prev_red": "Prev red",
                        "red_points_total": "Current red",
                        "red_delta": "Red delta",
                        "prev_blue": "Prev blue",
                        "blue_points_total": "Current blue",
                        "blue_delta": "Blue delta",
                        "prev_violations": "Prev violations",
                        "violation_count_total": "Current violations",
                        "violation_delta": "Violation delta",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=360,
            )

        elif selected_question_id == "O5":
            o5_city = st.selectbox(
                "Area (city)",
                ["All"] + city_options,
                index=0,
                key="o5_city",
            )
            max_date = filtered_violations["inspection_date_dt"].dropna().max()
            if pd.isna(max_date):
                st.info("No violation date data is available.")
                return
            recent_start = max_date - pd.Timedelta(days=90)
            prior_start = recent_start - pd.Timedelta(days=90)

            base = filtered_violations.copy()
            if o5_city != "All":
                base = base[base["city_display"] == o5_city]
            base = base[base["inspection_date_dt"].notna()].copy()
            if base.empty:
                st.info("No records in the selected area.")
                return

            recent = base[
                (base["inspection_date_dt"] >= recent_start)
                & (base["inspection_date_dt"] <= max_date)
            ]
            prior = base[
                (base["inspection_date_dt"] >= prior_start)
                & (base["inspection_date_dt"] < recent_start)
            ]
            recent_cat = recent.groupby("action_category", dropna=False).size().rename("recent_count")
            prior_cat = prior.groupby("action_category", dropna=False).size().rename("prior_count")
            growth = pd.concat([recent_cat, prior_cat], axis=1).fillna(0).reset_index()
            growth["delta"] = growth["recent_count"] - growth["prior_count"]
            growth["growth_rate"] = np.where(
                growth["prior_count"] > 0,
                growth["delta"] / growth["prior_count"],
                np.nan,
            )
            growth = growth.sort_values(by=["delta", "recent_count"], ascending=[False, False])
            if growth.empty:
                st.info("No category records available for window comparison.")
                return

            st.dataframe(
                growth.rename(
                    columns={
                        "action_category": "Category",
                        "recent_count": "Recent 90d count",
                        "prior_count": "Previous 90d count",
                        "delta": "Change (recent - previous)",
                        "growth_rate": "Growth rate",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=360,
            )
            st.bar_chart(
                growth.head(12).set_index("action_category")[["delta"]],
                use_container_width=True,
            )

        elif selected_question_id == "O6":
            if not option_map:
                st.info("No restaurants available.")
                return
            selected_restaurant_label = st.selectbox(
                "Select restaurant",
                list(option_map.keys()),
                index=0,
                key="o6_restaurant",
            )
            selected_business_id = option_map[selected_restaurant_label]
            biz_events = filtered_events[
                filtered_events["business_id"].astype(str).str.strip() == selected_business_id
            ].sort_values("inspection_date_dt", ascending=True).copy()
            if len(biz_events) < 2:
                st.info("At least two inspections are needed to evaluate change.")
                return
            max_window = min(20, len(biz_events))
            window_n = int(
                st.slider(
                    "Lookback inspections",
                    min_value=2,
                    max_value=max_window,
                    value=min(8, max_window),
                    step=1,
                    key="o6_window",
                )
            )
            view = biz_events.tail(window_n).copy()
            first_row = view.iloc[0]
            last_row = view.iloc[-1]

            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Score change", f"{(last_row['inspection_score'] - first_row['inspection_score']):.1f}")
            t2.metric("Red-point change", f"{(last_row['red_points_total'] - first_row['red_points_total']):.1f}")
            t3.metric("Blue-point change", f"{(last_row['blue_points_total'] - first_row['blue_points_total']):.1f}")
            t4.metric("High-risk inspections (window)", f"{int(view['is_high_risk'].sum())}/{len(view)}")

            trend_view = view[
                [
                    "inspection_date_dt",
                    "inspection_score",
                    "red_points_total",
                    "blue_points_total",
                ]
            ].set_index("inspection_date_dt")
            st.line_chart(trend_view, use_container_width=True)
            st.dataframe(
                view[
                    [
                        "inspection_date",
                        "inspection_score",
                        "red_points_total",
                        "blue_points_total",
                        "violation_count_total",
                        "effective_rating_label",
                        "is_high_risk",
                    ]
                ].rename(
                    columns={
                        "inspection_date": "Inspection date",
                        "inspection_score": "Score",
                        "red_points_total": "Red points",
                        "blue_points_total": "Blue points",
                        "violation_count_total": "Violations",
                        "effective_rating_label": "Rating",
                        "is_high_risk": "High risk",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=320,
            )

    elif selected_role == "Regulator":
        if selected_question_id == "R1":
            dim = st.selectbox(
                "Area level",
                ["City", "Zip code"],
                index=0,
                key="r1_dim",
            )
            min_inspections = int(
                st.number_input("Min inspections", min_value=5, max_value=500, value=30, step=5, key="r1_min_ins")
            )
            if dim == "City":
                area_col = "city_display"
            else:
                area_col = "zip_code"

            area_df = filtered_events[filtered_events[area_col].astype(str).str.strip().ne("")].copy()
            if area_df.empty:
                st.info("No area-level data is available.")
                return
            area_agg = (
                area_df.groupby(area_col, dropna=False)
                .agg(
                    inspections=("business_id", "size"),
                    restaurants=("business_id", "nunique"),
                    high_risk_inspections=("is_high_risk", "sum"),
                    high_risk_rate=("is_high_risk", "mean"),
                    avg_red_points=("red_points_total", "mean"),
                )
                .reset_index()
            )
            area_agg = area_agg[area_agg["inspections"] >= min_inspections].copy()
            area_agg = area_agg.sort_values(
                by=["high_risk_rate", "high_risk_inspections", "inspections"],
                ascending=[False, False, False],
            )
            if area_agg.empty:
                st.info("No area passes the current minimum-inspection threshold.")
                return
            st.dataframe(
                area_agg.rename(
                    columns={
                        area_col: dim,
                        "inspections": "Inspections",
                        "restaurants": "Restaurants",
                        "high_risk_inspections": "High-risk inspections",
                        "high_risk_rate": "High-risk rate",
                        "avg_red_points": "Avg red points",
                    }
                ),
                    use_container_width=True,
                    hide_index=True,
                    height=380,
                )

    elif selected_role == "King County":
        quality_bundle = load_king_county_quality_bundle(str(root), clean_text(payload.get("run_id", "")))
        audit_snapshot_raw = quality_bundle.get("audit_snapshot", {})
        audit_snapshot = audit_snapshot_raw if isinstance(audit_snapshot_raw, dict) else {}
        issue_catalog_df = pd.DataFrame(quality_bundle.get("issue_catalog", []))
        samples_raw = quality_bundle.get("samples", {})
        sample_map = samples_raw if isinstance(samples_raw, dict) else {}
        history_df = pd.DataFrame(quality_bundle.get("history", []))

        def snapshot_int(key: str) -> int:
            value = audit_snapshot.get(key, 0)
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return 0

        def flatten_samples_by_prefix(prefixes: List[str]) -> pd.DataFrame:
            rows: List[Dict[str, Any]] = []
            for group_name, records in sample_map.items():
                if not any(str(group_name).startswith(prefix) for prefix in prefixes):
                    continue
                if not isinstance(records, list):
                    continue
                for rec in records:
                    if not isinstance(rec, dict):
                        continue
                    rows.append(
                        {
                            "Issue group": clean_text(group_name),
                            "Restaurant": clean_text(rec.get("name", "")),
                            "Address": clean_text(rec.get("address", "")),
                            "City token": clean_text(rec.get("city", "")),
                            "Zip": clean_text(rec.get("zip_code", "")),
                            "Inspection date": clean_text(rec.get("inspection_date", "")),
                            "Latitude": clean_text(rec.get("latitude", "")),
                            "Longitude": clean_text(rec.get("longitude", "")),
                        }
                    )
            return pd.DataFrame(rows)

        if selected_question_id == "K1":
            gap_df = build_official_grade_gap_df(filtered_events)
            if gap_df.empty:
                st.info("No current differences were found between the county's published grade and the dashboard calculation.")
                return
            mismatch_counts = (
                gap_df.groupby(["dashboard_rating_label", "official_rating_label"], dropna=False)
                .size()
                .reset_index(name="restaurants")
                .sort_values(["restaurants", "dashboard_rating_label"], ascending=[False, True])
            )
            na_terms = {"", RATING_NOT_AVAILABLE_LABEL}
            true_gap_df = gap_df[
                ~gap_df["dashboard_rating_label"].astype(str).str.strip().isin(na_terms)
                & ~gap_df["official_rating_label"].astype(str).str.strip().isin(na_terms)
            ].copy()

            k1, k2, k3 = st.columns(3)
            k1.metric("Restaurants with differences", f"{len(gap_df):,}")
            k2.metric("Mismatch patterns", f"{len(mismatch_counts):,}")
            k3.metric("True mismatches (excluding N/A)", f"{len(true_gap_df):,}")
            st.dataframe(
                mismatch_counts.rename(
                    columns={
                        "dashboard_rating_label": "Dashboard rating",
                        "official_rating_label": "Official county grade",
                        "restaurants": "Restaurants",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=220,
            )
            st.dataframe(
                gap_df[
                    [
                        "display_name",
                        "city_display",
                        "full_address_clean",
                        "inspection_date",
                        "risk_category",
                        "official_rating_label",
                        "dashboard_rating_label",
                        "avg_red_points_used",
                        "routine_inspections_used",
                        "city_scope",
                        "city_cleaning_reason",
                    ]
                ].rename(
                    columns={
                        "display_name": "Restaurant",
                        "city_display": "City",
                        "full_address_clean": "Address",
                        "inspection_date": "Latest inspection date",
                        "risk_category": "Risk category",
                        "official_rating_label": "Official county grade",
                        "dashboard_rating_label": "Dashboard rating",
                        "avg_red_points_used": "Avg red points used",
                        "routine_inspections_used": "Routine inspections used",
                        "city_scope": "City scope",
                        "city_cleaning_reason": "City cleaning reason",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=420,
            )

        elif selected_question_id == "K2":
            outside_tokens = audit_snapshot.get("top_known_outside_tokens", {})
            unknown_tokens = audit_snapshot.get("top_unknown_city_tokens", {})

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Outside-city rows", f"{snapshot_int('city_known_outside_rows'):,}")
            k2.metric("Unknown city-token rows", f"{snapshot_int('city_unknown_rows'):,}")
            k3.metric("Geo out-of-bounds rows", f"{snapshot_int('geo_out_of_bounds_rows'):,}")
            k4.metric("Distinct outside-city tokens", f"{snapshot_int('city_known_outside_distinct'):,}")

            outside_df = (
                pd.DataFrame(
                    [{"City token": k, "Rows": int(v)} for k, v in dict(outside_tokens).items()]
                ).sort_values("Rows", ascending=False)
                if isinstance(outside_tokens, dict) and outside_tokens
                else pd.DataFrame(columns=["City token", "Rows"])
            )
            unknown_df = (
                pd.DataFrame(
                    [{"City token": k, "Rows": int(v)} for k, v in dict(unknown_tokens).items()]
                ).sort_values("Rows", ascending=False)
                if isinstance(unknown_tokens, dict) and unknown_tokens
                else pd.DataFrame(columns=["City token", "Rows"])
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Top outside-city tokens**")
                st.dataframe(outside_df, use_container_width=True, hide_index=True, height=280)
            with c2:
                st.markdown("**Top unknown tokens**")
                st.dataframe(unknown_df, use_container_width=True, hide_index=True, height=280)

            sample_df = flatten_samples_by_prefix(["outside_", "variant_"])
            st.markdown("**Sample records for scope/locality issues**")
            if sample_df.empty:
                st.info("No sample rows found for scope/locality issue groups.")
            else:
                st.dataframe(sample_df, use_container_width=True, hide_index=True, height=360)

        elif selected_question_id == "K3":
            latest_df = latest_events_by_business(filtered_events).copy()
            latest_df["city_cleaning_reason"] = latest_df["city_cleaning_reason"].astype(str).str.strip()
            latest_df["city_scope"] = latest_df["city_scope"].astype(str).str.strip()
            reason_df = (
                latest_df["city_cleaning_reason"]
                .replace("", "(empty)")
                .value_counts()
                .rename_axis("City cleaning reason")
                .reset_index(name="Businesses")
            )
            reason_df["Share"] = np.where(
                len(latest_df) > 0, reason_df["Businesses"] / len(latest_df), np.nan
            )
            reason_df["Share"] = reason_df["Share"].map(
                lambda x: "-" if pd.isna(x) else f"{x:.2%}"
            )

            scope_df = (
                latest_df["city_scope"]
                .replace("", "(empty)")
                .value_counts()
                .rename_axis("City scope")
                .reset_index(name="Businesses")
            )
            corrected_businesses = int(
                latest_df["city_cleaning_reason"].ne("exact").sum()
            )
            outside_scope_businesses = int(
                latest_df["city_scope"].str.contains("Outside", case=False, na=False).sum()
            )

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("City casing-inconsistent rows", f"{snapshot_int('city_case_issue_rows'):,}")
            k2.metric("ZIPs with 3+ city variants", f"{snapshot_int('zip_with_3plus_city_variants'):,}")
            k3.metric("Businesses with city correction", f"{corrected_businesses:,}")
            k4.metric("Businesses outside scope", f"{outside_scope_businesses:,}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**City cleaning reason distribution (latest business snapshot)**")
                st.dataframe(reason_df, use_container_width=True, hide_index=True, height=340)
            with c2:
                st.markdown("**City scope distribution (latest business snapshot)**")
                st.dataframe(scope_df, use_container_width=True, hide_index=True, height=340)

        elif selected_question_id == "K4":
            if "category" in issue_catalog_df.columns:
                identifier_df = issue_catalog_df[
                    issue_catalog_df["category"].astype(str).eq("Identifier Completeness")
                ].copy()
            else:
                identifier_df = pd.DataFrame()

            k1, k2, k3 = st.columns(3)
            k1.metric("Missing inspection_serial_num", f"{snapshot_int('missing_inspection_serial_rows'):,}")
            k2.metric("Unparseable inspection_date rows", f"{snapshot_int('raw_date_parse_fail_rows'):,}")
            k3.metric("Missing violation_record_id", f"{snapshot_int('missing_violation_record_id_rows'):,}")

            if identifier_df.empty:
                st.info("No identifier-completeness issue catalog rows found for this run.")
            else:
                identifier_df["share_pct"] = pd.to_numeric(identifier_df["share_pct"], errors="coerce")
                identifier_df["Share"] = identifier_df["share_pct"].map(
                    lambda x: "-" if pd.isna(x) else f"{x:.2f}%"
                )
                st.dataframe(
                    identifier_df[
                        ["issue", "count", "denominator", "Share", "severity", "suggested_action"]
                    ].rename(
                        columns={
                            "issue": "Issue",
                            "count": "Rows affected",
                            "denominator": "Denominator",
                            "severity": "Severity",
                            "suggested_action": "Suggested action",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=320,
                )

            date_sample_df = flatten_samples_by_prefix(["date_parse_fail"])
            st.markdown("**Sample rows for date/key integrity issues**")
            if date_sample_df.empty:
                st.info("No date/key sample rows found.")
            else:
                st.dataframe(date_sample_df, use_container_width=True, hide_index=True, height=300)

        elif selected_question_id == "K5":
            if "category" in issue_catalog_df.columns:
                semantic_df = issue_catalog_df[
                    issue_catalog_df["category"].astype(str).isin(
                        ["Semantic Completeness", "Text Quality", "Numeric Validity"]
                    )
                ].copy()
            else:
                semantic_df = pd.DataFrame()

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Missing violation_type rows", f"{snapshot_int('violation_type_missing_rows'):,}")
            k2.metric("Missing official grade rows", f"{snapshot_int('grade_missing_rows'):,}")
            k3.metric("Truncated violation text rows", f"{snapshot_int('violation_desc_truncated_rows'):,}")
            k4.metric("Negative-score rows", f"{snapshot_int('score_negative_rows'):,}")

            if semantic_df.empty:
                st.info("No semantic/text issue catalog rows found for this run.")
            else:
                semantic_df["share_pct"] = pd.to_numeric(semantic_df["share_pct"], errors="coerce")
                semantic_df["Share"] = semantic_df["share_pct"].map(
                    lambda x: "-" if pd.isna(x) else f"{x:.2f}%"
                )
                st.dataframe(
                    semantic_df[
                        ["category", "issue", "count", "denominator", "Share", "severity", "suggested_action"]
                    ].rename(
                        columns={
                            "category": "Category",
                            "issue": "Issue",
                            "count": "Rows affected",
                            "denominator": "Denominator",
                            "severity": "Severity",
                            "suggested_action": "Suggested action",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=340,
                )

            sample_df = flatten_samples_by_prefix(
                ["truncated_violation", "grade_missing_with_risk_text", "negative_score"]
            )
            st.markdown("**Sample rows for semantic/text/numeric issues**")
            if sample_df.empty:
                st.info("No semantic/text sample rows found.")
            else:
                st.dataframe(sample_df, use_container_width=True, hide_index=True, height=320)

        elif selected_question_id == "K6":
            if history_df.empty:
                st.info("No cross-run quality history is available yet. Keep daily snapshots in outputs/analysis.")
                return
            history_df = history_df.sort_values("run_id").reset_index(drop=True)
            for col in [
                "city_outside_rate",
                "city_unknown_rate",
                "date_parse_fail_rate",
                "violation_truncated_rate",
                "true_mismatch_rate",
            ]:
                history_df[col] = pd.to_numeric(history_df[col], errors="coerce")

            latest_hist = history_df.iloc[-1]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Snapshots", f"{len(history_df):,}")
            k2.metric("Latest run", clean_text(latest_hist.get("run_id", "")) or "-")
            k3.metric(
                "Latest outside-city rate",
                "-" if pd.isna(latest_hist.get("city_outside_rate", np.nan)) else f"{float(latest_hist['city_outside_rate']):.2%}",
            )
            k4.metric(
                "Latest true-mismatch rate",
                "-" if pd.isna(latest_hist.get("true_mismatch_rate", np.nan)) else f"{float(latest_hist['true_mismatch_rate']):.2%}",
            )

            trend_chart_df = history_df.set_index("run_id")[
                [
                    "city_outside_rate",
                    "date_parse_fail_rate",
                    "violation_truncated_rate",
                    "true_mismatch_rate",
                ]
            ]
            st.line_chart(trend_chart_df, use_container_width=True)

            display_df = history_df.copy()
            display_df["rows_raw"] = pd.to_numeric(display_df["rows_raw"], errors="coerce").astype("Int64")
            for col in [
                "city_outside_rate",
                "city_unknown_rate",
                "date_parse_fail_rate",
                "violation_truncated_rate",
                "true_mismatch_rate",
            ]:
                display_df[col] = display_df[col].map(lambda x: "-" if pd.isna(x) else f"{x:.2%}")
            st.dataframe(
                display_df.rename(
                    columns={
                        "run_id": "Run ID",
                        "rows_raw": "Raw rows",
                        "city_outside_rate": "Outside-city rate",
                        "city_unknown_rate": "Unknown-city-token rate",
                        "date_parse_fail_rate": "Date-parse-fail rate",
                        "violation_truncated_rate": "Truncated-violation-text rate",
                        "true_mismatch_rate": "True-rating-mismatch rate",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=320,
            )
            if len(history_df) < 2:
                st.caption("Only one snapshot is currently available; trend direction becomes meaningful after 2+ runs.")


def render_essay_card(title: str, body: str) -> None:
    st.markdown(
        (
            "<div class='essay-card'>"
            f"<h4>{escape(title)}</h4>"
            f"<p>{escape(body)}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str, kicker: str = "") -> None:
    kicker_html = f"<div class='section-kicker'>{escape(kicker)}</div>" if kicker else ""
    st.markdown(
        (
            "<div class='section-hero'>"
            f"{kicker_html}"
            f"<div class='section-title'>{escape(title)}</div>"
            f"<div class='section-subtitle'>{escape(subtitle)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_subsection_label(title: str, hint: str = "") -> None:
    st.markdown(f"<div class='subsection-label'>{escape(title)}</div>", unsafe_allow_html=True)
    if hint:
        st.markdown(f"<div class='panel-hint'>{escape(hint)}</div>", unsafe_allow_html=True)


def render_takeaway_box(text: str) -> None:
    st.markdown(
        (
            "<div class='takeaway-box'>"
            "<strong>Takeaway.</strong> "
            f"{escape(text)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def build_manifest_metrics_df(models: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name, model_info in models.items():
        metrics = model_info.get("metrics", {})
        rows.append(
            {
                "Model": model_name,
                "Accuracy": float(metrics.get("Accuracy", float("nan"))),
                "Precision": float(metrics.get("Precision", float("nan"))),
                "Recall": float(metrics.get("Recall", float("nan"))),
                "F1": float(metrics.get("F1", float("nan"))),
                "ROC_AUC": float(metrics.get("ROC_AUC", float("nan"))),
            }
        )
    return pd.DataFrame(rows).sort_values(by=["F1", "ROC_AUC"], ascending=False)


def render_homework_field_dictionary() -> None:
    st.markdown("**Field Dictionary for the Two Source Tables**")
    st.caption(
        "The inspection-event table is the modeling backbone, while the violation-level table adds detailed findings and remediation guidance."
    )
    for idx, (group_name, rows) in enumerate(HOMEWORK_FIELD_GROUPS):
        with st.expander(group_name, expanded=idx == 0):
            field_df = pd.DataFrame(rows, columns=["Field", "Meaning"])
            st.dataframe(field_df, use_container_width=True, hide_index=True)


def build_executive_summary_tab(
    events_df: pd.DataFrame,
    violations_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    root: Path,
) -> None:
    render_section_header(
        "Executive Summary",
        "Owner-facing framing of the data asset, prediction target, business question, and the analysis path used in the assignment workflow.",
        "Assignment Workflow",
    )
    manifest_bundle = load_predict_manifest(str(root))
    if not manifest_bundle.get("available", False):
        st.warning(manifest_bundle.get("message", "Predict manifest not available."))
        return

    manifest = manifest_bundle.get("manifest", {})
    models = manifest.get("models", {})
    metrics_df = build_manifest_metrics_df(models)
    model_dataset = build_next_inspection_dataset(events_df)
    best_model_name = clean_text(manifest.get("best_model_name", ""))
    best_row = metrics_df[metrics_df["Model"] == best_model_name]
    if best_row.empty and not metrics_df.empty:
        best_row = metrics_df.head(1)
        best_model_name = clean_text(best_row.iloc[0]["Model"])

    shap_csv = clean_text((manifest.get("shap") or {}).get("mean_abs_shap_csv", ""))
    shap_df = load_csv_safe(shap_csv) if shap_csv else pd.DataFrame()
    top_signals = ", ".join(
        str(name).replace("num__", "").replace("cat__", "")
        for name in shap_df["feature"].head(4).tolist()
    ) or "grade_num, inspection_score, red_points_total, and inspection_result"

    valid_dates = model_dataset["inspection_date_dt"].dropna()
    min_date = valid_dates.min().date().isoformat() if not valid_dates.empty else "-"
    max_date = valid_dates.max().date().isoformat() if not valid_dates.empty else "-"
    positive_rate = (
        float(model_dataset["target_next_high_risk"].mean()) if len(model_dataset) else float("nan")
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Inspection events", f"{len(events_df):,}")
    k2.metric("Violation rows", f"{len(violations_df):,}")
    k3.metric("Model rows", f"{len(model_dataset):,}")
    k4.metric("Restaurants", f"{summary_df['business_id'].nunique():,}")

    render_subsection_label(
        "Project framing",
        "This section should read like a concise business brief before the reader enters the descriptive, modeling, and explainability tabs.",
    )
    intro_row1 = st.columns(2, gap="large")
    with intro_row1[0]:
        render_essay_card(
            "Dataset and database purpose",
            "This project uses King County restaurant inspection records to turn public food-safety data into an owner-facing decision tool. The database combines an inspection-event table, which records each inspection visit and its score, grade, result, and risk signals, with a violation-level table that stores individual findings, point severity, and remediation guidance.",
        )
    with intro_row1[1]:
        render_essay_card(
            "Prediction target",
            "The prediction target is target_next_high_risk: whether the same restaurant's next recorded inspection falls into the high-risk class. In this project, an inspection is treated as high risk when the published grade is Needs to Improve or when red points reach 25 or more.",
        )
    intro_row2 = st.columns(2, gap="large")
    with intro_row2[0]:
        render_essay_card(
            "Why this matters",
            "For a restaurant owner, the value is operational rather than academic. If the latest inspection profile can estimate next-inspection risk early enough, the owner can focus on the controllable signals most likely to trigger a poor next outcome before the next county visit happens.",
        )

    if not best_row.empty:
        row = best_row.iloc[0]
        with intro_row2[1]:
            render_essay_card(
                "Approach and key findings",
                f"The analysis converts repeated inspections into a next-inspection prediction dataset with {len(MODEL_NUMERIC_FEATURES)} numeric features and {len(MODEL_CATEGORICAL_FEATURES)} categorical features. Using the current saved artifacts, {best_model_name} is the best overall model with F1={float(row['F1']):.4f} and ROC_AUC={float(row['ROC_AUC']):.4f}; the recurring global signals are {top_signals}.",
            )

    render_subsection_label(
        "Database structure and business question",
        "These two tables are the backbone of the homework story: one defines the sequence of inspections, the other explains what was found and how to respond.",
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        render_essay_card(
            "inspection_event table",
            "One row represents one inspection event for one establishment. It is the table used to build the prediction target because it preserves inspection order over time and stores the current-state signals that are available before the next inspection occurs.",
        )
    with c2:
        render_essay_card(
            "dashboard_violation_explained table",
            "One row represents one violation finding linked back to an inspection event. This table explains what went wrong, how severe it was, and what remediation action category or priority should be considered next.",
        )

    framing_col, stats_col = st.columns([1.35, 1.0], gap="large")
    with framing_col:
        render_subsection_label(
            "Homework framing",
            "The assignment question has to support descriptive analysis, model comparison, and local explanation in a single workflow.",
        )
        st.write(
            "Primary question: "
            f"**{HOMEWORK_OWNER_QUESTION}**"
        )
        st.write(
            "The descriptive tab tests whether the latest inspection profile already contains visible warning signals. "
            "The model tab measures whether multiple algorithms can convert those signals into useful predictive performance. "
            "The explainability tab then shows which factors move one restaurant's predicted risk up or down for a decision-maker."
        )

    stats_df = pd.DataFrame(
        [
            ["Coverage window", f"{min_date} to {max_date}"],
            ["Target positive rate", f"{positive_rate:.2%}" if pd.notna(positive_rate) else "-"],
            ["Model numeric features", ", ".join(MODEL_NUMERIC_FEATURES)],
            ["Model categorical features", ", ".join(MODEL_CATEGORICAL_FEATURES)],
        ],
        columns=["Item", "Value"],
    )
    with stats_col:
        render_subsection_label("Coverage and feature scope")
        st.dataframe(stats_df, use_container_width=True, hide_index=True, height=210)

    render_subsection_label(
        "Field dictionary",
        "The full field map is kept in expanders so the section stays readable while still covering the grading requirement.",
    )
    render_homework_field_dictionary()


def build_descriptive_analytics_tab(events_df: pd.DataFrame) -> None:
    model_dataset = build_next_inspection_dataset(events_df)
    if model_dataset.empty:
        st.info("No data available for descriptive analytics.")
        return

    render_section_header(
        "Descriptive Analytics",
        "Compact evidence for whether the current inspection profile already contains visible signals of future high-risk outcomes.",
        "Assignment Workflow",
    )
    positive_rate = float(model_dataset["target_next_high_risk"].mean())
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Model rows", f"{len(model_dataset):,}")
    d2.metric("Restaurants", f"{events_df['business_id'].nunique():,}")
    d3.metric("Positive class rate", f"{positive_rate:.1%}")
    d4.metric("Model features", f"{len(MODEL_ALL_FEATURES)}")
    render_subsection_label(
        "Dataset introduction",
        "The descriptive tab should establish dataset size, target balance, and the main warning signals before model comparison begins.",
    )
    render_essay_card(
        "Dataset introduction",
        f"The modeling table contains {len(model_dataset):,} ordered inspection records drawn from {events_df['business_id'].nunique():,} restaurants. It predicts the next inspection's high-risk flag from the current inspection profile using {len(MODEL_ALL_FEATURES)} features ({len(MODEL_NUMERIC_FEATURES)} numeric and {len(MODEL_CATEGORICAL_FEATURES)} categorical).",
    )
    render_takeaway_box(
        "The data is large enough for a meaningful classification workflow, but the target is imbalanced, so F1, recall, ROC-AUC, and class-aware modeling choices matter more than raw accuracy alone."
    )

    render_subsection_label(
        "Signal comparison charts",
        "These plots are intentionally smaller and paired so the user can scan the full narrative without endless vertical scrolling.",
    )
    target_counts = (
        model_dataset["target_next_high_risk"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    low_score = model_dataset.loc[model_dataset["target_next_high_risk"] == 0, "inspection_score"].dropna()
    high_score = model_dataset.loc[model_dataset["target_next_high_risk"] == 1, "inspection_score"].dropna()

    low_red = model_dataset.loc[model_dataset["target_next_high_risk"] == 0, "red_points_total"].dropna()
    high_red = model_dataset.loc[model_dataset["target_next_high_risk"] == 1, "red_points_total"].dropna()

    low_viol = model_dataset.loc[model_dataset["target_next_high_risk"] == 0, "violation_count_total"].dropna()
    high_viol = model_dataset.loc[model_dataset["target_next_high_risk"] == 1, "violation_count_total"].dropna()
    pair_top_left, pair_top_right = st.columns(2, gap="large")
    with pair_top_left:
        st.markdown("**Target distribution of target_next_high_risk**")
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        ax.bar(
            ["Low-risk next inspection", "High-risk next inspection"],
            target_counts.values,
            color=["#8cd3dd", "#0d8a98"],
        )
        ax.set_ylabel("Inspection rows")
        ax.set_title("Target distribution")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            "Most inspection rows lead to a low-risk next inspection, while only a small minority lead to a future high-risk outcome. "
            "That imbalance is exactly why this project emphasizes class-aware metrics, uses balanced or tuned models, and avoids reading accuracy as the primary quality signal."
        )
        render_takeaway_box(
            f"Only about {positive_rate:.1%} of rows are positive, so the modeling problem is detection of a relatively rare but operationally important failure state."
        )
    with pair_top_right:
        st.markdown("**Inspection score by future high-risk target**")
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        ax.boxplot([low_score, high_score], tick_labels=["Next low risk", "Next high risk"], showfliers=False)
        ax.set_ylabel("Inspection score")
        ax.set_title("Current score vs. future target")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            "The high-risk-next group has a clearly worse current inspection score distribution, which suggests the current inspection already contains usable early warning information. "
            "The distributions still overlap materially, so a single cut-off score is not enough; the prediction task needs multiple variables working together."
        )
        render_takeaway_box(
            "Inspection score is informative, but not decisive on its own. Owners should treat it as one signal in a broader risk profile rather than as a standalone decision rule."
        )

    pair_mid_left, pair_mid_right = st.columns(2, gap="large")
    with pair_mid_left:
        st.markdown("**Red points by future high-risk target**")
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        ax.boxplot([low_red, high_red], tick_labels=["Next low risk", "Next high risk"], showfliers=False)
        ax.set_ylabel("Red points")
        ax.set_title("Current red points vs. future target")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            "Red points are much more concentrated in the group that later returns as high risk, which is consistent with their role as a high-severity inspection signal. "
            "This is useful for restaurant owners because red points are directly actionable: reducing them should improve both current compliance and future inspection resilience."
        )
        render_takeaway_box(
            "Red-point severity is one of the strongest warning indicators in the dataset and should be the first operational triage signal for a restaurant owner."
        )
    with pair_mid_right:
        st.markdown("**Violation count by future high-risk target**")
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        ax.boxplot([low_viol, high_viol], tick_labels=["Next low risk", "Next high risk"], showfliers=False)
        ax.set_ylabel("Violation count")
        ax.set_title("Current violation count vs. future target")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            "Inspection rows that later lead to a high-risk next visit also tend to carry more total violations in the current visit. "
            "This suggests that repeated process-control weakness, not just one extreme finding, is part of the risk pattern captured by the model."
        )
        render_takeaway_box(
            "A high volume of findings is itself a signal that the business may not have stabilized its food-safety process before the next inspection."
        )

    result_rate = (
        model_dataset.groupby("inspection_result", dropna=False)
        .agg(high_risk_rate=("target_next_high_risk", "mean"), rows=("target_next_high_risk", "size"))
        .reset_index()
    )
    result_rate = result_rate[result_rate["rows"] >= 100].sort_values("high_risk_rate", ascending=False).head(10)
    pair_bottom_left, pair_bottom_right = st.columns(2, gap="large")
    if not result_rate.empty:
        with pair_bottom_left:
            st.markdown("**Inspection result by future high-risk rate**")
            fig, ax = plt.subplots(figsize=(5.7, 3.4))
            ax.bar(result_rate["inspection_result"], result_rate["high_risk_rate"], color="#0d8a98")
            ax.set_ylabel("Future high-risk rate")
            ax.set_xlabel("Current inspection result")
            ax.set_title("Outcome categories and future risk")
            ax.tick_params(axis="x", rotation=35)
            ax.set_ylim(0, min(1.0, float(result_rate["high_risk_rate"].max()) * 1.15 + 0.02))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                "Not all published inspection results carry the same downstream implication. "
                "Outcome categories such as unsatisfactory or return-oriented results show materially higher future high-risk rates, which supports using the categorical outcome itself in the model."
            )
            render_takeaway_box(
                "The current inspection result encodes process state. Owners should treat an unsatisfactory or return-oriented status as a strong signal that next-visit risk remains elevated."
            )

    city_rate = (
        model_dataset.groupby("city_canonical", dropna=False)
        .agg(high_risk_rate=("target_next_high_risk", "mean"), rows=("target_next_high_risk", "size"))
        .reset_index()
    )
    city_rate = city_rate[city_rate["rows"] >= 200].sort_values("high_risk_rate", ascending=False).head(12)
    if not city_rate.empty:
        with pair_bottom_right:
            st.markdown("**City-level future high-risk pattern**")
            fig, ax = plt.subplots(figsize=(5.7, 3.4))
            ax.bar(city_rate["city_canonical"].map(format_city_name), city_rate["high_risk_rate"], color="#27c2a8")
            ax.set_ylabel("Future high-risk rate")
            ax.set_xlabel("City")
            ax.set_title("Higher-volume cities by future high-risk rate")
            ax.tick_params(axis="x", rotation=35)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                "Even after requiring a reasonable number of rows per city, future high-risk rates are not uniform across locations. "
                "That pattern does not mean geography alone causes risk, but it indicates that city context may proxy for operating environment, restaurant mix, or enforcement intensity."
            )
            render_takeaway_box(
                "Location is a useful context feature, but it should be interpreted as environment and mix, not as a causal explanation by itself."
            )

    render_subsection_label(
        "Correlation structure",
        "The heatmap stays full-width because it is a cross-signal reference rather than a single-variable chart.",
    )
    st.markdown("**Correlation heatmap**")
    corr_cols = [
        "inspection_score",
        "red_points_total",
        "blue_points_total",
        "violation_count_total",
        "grade_num",
        "is_high_risk",
        "target_next_high_risk",
    ]
    corr_df = model_dataset[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    im = ax.imshow(corr_df.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.columns)
    ax.set_title("Correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "The strongest correlations cluster among score, point totals, violation count, and current high-risk status, which confirms that these variables describe related hygiene severity dimensions. "
        "None of them fully subsume the target, which is why combining them in a predictive model remains justified."
    )
    render_takeaway_box(
        "The heatmap supports the modeling strategy: related signals move together, but no single column is enough to explain next-inspection risk by itself."
    )


def build_model_performance_tab(root: Path) -> None:
    manifest_bundle = load_predict_manifest(str(root))
    if not manifest_bundle.get("available", False):
        st.warning(manifest_bundle.get("message", "Predict manifest not available."))
        return

    manifest = manifest_bundle.get("manifest", {})
    models = manifest.get("models", {})
    metrics_df = build_manifest_metrics_df(models)
    if metrics_df.empty:
        st.warning("No trained models found in manifest.")
        return

    render_section_header(
        "Model Performance",
        "Compare saved classification models, review tuned settings, and keep the bonus MLP artifacts visible without forcing oversized plots.",
        "Assignment Workflow",
    )
    render_essay_card(
        "Data preparation and evaluation design",
        "The workflow builds X and y from the event-level dataset, orders inspections chronologically, keeps a held-out test set, and applies preprocessing inside each saved pipeline. "
        "Numerical features are imputed, categorical features are encoded, class imbalance is handled through model settings or metric choice, and all final metrics shown here come from saved test-set evaluation artifacts.",
    )

    show_df = metrics_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
        show_df[col] = show_df[col].round(4)

    render_subsection_label(
        "Comparison summary",
        "The table and F1 view stay near the top so the reader can see the leaderboard before opening model-specific detail panels.",
    )
    st.markdown("**Model comparison summary table**")
    st.dataframe(show_df, use_container_width=True, hide_index=True)
    st.markdown("**Key metric comparison (F1)**")
    st.bar_chart(show_df.set_index("Model")[["F1"]], use_container_width=True)

    best_model_name = clean_text(manifest.get("best_model_name", "")) or clean_text(show_df.iloc[0]["Model"])
    best_row = show_df[show_df["Model"] == best_model_name].head(1)
    if not best_row.empty:
        row = best_row.iloc[0]
        render_essay_card(
            "Comparison interpretation",
            f"{best_model_name} is the current best overall model on F1 with ROC_AUC={row['ROC_AUC']:.4f}. "
            "That result is not surprising because the task mixes strong linear severity signals with a relatively rare target, so a balanced logistic baseline can remain competitive while still being interpretable.",
        )
    render_takeaway_box(
        "Accuracy alone is misleading in this dataset. The comparison should be read through F1, recall, and ROC-AUC because missing a future high-risk case is more costly than labeling too many rows as safe."
    )

    render_subsection_label(
        "Best hyperparameters",
        "The saved manifest keeps the tuning output explicit so the workflow can be audited model by model.",
    )
    st.markdown("**Best hyperparameters by model**")
    params_df = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Best Hyperparameters": json.dumps(model_info.get("best_params", {}), ensure_ascii=False)
                if model_info.get("best_params")
                else "Baseline defaults / no grid search",
            }
            for model_name, model_info in models.items()
        ]
    )
    st.dataframe(params_df, use_container_width=True, hide_index=True)

    render_subsection_label(
        "ROC curves",
        "These overview plots stay in a two-column grid to avoid one long image wall.",
    )
    st.markdown("**ROC curves for all classification models**")
    roc_cols = st.columns(2)
    roc_idx = 0
    for model_name, model_info in models.items():
        roc_path = clean_text(model_info.get("roc_plot_path", ""))
        if roc_path and Path(roc_path).exists():
            with roc_cols[roc_idx % 2]:
                render_static_image(roc_path, caption=f"ROC curve — {model_name}", width="stretch")
            roc_idx += 1

    for model_name, model_info in models.items():
        metrics = model_info.get("metrics", {})
        extra = model_info.get("extra", {}) or {}
        with st.expander(f"{model_name} details", expanded=(model_name == best_model_name)):
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{float(metrics.get('Accuracy', 0.0)):.4f}")
            m2.metric("Precision", f"{float(metrics.get('Precision', 0.0)):.4f}")
            m3.metric("Recall", f"{float(metrics.get('Recall', 0.0)):.4f}")
            m4.metric("F1", f"{float(metrics.get('F1', 0.0)):.4f}")
            m5.metric("ROC_AUC", f"{float(metrics.get('ROC_AUC', 0.0)):.4f}")
            st.json(model_info.get("best_params", {}), expanded=False)

            media_col_left, media_col_right = st.columns(2, gap="large")
            roc_path = clean_text(model_info.get("roc_plot_path", ""))
            if roc_path and Path(roc_path).exists():
                with media_col_left:
                    render_static_image(roc_path, caption=f"ROC curve — {model_name}", width="stretch")

            tree_plot_path = clean_text(extra.get("tree_plot_path", ""))
            if tree_plot_path and Path(tree_plot_path).exists():
                with media_col_right:
                    render_static_image(tree_plot_path, caption="Best tuned decision tree", width="stretch")

            feature_csv = clean_text(extra.get("feature_importance_csv", "")) or clean_text(
                extra.get("coefficients_csv", "")
            )
            if feature_csv and Path(feature_csv).exists():
                feat_df = load_csv_safe(feature_csv)
                st.dataframe(feat_df.head(20), use_container_width=True, hide_index=True)
                value_col = "importance" if "importance" in feat_df.columns else "abs_coefficient"
                if value_col in feat_df.columns and "feature" in feat_df.columns:
                    st.bar_chart(feat_df.head(15).set_index("feature")[[value_col]], use_container_width=True)

            history_plot = clean_text(extra.get("history_plot_path", ""))
            if history_plot and Path(history_plot).exists():
                render_static_image(history_plot, caption="MLP training history", width="stretch")

            tuning_csv = clean_text(extra.get("tuning_results_csv", ""))
            tuning_plot = clean_text(extra.get("tuning_plot_path", ""))
            if tuning_csv and Path(tuning_csv).exists():
                tuning_df = load_csv_safe(tuning_csv)
                st.markdown("**Bonus — MLP hyperparameter tuning results**")
                st.dataframe(tuning_df, use_container_width=True, hide_index=True)
            if tuning_plot and Path(tuning_plot).exists():
                render_static_image(tuning_plot, caption="MLP tuning top configurations", width="stretch")

    render_essay_card(
        "Model trade-offs",
        "The strongest score comes from Logistic Regression, which keeps the final prediction layer relatively interpretable. "
        "Tree-based models remain valuable because they capture threshold effects and support SHAP-based explanation, while the MLP demonstrates a non-linear alternative and satisfies the assignment's neural-network requirement even though its final F1 does not beat the baseline.",
    )


def build_explainability_prediction_tab(
    events_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    root: Path,
) -> None:
    render_section_header(
        "Explainability & Interactive Prediction",
        "Use one saved model for live probability scoring and a tree-based SHAP path for consistent global and local explanations.",
        "Assignment Workflow",
    )
    manifest_bundle = load_predict_manifest(str(root))
    if not manifest_bundle.get("available", False):
        st.warning(manifest_bundle.get("message", "Predict manifest not available."))
        return

    manifest = manifest_bundle.get("manifest", {})
    models = manifest.get("models", {})
    if not models:
        st.warning("No trained models found in manifest.")
        return

    available_prediction_models = {
        name: info
        for name, info in models.items()
        if clean_text(info.get("kind", "")) != "pytorch_mlp" or TORCH_AVAILABLE
    }
    if not available_prediction_models:
        st.warning("No prediction models are available in the current deployment environment.")
        return
    if len(available_prediction_models) != len(models):
        st.caption(
            "MLP remains documented in Model Performance, but live MLP inference is disabled in the lightweight cloud deployment to keep startup stable."
        )

    shap_meta = manifest.get("shap", {})
    shap_model_name = get_best_shap_model_name(manifest)
    render_essay_card(
        "Explainability setup",
        f"Prediction can use any saved model, but the SHAP explanation view is anchored to {shap_model_name}. "
        "This keeps the prediction interaction flexible while using a tree-based model for local additive explanation and the required waterfall decomposition.",
    )

    render_subsection_label(
        "Global SHAP views",
        "The global views are paired side by side so the reader can compare directional spread and overall importance without oversized plots.",
    )
    shap_summary_path = clean_text(shap_meta.get("summary_plot_path", ""))
    shap_bar_path = clean_text(shap_meta.get("bar_plot_path", ""))
    shap_left, shap_right = st.columns(2, gap="large")
    if shap_summary_path and Path(shap_summary_path).exists():
        with shap_left:
            render_static_image(shap_summary_path, caption=f"SHAP summary plot — {shap_model_name}", width="stretch")
            st.caption(
                "The beeswarm plot shows both importance and direction: points to the right push the prediction toward the positive class, while points to the left pull it down. "
                "Features near the top influence the model most often and most strongly across the sample."
            )
            render_takeaway_box(
                "Global SHAP confirms that the model is reacting to a compact set of operational signals rather than to arbitrary noise."
            )
    if shap_bar_path and Path(shap_bar_path).exists():
        with shap_right:
            render_static_image(shap_bar_path, caption=f"Mean absolute SHAP values — {shap_model_name}", width="stretch")
            st.caption(
                "The bar plot removes direction and ranks variables by average absolute impact. "
                "It is useful for a manager because it summarizes where intervention effort is most likely to change model output."
            )
            render_takeaway_box(
                "The highest-ranked features define the inspection profile an owner should review first when deciding where to intervene before the next county visit."
            )

    shap_csv = clean_text(shap_meta.get("mean_abs_shap_csv", ""))
    if shap_csv and Path(shap_csv).exists():
        shap_df = load_csv_safe(shap_csv)
        st.dataframe(shap_df.head(15), use_container_width=True, hide_index=True)

    render_subsection_label(
        "Interactive what-if prediction",
        "The workflow is intentionally linear: choose a model, load a restaurant template, edit controllable inputs, then request the local SHAP waterfall only when needed.",
    )
    model_names = list(available_prediction_models.keys())
    default_predict_model = (
        manifest.get("best_model_name") if manifest.get("best_model_name") in model_names else model_names[0]
    )
    top_left, top_right = st.columns([1.2, 1.8])
    with top_left:
        selected_predict_model = st.selectbox(
            "Prediction model",
            model_names,
            index=model_names.index(default_predict_model),
            key="predict_model_select",
        )
    with top_right:
        predict_options = {
            f"{row.display_name} | {row.full_address_clean} | {row.city_display} ({row.business_id})": row.business_id
            for row in summary_df.sort_values(by=["display_name", "city_display"], ascending=[True, True]).itertuples()
        }
        if not predict_options:
            st.info("No restaurants available for prediction.")
            return
        selected_restaurant_label = st.selectbox(
            "Restaurant template (loads latest inspection values)",
            list(predict_options.keys()),
            index=0,
            key="predict_restaurant_select",
        )
    selected_business_id = predict_options[selected_restaurant_label]

    global_defaults = manifest.get("interactive_defaults", {})
    if st.session_state.get("predict_prev_business_id") != selected_business_id:
        latest_defaults = build_predict_defaults_for_business(events_df, selected_business_id, global_defaults)
        for feature_name in MODEL_ALL_FEATURES:
            if feature_name in latest_defaults:
                value = latest_defaults[feature_name]
                if feature_name == "grade_num":
                    value = int(min(max(round(float(value)), 1), 4))
                elif feature_name == "is_high_risk":
                    value = int(round(float(value)))
                st.session_state[f"predict_input_{feature_name}"] = value
        st.session_state["predict_prev_business_id"] = selected_business_id

    with st.expander("Edit the custom prediction inputs", expanded=True):
        n1, n2, n3 = st.columns(3)
        n1.number_input(
            "Inspection score",
            min_value=0.0,
            max_value=200.0,
            step=1.0,
            key="predict_input_inspection_score",
        )
        n2.number_input(
            "Red points",
            min_value=0.0,
            max_value=200.0,
            step=1.0,
            key="predict_input_red_points_total",
        )
        n3.number_input(
            "Blue points",
            min_value=0.0,
            max_value=200.0,
            step=1.0,
            key="predict_input_blue_points_total",
        )

        n4, n5, n6 = st.columns(3)
        n4.number_input(
            "Violation count",
            min_value=0.0,
            max_value=200.0,
            step=1.0,
            key="predict_input_violation_count_total",
        )
        n5.selectbox("Current official grade", options=[1, 2, 3, 4], key="predict_input_grade_num")
        n6.selectbox("Current high-risk flag", options=[0, 1], key="predict_input_is_high_risk")

        cat_options = global_defaults.get("categorical_options", {})
        c1, c2, c3 = st.columns(3)
        for col, label, col_ui in [
            ("inspection_type", "Inspection type", c1),
            ("inspection_result", "Inspection result", c2),
            ("city_canonical", "City", c3),
        ]:
            key = f"predict_input_{col}"
            options = [clean_text(v) for v in (cat_options.get(col, []) or []) if clean_text(v)]
            current = clean_text(st.session_state.get(key, ""))
            if current and current not in options:
                options = [current] + options
            if not options:
                options = [""]
            if key not in st.session_state or clean_text(st.session_state.get(key, "")) not in options:
                st.session_state[key] = options[0]
            if col == "city_canonical":
                col_ui.selectbox(label, options=options, key=key, format_func=format_city_name)
            else:
                col_ui.selectbox(label, options=options, key=key)

    predict_row = {
        "inspection_score": float(st.session_state.get("predict_input_inspection_score", 0.0)),
        "red_points_total": float(st.session_state.get("predict_input_red_points_total", 0.0)),
        "blue_points_total": float(st.session_state.get("predict_input_blue_points_total", 0.0)),
        "violation_count_total": float(st.session_state.get("predict_input_violation_count_total", 0.0)),
        "grade_num": float(st.session_state.get("predict_input_grade_num", 1)),
        "is_high_risk": int(st.session_state.get("predict_input_is_high_risk", 0)),
        "inspection_type": clean_text(st.session_state.get("predict_input_inspection_type", "")),
        "inspection_result": clean_text(st.session_state.get("predict_input_inspection_result", "")),
        "city_canonical": clean_text(st.session_state.get("predict_input_city_canonical", "")),
    }
    predict_df = pd.DataFrame([predict_row])[MODEL_ALL_FEATURES]

    model_info = available_prediction_models[selected_predict_model]
    try:
        risk_prob = predict_probability_with_manifest_model(model_info, predict_df)
        risk_band = probability_to_band(risk_prob)
        predicted_class = "Predicted high risk" if risk_prob >= 0.5 else "Predicted not high risk"
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Predicted probability", f"{risk_prob:.1%}")
        p2.metric("Predicted class", predicted_class)
        p3.metric("Risk band", risk_band)
        p4.metric("SHAP explanation model", shap_model_name or "-")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    render_takeaway_box(
        "Use the probability as a prioritization signal. If the score is elevated, the local SHAP waterfall below shows which parts of the current inspection profile are pushing the risk upward for this specific input."
    )

    st.markdown("**Custom-input SHAP waterfall**")
    if st.button("Generate local SHAP waterfall", key="predict_generate_waterfall", type="secondary"):
        st.session_state["predict_waterfall_enabled"] = True
    if st.session_state.get("predict_waterfall_enabled", False):
        try:
            maybe_render_shap_waterfall(manifest, shap_model_name or selected_predict_model, predict_df)
            st.caption(
                "Positive SHAP contributions move the predicted probability upward, while negative contributions pull it downward. "
                "For a restaurant owner, this makes the prediction operational because it translates a model score into the specific conditions that most need attention."
            )
        except Exception as exc:
            st.warning(f"Unable to render SHAP waterfall: {exc}")
    else:
        st.caption(
            "The waterfall explanation is generated on demand. This keeps the cloud app responsive while preserving the required local explanation workflow."
        )


def render_main_navigation() -> str:
    nav_labels = dict(PUBLIC_NAV_ITEMS + ASSIGNMENT_NAV_ITEMS)
    default_nav = PUBLIC_NAV_ITEMS[0][0]
    current_nav = clean_text(st.session_state.get("main_nav_section", default_nav))
    if current_nav not in nav_labels:
        current_nav = default_nav
        st.session_state["main_nav_section"] = current_nav

    st.markdown(
        (
            "<div class='nav-shell'>"
            "<div class='nav-shell-title'>Navigation</div>"
            "<div class='nav-shell-note'>Public release on the left, assignment workflow on the right.</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    public_col, assignment_col = st.columns([1.0, 1.45], gap="large")

    def render_group(container: Any, title: str, title_class: str, items: List[Tuple[str, str]]) -> None:
        container.markdown(
            f"<div class='nav-group-label {title_class}'>{escape(title)}</div>",
            unsafe_allow_html=True,
        )
        button_weights = [max(1.0, min(1.45, len(nav_label) / 18.0)) for _, nav_label in items]
        button_cols = container.columns(button_weights, gap="small")
        for col, (nav_key, nav_label) in zip(button_cols, items):
            button_type = "primary" if current_nav == nav_key else "secondary"
            if col.button(nav_label, key=f"main_nav_{nav_key}", use_container_width=True, type=button_type):
                if st.session_state.get("main_nav_section") != nav_key:
                    st.session_state["main_nav_section"] = nav_key
                    st.rerun()

    render_group(public_col, "Public Release", "nav-group-public-label", PUBLIC_NAV_ITEMS)
    render_group(assignment_col, "Assignment Workflow", "nav-group-assignment-label", ASSIGNMENT_NAV_ITEMS)
    return clean_text(st.session_state.get("main_nav_section", current_nav)) or default_nav


def render_panel_with_guard(panel_label: str, render_func: Any, *args: Any) -> None:
    try:
        render_func(*args)
    except KeyError as exc:
        st.error(f"{panel_label} failed to render because a required field is missing: {exc}")
    except Exception as exc:
        st.error(f"{panel_label} failed to render: {exc}")


def main() -> None:
    if st is None:
        raise RuntimeError(
            "streamlit is required. Install dependencies with: pip install -r requirements.txt"
        )

    st.set_page_config(page_title="King County Restaurant Safety Dashboard", layout="wide")
    apply_global_styles()
    root = app_root()

    st.session_state["locale"] = "en"

    render_section_header(
        "King County Restaurant Safety Dashboard",
        "Assignment-focused workflow for executive summary, descriptive analytics, model performance, and explainability.",
        "MSIS 522 HW1"
    )

    try:
        events_df, violations_df, payload = load_data(str(root))
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    summary_df = load_prepared_business_summary(str(root))
    if summary_df.empty:
        summary_df = build_business_summary(events_df)
    if summary_df.empty:
        st.warning("No restaurant data available.")
        st.stop()
    tab_labels = [label for _, label in ASSIGNMENT_NAV_ITEMS]
    executive_tab, descriptive_tab, performance_tab, explainability_tab = st.tabs(tab_labels)

    with executive_tab:
        render_panel_with_guard(
            "Executive Summary",
            build_executive_summary_tab,
            events_df,
            violations_df,
            summary_df,
            root,
        )
    with descriptive_tab:
        render_panel_with_guard("Descriptive Analytics", build_descriptive_analytics_tab, events_df)
    with performance_tab:
        render_panel_with_guard("Model Performance", build_model_performance_tab, root)
    with explainability_tab:
        render_panel_with_guard(
            "Explainability & Interactive Prediction",
            build_explainability_prediction_tab,
            events_df,
            summary_df,
            root,
        )


if __name__ == "__main__":
    main()
