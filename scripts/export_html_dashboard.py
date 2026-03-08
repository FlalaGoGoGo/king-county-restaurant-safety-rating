#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

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
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False

DATASET_ID = "f29f-zza5"
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
RATING_NOT_AVAILABLE_LABEL = "Rating not available"
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
HOMEWORK_FIELD_GROUPS: List[Tuple[str, List[List[str]]]] = [
    (
        "Identifiers",
        [
            ["inspection_event_id", "Unique event-level identifier for one inspection visit."],
            ["inspection_serial_num", "County-published serial number for the inspection record."],
            ["business_id", "Stable establishment identifier used to link repeated inspections."],
            ["row_id", "Unique row identifier in the violation-level table."],
        ],
    ),
    (
        "Business / Location",
        [
            ["business_name_official", "Official restaurant name in the published source."],
            ["business_name_alt", "Alternate or legacy business name when available."],
            ["search_name_norm", "Normalized text field used to improve search matching."],
            ["full_address_clean", "Cleaned full street address for mapping and search."],
            ["city_canonical", "Standardized city/locality label after cleaning rules."],
            ["zip_code", "ZIP code associated with the establishment address."],
            ["latitude", "Latitude used for map display."],
            ["longitude", "Longitude used for map display."],
        ],
    ),
    (
        "Inspection Outcome",
        [
            ["inspection_date", "Date of the inspection event."],
            ["inspection_type", "Inspection category such as routine, return, or consultation."],
            ["inspection_result", "Published inspection outcome or status."],
            ["inspection_score", "Numerical score recorded for that inspection."],
            ["inspection_closed_business", "Flag indicating whether the business was closed during inspection."],
            ["source_row_count", "Number of raw rows consolidated into the event-level record."],
            ["generated_from_missing_serial", "Flag showing whether an event id was reconstructed from missing source metadata."],
        ],
    ),
    (
        "Rating / Risk",
        [
            ["grade", "Published county grade code."],
            ["grade_label", "Human-readable label for the published grade."],
            ["rating_not_available", "Flag indicating that no public rating was available."],
            ["red_points_total", "Total red-point severity recorded on the inspection."],
            ["blue_points_total", "Total blue-point severity recorded on the inspection."],
            ["red_violation_count", "Count of red violations on the inspection."],
            ["blue_violation_count", "Count of blue violations on the inspection."],
            ["violation_count_total", "Total number of violations on the inspection."],
        ],
    ),
    (
        "Violation / Remediation",
        [
            ["violation_type", "Red or blue violation category."],
            ["violation_code", "County violation code linked to the finding."],
            ["violation_points", "Points assigned to the specific violation."],
            ["violation_desc_clean", "Cleaned violation description for analysis."],
            ["violation_desc_raw", "Original raw violation description from source data."],
            ["dictionary_default_points_mode", "Dictionary rule describing how default points were interpreted."],
            ["dictionary_canonical_description", "Canonical description from the remediation dictionary."],
            ["action_category", "Grouped remediation theme for the violation."],
            ["action_priority", "Priority label for the recommended corrective action."],
            ["action_summary_zh", "Chinese remediation summary stored in the dictionary."],
            ["action_steps_zh", "Chinese action steps stored in the dictionary."],
            ["action_summary_en", "English remediation summary used in the dashboard."],
            ["safe_food_handling_refs", "Reference links or citations for safe-food-handling guidance."],
            ["action_source", "Source used to derive the remediation guidance."],
        ],
    ),
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def format_city_name(value: Any) -> str:
    city = clean_text(value)
    if not city:
        return ""
    if city == OUTSIDE_KING_COUNTY_LABEL:
        return "Outside King County"
    if city == city.upper() or city == city.lower():
        return city.title()
    return city


def clean_city_token(value: Any) -> str:
    city = clean_text(value).upper().rstrip(",")
    city = re.sub(r"\bWA\b$", "", city).strip()
    city = re.sub(r"\s+", " ", city)
    if city in CITY_DIRECT_CORRECTIONS:
        return CITY_DIRECT_CORRECTIONS[city]
    return city


def rating_label_from_values(grade_code: Any, grade_label: Any) -> str:
    code = clean_text(grade_code)
    if code in GRADE_CODE_TO_LABEL:
        return GRADE_CODE_TO_LABEL[code]
    return clean_text(grade_label)


def parse_risk_level_from_description(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    match = RISK_LEVEL_RE.search(text)
    if not match:
        return ""
    return RISK_ROMAN_TO_LEVEL.get(match.group(1).upper(), "")


def format_risk_level_label(value: Any) -> str:
    return RISK_LEVEL_LABELS.get(clean_text(value), "")


def format_rating_source_label(value: Any) -> str:
    mapping = {
        "dashboard_recent_routine_average": "Dashboard calculation: recent routine-inspection average",
        "dashboard_recent_closure_or_multiple_returns": (
            "Dashboard calculation: recent closure or multiple return inspections"
        ),
        "rating_not_available": RATING_NOT_AVAILABLE_LABEL,
    }
    return mapping.get(clean_text(value), clean_text(value))


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


def load_king_county_boundary_line_coords(root: Path) -> Dict[str, List[Optional[float]]]:
    boundary_path = root / "data" / "reference" / "king_county_boundary.geojson"
    if not boundary_path.exists():
        return {"lon": [], "lat": []}

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

    return {"lon": lon_values, "lat": lat_values}


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


def build_zip_to_locality_lookup(df: pd.DataFrame) -> Dict[str, str]:
    if df.empty or "zip_code" not in df.columns or "city_canonical" not in df.columns:
        return {}
    base = df.copy()
    base["zip_code"] = base["zip_code"].astype(str).str.strip()
    base["city_base"] = base["city_canonical"].map(clean_city_token)
    valid = base[base["zip_code"].ne("") & base["city_base"].isin(KING_COUNTY_LOCALITY_SET)].copy()
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
    out["city_display"] = out["city_canonical"].map(format_city_name)
    return out


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


def load_latest_payload(root: Path) -> Dict[str, Any]:
    state_path = root / "Data" / "state" / f"{DATASET_ID}_latest_run.json"
    if not state_path.exists():
        raise FileNotFoundError(f"latest run state not found: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def resolve_paths(root: Path, payload: Dict[str, Any]) -> Tuple[Path, Path]:
    run_id = clean_text(payload.get("run_id", ""))
    silver_event_csv = Path(clean_text(payload.get("silver_event_csv", "")))
    if not silver_event_csv.exists():
        silver_event_csv = root / "Data" / "silver" / DATASET_ID / run_id / "inspection_event.csv"
    violation_csv = Path(clean_text(payload.get("dashboard_violation_explained_csv", "")))
    if not violation_csv.exists():
        violation_csv = root / "Data" / "gold" / DATASET_ID / run_id / "dashboard_violation_explained.csv"
    if not silver_event_csv.exists():
        raise FileNotFoundError(f"silver event csv missing: {silver_event_csv}")
    if not violation_csv.exists():
        raise FileNotFoundError(f"dashboard violation csv missing: {violation_csv}")
    return silver_event_csv, violation_csv


def resolve_bronze_raw_csv_path(root: Path, payload: Dict[str, Any]) -> Path:
    run_id = clean_text(payload.get("run_id", ""))
    return root / "Data" / "bronze" / DATASET_ID / run_id / "raw.csv"


def load_risk_description_lookups(root: Path, payload: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_csv = resolve_bronze_raw_csv_path(root, payload)
    if not raw_csv.exists():
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
    events_df: pd.DataFrame, root: Path, payload: Dict[str, Any]
) -> pd.DataFrame:
    out = events_df.copy()
    if "risk_description_raw" not in out.columns:
        out["risk_description_raw"] = ""
    if "risk_level" not in out.columns:
        out["risk_level"] = ""

    serial_lookup, group_lookup = load_risk_description_lookups(root, payload)
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

            recalculated_num = (
                {"Excellent": "1", "Good": "2", "Okay": "3", "Needs to Improve": "4"}.get(calculated_label, "")
                if calculated_label
                else ""
            )
            official_label = rating_label_from_values(
                getattr(row, "grade", ""), getattr(row, "grade_label", "")
            )
            official_num = clean_text(getattr(row, "grade", ""))
            if official_num not in GRADE_CODE_TO_LABEL:
                official_num = {"Excellent": "1", "Good": "2", "Okay": "3", "Needs to Improve": "4"}.get(
                    official_label, ""
                )

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


def prepare_events_df(events_df: pd.DataFrame, root: Path, payload: Dict[str, Any]) -> pd.DataFrame:
    events_df = events_df.fillna("").copy()
    if "risk_description_raw" not in events_df.columns:
        events_df["risk_description_raw"] = ""
    if "risk_level" not in events_df.columns:
        events_df["risk_level"] = ""
    if "city_raw_original" not in events_df.columns:
        events_df["city_raw_original"] = ""
    if "city_scope" not in events_df.columns:
        events_df["city_scope"] = ""
    if "city_cleaning_reason" not in events_df.columns:
        events_df["city_cleaning_reason"] = ""
    for col in ["inspection_score", "red_points_total", "blue_points_total", "violation_count_total"]:
        events_df[col] = pd.to_numeric(events_df.get(col, ""), errors="coerce")
    city_zip_lookup = build_zip_to_locality_lookup(events_df)
    events_df = apply_city_quality_rules(events_df, city_zip_lookup)
    events_df["inspection_date_dt"] = pd.to_datetime(events_df.get("inspection_date", ""), errors="coerce")
    events_df["grade_num"] = pd.to_numeric(events_df.get("grade", ""), errors="coerce")
    events_df = attach_risk_metadata(events_df, root, payload)
    events_df = append_effective_rating_columns(events_df)
    events_df["latest_rating"] = events_df["effective_rating_label"].astype(str).str.strip()
    events_df["latest_rating"] = events_df["latest_rating"].replace("", RATING_NOT_AVAILABLE_LABEL)
    grade_high_risk = events_df.get("grade", "").astype(str).str.strip().isin(HIGH_RISK_GRADES)
    red_points_high_risk = events_df["red_points_total"].fillna(0) >= HIGH_RISK_RED_POINTS_THRESHOLD
    events_df["is_high_risk"] = (grade_high_risk | red_points_high_risk).astype(int)
    return events_df


def prepare_violations_df(violations_df: pd.DataFrame, zip_lookup: Dict[str, str]) -> pd.DataFrame:
    violations_df = violations_df.fillna("").copy()
    if "city_raw_original" not in violations_df.columns:
        violations_df["city_raw_original"] = ""
    if "city_scope" not in violations_df.columns:
        violations_df["city_scope"] = ""
    if "city_cleaning_reason" not in violations_df.columns:
        violations_df["city_cleaning_reason"] = ""
    violations_df = apply_city_quality_rules(violations_df, zip_lookup)
    violations_df["inspection_date_dt"] = pd.to_datetime(
        violations_df.get("inspection_date", ""), errors="coerce"
    )
    violations_df["violation_points_num"] = pd.to_numeric(
        violations_df.get("violation_points", ""), errors="coerce"
    )
    if "action_summary_en" not in violations_df.columns:
        violations_df["action_summary_en"] = ""
    if "action_summary_zh" not in violations_df.columns:
        violations_df["action_summary_zh"] = ""
    violations_df["action_summary_en"] = violations_df["action_summary_en"].where(
        violations_df["action_summary_en"].astype(str).str.strip() != "",
        violations_df["action_summary_zh"],
    )
    return violations_df


def build_business_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    ordered = events_df.sort_values("inspection_date_dt")
    latest = ordered.drop_duplicates(subset=["business_id"], keep="last").copy()
    counts = events_df.groupby("business_id", dropna=False).size().rename("inspection_count")
    summary = latest.merge(counts, left_on="business_id", right_index=True, how="left")
    summary["inspection_count"] = summary["inspection_count"].fillna(0).astype(int)
    summary["display_name"] = summary["business_name_official"].where(
        summary["business_name_official"].astype(str).str.strip() != "",
        summary["business_name_alt"],
    )
    summary["display_name"] = summary["display_name"].where(
        summary["display_name"].astype(str).str.strip() != "",
        summary["business_id"],
    )
    summary["city_display"] = summary["city_canonical"].map(format_city_name)
    summary["latest_rating"] = summary["effective_rating_label"].astype(str).str.strip().replace(
        "", RATING_NOT_AVAILABLE_LABEL
    )
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
    return summary


def build_monthly_trend(events_df: pd.DataFrame) -> pd.DataFrame:
    trend = events_df.dropna(subset=["inspection_date_dt"]).copy()
    if trend.empty:
        return pd.DataFrame()
    trend["month"] = trend["inspection_date_dt"].dt.to_period("M").astype(str)
    out = (
        trend.groupby("month", dropna=False)
        .agg(
            inspections=("business_id", "size"),
            high_risk_inspections=("is_high_risk", "sum"),
            avg_score=("inspection_score", "mean"),
        )
        .reset_index()
    )
    out["high_risk_rate"] = out["high_risk_inspections"] / out["inspections"]
    out["avg_score"] = out["avg_score"].round(2)
    out["high_risk_rate"] = out["high_risk_rate"].round(4)
    return out.sort_values("month")


def build_rating_changes_map(events_df: pd.DataFrame, source: str) -> Dict[str, List[List[Any]]]:
    grade_df = events_df.copy()
    grade_df = grade_df[grade_df["inspection_date_dt"].notna()].copy()
    if grade_df.empty:
        return {}

    if source == "official_grade":
        grade_df["rating_num"] = pd.to_numeric(grade_df["effective_rating_num"], errors="coerce")
    else:
        grade_df["rating_label"] = grade_df["inspection_score"].map(score_to_rating_band)
        grade_df["rating_num"] = pd.to_numeric(
            grade_df["rating_label"].map({v: k for k, v in GRADE_CODE_TO_LABEL.items()}),
            errors="coerce",
        )

    grade_df = grade_df[grade_df["rating_num"].isin([1, 2, 3, 4])].copy()
    if grade_df.empty:
        return {}

    grade_df = grade_df.sort_values(["business_id", "inspection_date_dt"])
    grade_df["prev_rating_num"] = grade_df.groupby("business_id")["rating_num"].shift(1)
    grade_df["prev_date"] = grade_df.groupby("business_id")["inspection_date_dt"].shift(1)
    grade_df["city_display"] = grade_df["city_canonical"].map(format_city_name)
    grade_df["display_name"] = grade_df["business_name_official"].where(
        grade_df["business_name_official"].astype(str).str.strip() != "",
        grade_df["business_name_alt"],
    )
    grade_df["display_name"] = grade_df["display_name"].where(
        grade_df["display_name"].astype(str).str.strip() != "",
        grade_df["business_id"],
    )

    grade_df["month"] = grade_df["inspection_date_dt"].dt.to_period("M").astype(str)
    latest_in_month = grade_df.sort_values("inspection_date_dt").drop_duplicates(
        subset=["business_id", "month"], keep="last"
    )

    changed = latest_in_month[
        latest_in_month["prev_rating_num"].notna()
        & (latest_in_month["rating_num"] != latest_in_month["prev_rating_num"])
    ].copy()
    if changed.empty:
        return {}

    changed["from_rating"] = (
        changed["prev_rating_num"].astype(int).astype(str).map(GRADE_CODE_TO_LABEL)
    )
    changed["to_rating"] = (
        changed["rating_num"].astype(int).astype(str).map(GRADE_CODE_TO_LABEL)
    )
    changed["change_type"] = changed.apply(
        lambda row: "Improved" if row["rating_num"] < row["prev_rating_num"] else "Declined",
        axis=1,
    )
    changed["transition"] = changed["from_rating"] + " -> " + changed["to_rating"]
    changed["prev_date"] = changed["prev_date"].dt.strftime("%Y-%m-%d")
    changed["inspection_date"] = changed["inspection_date_dt"].dt.strftime("%Y-%m-%d")

    cols = [
        "display_name",
        "city_display",
        "full_address_clean",
        "prev_date",
        "inspection_date",
        "transition",
        "change_type",
    ]
    out: Dict[str, List[List[Any]]] = {}
    for month_key, group in changed.groupby("month", dropna=False):
        out[str(month_key)] = group[cols].values.tolist()
    return out


def build_movement_by_month(events_df: pd.DataFrame) -> Dict[str, Dict[str, List[List[Any]]]]:
    valid = events_df.dropna(subset=["inspection_date_dt"]).copy()
    if valid.empty:
        return {}
    all_months = sorted(valid["inspection_date_dt"].dt.to_period("M").astype(str).unique().tolist())
    official_map = build_rating_changes_map(valid, source="official_grade")
    proxy_map = build_rating_changes_map(valid, source="score_band")
    out: Dict[str, Dict[str, List[List[Any]]]] = {}
    for month_key in all_months:
        out[month_key] = {
            "official": official_map.get(month_key, []),
            "proxy": proxy_map.get(month_key, []),
        }
    return out


def build_consumer_top_high_risk_rows(events_df: pd.DataFrame, top_n: int = 20) -> List[List[Any]]:
    valid = events_df[events_df["inspection_date_dt"].notna()].copy()
    if valid.empty:
        return []

    latest = valid.sort_values("inspection_date_dt").drop_duplicates(
        subset=["business_id"], keep="last"
    )
    latest["display_name"] = latest["business_name_official"].where(
        latest["business_name_official"].astype(str).str.strip() != "",
        latest["business_name_alt"],
    )
    latest["display_name"] = latest["display_name"].where(
        latest["display_name"].astype(str).str.strip() != "",
        latest["business_id"],
    )
    latest["latest_rating"] = latest["effective_rating_label"].astype(str).str.strip().replace(
        "", RATING_NOT_AVAILABLE_LABEL
    )

    high_latest = latest[latest["is_high_risk"] == 1].copy()
    high_latest = high_latest.sort_values(
        by=["red_points_total", "inspection_date_dt"],
        ascending=[False, False],
        na_position="last",
    )

    rows: List[List[Any]] = []
    for r in high_latest.head(top_n).itertuples():
        score_val = getattr(r, "inspection_score", None)
        red_val = getattr(r, "red_points_total", None)
        rows.append(
            [
                clean_text(getattr(r, "display_name", "")),
                clean_text(getattr(r, "city_display", "")),
                clean_text(getattr(r, "full_address_clean", "")),
                clean_text(getattr(r, "inspection_date", "")),
                clean_text(getattr(r, "latest_rating", "")),
                "" if pd.isna(score_val) else round(float(score_val), 1),
                "" if pd.isna(red_val) else int(float(red_val)),
            ]
        )
    return rows


def build_owner_view_payload(violations_df: pd.DataFrame) -> Dict[str, Any]:
    code_rows = violations_df[
        violations_df["violation_code"].astype(str).str.strip().ne("")
    ].copy()
    if code_rows.empty:
        return {
            "ranking_rows": [],
            "top_code_rows": [],
            "priority_rows": [],
        }

    group_cols = [
        "violation_code",
        "violation_type",
        "action_category",
        "action_priority",
        "action_summary_en",
    ]

    ranking = (
        code_rows.groupby(group_cols, dropna=False)
        .agg(
            occurrences=("row_id", "count"),
            affected_restaurants=("business_id", "nunique"),
            avg_points=("violation_points_num", "mean"),
        )
        .reset_index()
    )
    ranking["avg_points"] = ranking["avg_points"].round(2)
    ranking["code_type"] = ranking["violation_code"] + "-" + ranking["violation_type"]

    priority_rank = {"high": 0, "medium": 1, "low": 2}
    ranking["priority_rank"] = ranking["action_priority"].map(priority_rank).fillna(3)
    ranking = ranking.sort_values(
        by=["occurrences", "priority_rank", "affected_restaurants"],
        ascending=[False, True, False],
    )

    ranking_rows: List[List[Any]] = []
    for r in ranking.head(25).itertuples():
        ranking_rows.append(
            [
                clean_text(getattr(r, "violation_code", "")),
                clean_text(getattr(r, "violation_type", "")),
                int(getattr(r, "occurrences", 0)),
                int(getattr(r, "affected_restaurants", 0)),
                "" if pd.isna(getattr(r, "avg_points", None)) else float(getattr(r, "avg_points", 0)),
                clean_text(getattr(r, "action_priority", "")),
                clean_text(getattr(r, "action_category", "")),
                clean_text(getattr(r, "action_summary_en", "")),
            ]
        )

    top_code_rows: List[List[Any]] = []
    for r in ranking[["code_type", "occurrences"]].head(10).itertuples(index=False):
        top_code_rows.append([clean_text(r.code_type), int(r.occurrences)])

    priority_dist = (
        ranking.groupby("action_priority", dropna=False)["occurrences"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    priority_rows: List[List[Any]] = []
    for r in priority_dist.itertuples(index=False):
        priority_rows.append([clean_text(r.action_priority), int(r.occurrences)])

    return {
        "ranking_rows": ranking_rows,
        "top_code_rows": top_code_rows,
        "priority_rows": priority_rows,
    }


def build_regulator_view_payload(
    events_df: pd.DataFrame, min_inspections: int = 30
) -> Dict[str, Any]:
    city_events = events_df[
        events_df["city_display"].astype(str).str.strip().ne("")
    ].copy()
    if city_events.empty:
        return {
            "min_inspections": min_inspections,
            "city_rows": [],
            "top_city_rows": [],
            "monthly_rows": [],
        }

    city_agg = (
        city_events.groupby("city_display", dropna=False)
        .agg(
            inspection_count=("business_id", "size"),
            restaurant_count=("business_id", "nunique"),
            high_risk_count=("is_high_risk", "sum"),
            high_risk_rate=("is_high_risk", "mean"),
            avg_red_points=("red_points_total", "mean"),
        )
        .reset_index()
    )
    city_agg["high_risk_rate"] = city_agg["high_risk_rate"].round(4)
    city_agg["avg_red_points"] = city_agg["avg_red_points"].round(2)

    city_view = city_agg[city_agg["inspection_count"] >= min_inspections].copy()
    city_view = city_view.sort_values(
        by=["high_risk_rate", "inspection_count"],
        ascending=[False, False],
    )

    city_rows: List[List[Any]] = []
    for r in city_view.itertuples(index=False):
        city_rows.append(
            [
                clean_text(r.city_display),
                int(r.inspection_count),
                int(r.restaurant_count),
                int(r.high_risk_count),
                float(r.high_risk_rate),
                float(r.avg_red_points),
            ]
        )

    top_city_rows: List[List[Any]] = []
    for r in city_view[["city_display", "high_risk_rate"]].head(15).itertuples(index=False):
        top_city_rows.append([clean_text(r.city_display), float(r.high_risk_rate)])

    month_agg = city_events.dropna(subset=["inspection_date_dt"]).copy()
    monthly_rows: List[List[Any]] = []
    if not month_agg.empty:
        month_agg["month"] = month_agg["inspection_date_dt"].dt.to_period("M").astype(str)
        month_view = (
            month_agg.groupby("month", dropna=False)
            .agg(
                inspections=("business_id", "size"),
                high_risk_inspections=("is_high_risk", "sum"),
            )
            .reset_index()
            .sort_values("month")
        )
        for r in month_view.itertuples(index=False):
            monthly_rows.append([clean_text(r.month), int(r.inspections), int(r.high_risk_inspections)])

    return {
        "min_inspections": min_inspections,
        "city_rows": city_rows,
        "top_city_rows": top_city_rows,
        "monthly_rows": monthly_rows,
    }


def sanitize_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_json_like(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json_like(v) for v in value]
    if isinstance(value, float):
        if pd.isna(value) or np.isinf(value):
            return ""
    return value


def load_quality_payload(root: Path, run_id: str) -> Dict[str, Any]:
    analysis_dir = root / "outputs" / "analysis"
    run_stamp = clean_text(run_id)[:8]

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

    def safe_int(value: Any) -> Any:
        n = to_float(value)
        if pd.isna(n):
            return ""
        return int(n)

    def safe_num(value: Any) -> Any:
        n = to_float(value)
        if pd.isna(n) or np.isinf(n):
            return ""
        return float(n)

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
    issue_rows: List[List[Any]] = []
    if run_stamp:
        issue_path = analysis_dir / f"king_county_issue_catalog_{run_stamp}.csv"
        if issue_path.exists():
            try:
                issue_df = pd.read_csv(issue_path)
                for col in issue_columns:
                    if col not in issue_df.columns:
                        issue_df[col] = ""
                issue_df = issue_df[issue_columns].copy()
                for row in issue_df.itertuples(index=False):
                    issue_rows.append(
                        [
                            clean_text(getattr(row, "category", "")),
                            clean_text(getattr(row, "issue", "")),
                            safe_int(getattr(row, "count", "")),
                            safe_int(getattr(row, "denominator", "")),
                            safe_num(getattr(row, "share_pct", "")),
                            clean_text(getattr(row, "severity", "")),
                            clean_text(getattr(row, "why_it_matters", "")),
                            clean_text(getattr(row, "suggested_action", "")),
                            clean_text(getattr(row, "owner", "")),
                        ]
                    )
            except Exception:
                issue_rows = []

    history_columns = [
        "run_id",
        "rows_raw",
        "city_outside_rate",
        "city_unknown_rate",
        "date_parse_fail_rate",
        "violation_truncated_rate",
        "true_mismatch_rate",
    ]
    history_rows: List[List[Any]] = []
    for audit_file in sorted(analysis_dir.glob("db_quality_audit_*.json")):
        try:
            snap = json.loads(audit_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        snap_run_id = clean_text(snap.get("run_id", ""))
        rows_raw = to_float(snap.get("rows_raw", np.nan))
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
            [
                snap_run_id or run_stamp_i,
                safe_int(rows_raw),
                safe_num(safe_rate(snap.get("city_known_outside_rows", np.nan), rows_raw)),
                safe_num(safe_rate(snap.get("city_unknown_rows", np.nan), rows_raw)),
                safe_num(safe_rate(snap.get("raw_date_parse_fail_rows", np.nan), rows_raw)),
                safe_num(safe_rate(snap.get("violation_desc_truncated_rows", np.nan), rows_raw)),
                safe_num(true_mismatch_rate),
            ]
        )
    history_rows.sort(key=lambda x: clean_text(x[0]))

    return {
        "run_id": clean_text(run_id),
        "run_stamp": run_stamp,
        "audit_snapshot": audit_snapshot,
        "issue_catalog_columns": issue_columns,
        "issue_catalog_rows": issue_rows,
        "history_columns": history_columns,
        "history_rows": history_rows,
        "samples": samples,
    }


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


if TORCH_AVAILABLE:

    class MLPBinaryClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_layers: Tuple[int, ...], dropout: float) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            prev_dim = input_dim
            for h in hidden_layers:
                layers.append(nn.Linear(prev_dim, int(h)))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(float(dropout)))
                prev_dim = int(h)
            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


def probability_to_band(prob: float) -> str:
    if prob >= 0.5:
        return "High"
    if prob >= 0.25:
        return "Medium"
    return "Low"


def load_predict_manifest_payload(root: Path) -> Dict[str, Any]:
    latest_pointer = root / "models" / "hw1_predict" / "latest_manifest.json"
    default_manifest = root / "models" / "hw1_predict" / "manifest.json"

    manifest_path = default_manifest
    if latest_pointer.exists():
        try:
            pointer = json.loads(latest_pointer.read_text(encoding="utf-8"))
            maybe_path = Path(clean_text(pointer.get("manifest_path", "")))
            if maybe_path.exists():
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
            "manifest_path": str(manifest_path),
            "manifest": {},
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "message": "",
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def predict_probabilities_for_model(model_info: Dict[str, Any], feature_df: pd.DataFrame) -> np.ndarray:
    kind = clean_text(model_info.get("kind", ""))
    model_path = clean_text(model_info.get("model_path", ""))
    if not model_path:
        raise RuntimeError("model_path missing in manifest.")

    if kind == "sklearn_pipeline":
        pipeline = joblib.load(model_path)
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(feature_df)[:, 1]
        else:
            preds = pipeline.predict(feature_df)
            probs = np.asarray(preds, dtype=np.float64)
        return np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)

    if kind == "pytorch_mlp":
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required but not installed.")
        pre_path = clean_text(model_info.get("preprocessor_path", ""))
        if not pre_path:
            raise RuntimeError("preprocessor_path missing for PyTorch MLP.")
        checkpoint = torch.load(model_path, map_location="cpu")
        input_dim = int(checkpoint.get("input_dim", 0))
        if input_dim <= 0:
            raise RuntimeError("Invalid PyTorch checkpoint.")
        hidden_layers = checkpoint.get("hidden_layers", [128, 128])
        hidden_layers = tuple(int(x) for x in hidden_layers)
        dropout = float(checkpoint.get("dropout", 0.0))
        model = MLPBinaryClassifier(input_dim=input_dim, hidden_layers=hidden_layers, dropout=dropout)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        pre = joblib.load(pre_path)
        arr = np.asarray(pre.transform(feature_df), dtype=np.float32)
        with torch.no_grad():
            probs = model(torch.from_numpy(arr)).numpy().reshape(-1)
        return np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)

    raise RuntimeError(f"Unsupported model kind: {kind}")


def build_predict_payload(root: Path, summary_df: pd.DataFrame) -> Dict[str, Any]:
    manifest_bundle = load_predict_manifest_payload(root)
    if not manifest_bundle.get("available", False):
        return {
            "available": False,
            "message": manifest_bundle.get("message", "Predict manifest unavailable."),
        }

    manifest = manifest_bundle["manifest"]
    models = manifest.get("models", {})
    if not models:
        return {
            "available": False,
            "message": "No models found in predict manifest.",
        }

    metrics_columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    metrics_rows: List[List[Any]] = []
    for model_name, model_info in models.items():
        m = model_info.get("metrics", {})
        metrics_rows.append(
            [
                model_name,
                round(float(m.get("Accuracy", 0.0)), 4),
                round(float(m.get("Precision", 0.0)), 4),
                round(float(m.get("Recall", 0.0)), 4),
                round(float(m.get("F1", 0.0)), 4),
                round(float(m.get("ROC_AUC", 0.0)), 4),
            ]
        )
    metrics_rows.sort(key=lambda row: (row[4], row[5]), reverse=True)

    model_details: Dict[str, Dict[str, Any]] = {}
    for model_name, model_info in models.items():
        extra = model_info.get("extra", {}) or {}
        top_feature_rows: List[List[Any]] = []
        top_feature_path = clean_text(extra.get("feature_importance_csv", "")) or clean_text(
            extra.get("coefficients_csv", "")
        )
        if top_feature_path and Path(top_feature_path).exists():
            feat_df = pd.read_csv(top_feature_path)
            if "feature" in feat_df.columns:
                if "importance" in feat_df.columns:
                    value_col = "importance"
                elif "abs_coefficient" in feat_df.columns:
                    value_col = "abs_coefficient"
                elif "coefficient" in feat_df.columns:
                    value_col = "coefficient"
                else:
                    value_col = ""
                if value_col:
                    for row in feat_df.head(15).itertuples(index=False):
                        top_feature_rows.append(
                            [clean_text(getattr(row, "feature", "")), float(getattr(row, value_col))]
                        )

        model_details[model_name] = {
            "kind": clean_text(model_info.get("kind", "")),
            "metrics": model_info.get("metrics", {}),
            "best_params": model_info.get("best_params", {}),
            "roc_plot_path": clean_text(model_info.get("roc_plot_path", "")),
            "history_plot_path": clean_text(extra.get("history_plot_path", "")),
            "tree_plot_path": clean_text(extra.get("tree_plot_path", "")),
            "tuning_results_csv": clean_text(extra.get("tuning_results_csv", "")),
            "tuning_plot_path": clean_text(extra.get("tuning_plot_path", "")),
            "top_features_rows": top_feature_rows,
            "tuning_top_rows": [],
        }

        tuning_csv_path = model_details[model_name]["tuning_results_csv"]
        if tuning_csv_path and Path(tuning_csv_path).exists():
            try:
                tdf = pd.read_csv(tuning_csv_path).head(10)
                tuning_rows: List[List[Any]] = []
                for r in tdf.itertuples(index=False):
                    tuning_rows.append(
                        [
                            clean_text(getattr(r, "hidden_layers", "")),
                            float(getattr(r, "learning_rate", 0.0)),
                            float(getattr(r, "dropout", 0.0)),
                            int(getattr(r, "epochs_trained", 0)),
                            float(getattr(r, "val_f1", 0.0)),
                            float(getattr(r, "val_roc_auc", 0.0)),
                        ]
                    )
                model_details[model_name]["tuning_top_rows"] = tuning_rows
            except Exception:
                model_details[model_name]["tuning_top_rows"] = []

    feature_df = summary_df[MODEL_ALL_FEATURES].copy()
    for col in MODEL_NUMERIC_FEATURES:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(0.0)
    for col in MODEL_CATEGORICAL_FEATURES:
        feature_df[col] = feature_df[col].astype(str).fillna("")
    business_ids = summary_df["business_id"].astype(str).tolist()

    predictions_by_model_business: Dict[str, Dict[str, List[Any]]] = {}
    for model_name, model_info in models.items():
        try:
            probs = predict_probabilities_for_model(model_info, feature_df)
            per_business = {}
            for business_id, prob in zip(business_ids, probs):
                p = float(prob)
                per_business[business_id] = [round(p, 4), probability_to_band(p)]
            predictions_by_model_business[model_name] = per_business
        except Exception:
            predictions_by_model_business[model_name] = {}

    default_inputs_by_business: Dict[str, Dict[str, Any]] = {}
    for row in summary_df.itertuples(index=False):
        business_id = clean_text(getattr(row, "business_id", ""))
        if not business_id:
            continue
        default_inputs_by_business[business_id] = {
            "inspection_score": 0.0 if pd.isna(getattr(row, "inspection_score", None)) else float(getattr(row, "inspection_score", 0.0)),
            "red_points_total": 0.0 if pd.isna(getattr(row, "red_points_total", None)) else float(getattr(row, "red_points_total", 0.0)),
            "blue_points_total": 0.0 if pd.isna(getattr(row, "blue_points_total", None)) else float(getattr(row, "blue_points_total", 0.0)),
            "violation_count_total": 0.0 if pd.isna(getattr(row, "violation_count_total", None)) else float(getattr(row, "violation_count_total", 0.0)),
            "grade_num": 1 if pd.isna(getattr(row, "grade_num", None)) else int(max(1, min(4, round(float(getattr(row, "grade_num", 1)))))),
            "is_high_risk": int(getattr(row, "is_high_risk", 0) or 0),
            "inspection_type": clean_text(getattr(row, "inspection_type", "")),
            "inspection_result": clean_text(getattr(row, "inspection_result", "")),
            "city_canonical": clean_text(getattr(row, "city_canonical", "")),
        }

    return {
        "available": True,
        "message": "",
        "manifest_path": manifest_bundle.get("manifest_path", ""),
        "best_model_name": clean_text(manifest.get("best_model_name", "")),
        "best_tree_model_name": clean_text(manifest.get("best_tree_model_name", "")),
        "train_rows": int(manifest.get("train_rows", 0)),
        "test_rows": int(manifest.get("test_rows", 0)),
        "positive_rate_train": float(manifest.get("positive_rate_train", 0.0)),
        "positive_rate_test": float(manifest.get("positive_rate_test", 0.0)),
        "metrics_columns": metrics_columns,
        "metrics_rows": metrics_rows,
        "model_details": model_details,
        "predictions_by_model_business": predictions_by_model_business,
        "default_inputs_by_business": default_inputs_by_business,
        "shap": manifest.get("shap", {}),
    }


def sample_numeric_series(series: pd.Series, max_n: int = 1400) -> List[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return []
    if len(clean) > max_n:
        clean = clean.sample(n=max_n, random_state=42)
    return [round(float(v), 4) for v in clean.tolist()]


def build_homework_payload(
    root: Path,
    events_df: pd.DataFrame,
    violations_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> Dict[str, Any]:
    manifest_bundle = load_predict_manifest_payload(root)
    manifest = manifest_bundle.get("manifest", {})
    models = manifest.get("models", {}) if manifest_bundle.get("available", False) else {}
    model_dataset = build_next_inspection_dataset(events_df)
    if model_dataset.empty:
        return {
            "available": False,
            "message": "No rows available for homework analytics.",
            "field_groups": [{"title": title, "rows": rows} for title, rows in HOMEWORK_FIELD_GROUPS],
        }

    metrics_rows: List[List[Any]] = []
    for model_name, model_info in models.items():
        m = model_info.get("metrics", {})
        metrics_rows.append(
            [
                model_name,
                round(float(m.get("Accuracy", 0.0)), 4),
                round(float(m.get("Precision", 0.0)), 4),
                round(float(m.get("Recall", 0.0)), 4),
                round(float(m.get("F1", 0.0)), 4),
                round(float(m.get("ROC_AUC", 0.0)), 4),
            ]
        )
    metrics_rows.sort(key=lambda row: (row[4], row[5]), reverse=True)

    target_counts = (
        model_dataset["target_next_high_risk"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    result_rate_df = (
        model_dataset.groupby("inspection_result", dropna=False)
        .agg(high_risk_rate=("target_next_high_risk", "mean"), rows=("target_next_high_risk", "size"))
        .reset_index()
    )
    result_rate_df = result_rate_df[result_rate_df["rows"] >= 100].sort_values(
        "high_risk_rate", ascending=False
    ).head(10)

    city_rate_df = (
        model_dataset.groupby("city_canonical", dropna=False)
        .agg(high_risk_rate=("target_next_high_risk", "mean"), rows=("target_next_high_risk", "size"))
        .reset_index()
    )
    city_rate_df = city_rate_df[city_rate_df["rows"] >= 200].sort_values(
        "high_risk_rate", ascending=False
    ).head(12)

    corr_cols = [
        "inspection_score",
        "red_points_total",
        "blue_points_total",
        "violation_count_total",
        "grade_num",
        "is_high_risk",
        "target_next_high_risk",
    ]
    corr_df = model_dataset[corr_cols].corr(numeric_only=True).round(4)

    valid_dates = model_dataset["inspection_date_dt"].dropna()
    min_date = valid_dates.min().date().isoformat() if not valid_dates.empty else "-"
    max_date = valid_dates.max().date().isoformat() if not valid_dates.empty else "-"

    shap_meta = manifest.get("shap", {}) or {}
    shap_csv_path = clean_text(shap_meta.get("mean_abs_shap_csv", ""))
    shap_rows: List[List[Any]] = []
    if shap_csv_path and Path(shap_csv_path).exists():
        shap_df = pd.read_csv(shap_csv_path).head(15)
        for row in shap_df.itertuples(index=False):
            shap_rows.append(
                [
                    clean_text(getattr(row, "feature", "")),
                    round(float(getattr(row, "mean_abs_shap", 0.0)), 6),
                ]
            )

    best_model_name = clean_text(manifest.get("best_model_name", ""))
    best_tree_model_name = clean_text(shap_meta.get("best_tree_model_name", "")) or clean_text(
        manifest.get("best_tree_model_name", "")
    )
    default_best_row = metrics_rows[0] if metrics_rows else ["", 0.0, 0.0, 0.0, 0.0, 0.0]
    best_row = next(
        (row for row in metrics_rows if clean_text(row[0]) == best_model_name),
        default_best_row,
    )

    return {
        "available": True,
        "message": "",
        "owner_question": HOMEWORK_OWNER_QUESTION,
        "field_groups": [{"title": title, "rows": rows} for title, rows in HOMEWORK_FIELD_GROUPS],
        "stats": {
            "event_rows": int(len(events_df)),
            "violation_rows": int(len(violations_df)),
            "restaurant_count": int(summary_df["business_id"].nunique()),
            "model_rows": int(len(model_dataset)),
            "positive_rate": round(float(model_dataset["target_next_high_risk"].mean()), 4),
            "feature_count": int(len(MODEL_ALL_FEATURES)),
            "numeric_feature_count": int(len(MODEL_NUMERIC_FEATURES)),
            "categorical_feature_count": int(len(MODEL_CATEGORICAL_FEATURES)),
            "min_date": min_date,
            "max_date": max_date,
        },
        "executive": {
            "best_model_name": best_model_name,
            "best_tree_model_name": best_tree_model_name,
            "best_model_row": best_row,
        },
        "descriptive": {
            "target_counts": [
                ["Low-risk next inspection", int(target_counts.loc[0])],
                ["High-risk next inspection", int(target_counts.loc[1])],
            ],
            "inspection_score_samples": {
                "low": sample_numeric_series(model_dataset.loc[model_dataset["target_next_high_risk"] == 0, "inspection_score"]),
                "high": sample_numeric_series(model_dataset.loc[model_dataset["target_next_high_risk"] == 1, "inspection_score"]),
            },
            "red_points_samples": {
                "low": sample_numeric_series(model_dataset.loc[model_dataset["target_next_high_risk"] == 0, "red_points_total"]),
                "high": sample_numeric_series(model_dataset.loc[model_dataset["target_next_high_risk"] == 1, "red_points_total"]),
            },
            "violation_count_samples": {
                "low": sample_numeric_series(model_dataset.loc[model_dataset["target_next_high_risk"] == 0, "violation_count_total"]),
                "high": sample_numeric_series(model_dataset.loc[model_dataset["target_next_high_risk"] == 1, "violation_count_total"]),
            },
            "result_rate_rows": [
                [clean_text(r.inspection_result), round(float(r.high_risk_rate), 4), int(r.rows)]
                for r in result_rate_df.itertuples(index=False)
            ],
            "city_rate_rows": [
                [format_city_name(r.city_canonical), round(float(r.high_risk_rate), 4), int(r.rows)]
                for r in city_rate_df.itertuples(index=False)
            ],
            "corr_columns": corr_cols,
            "corr_values": corr_df.values.tolist(),
        },
        "model_metrics": {
            "columns": ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"],
            "rows": metrics_rows,
        },
        "shap_rows": shap_rows,
    }


def build_overview_payload(
    root: Path,
    events_df: pd.DataFrame,
    violations_df: pd.DataFrame,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    red_count = int(
        violations_df["violation_type"].astype(str).str.upper().str.strip().eq("RED").sum()
    )
    blue_count = int(
        violations_df["violation_type"].astype(str).str.upper().str.strip().eq("BLUE").sum()
    )
    top_codes = (
        violations_df[
            violations_df["violation_code"].astype(str).str.strip().ne("")
            & violations_df["violation_type"].astype(str).str.strip().ne("")
        ]
        .groupby(["violation_code", "violation_type", "violation_desc_clean"], dropna=False)
        .size()
        .reset_index(name="occurrences")
        .sort_values("occurrences", ascending=False)
        .head(12)
    )
    top_code_rows: List[List[Any]] = []
    for row in top_codes.itertuples(index=False):
        top_code_rows.append(
            [
                clean_text(getattr(row, "violation_code", "")),
                clean_text(getattr(row, "violation_type", "")),
                clean_text(getattr(row, "violation_desc_clean", "")),
                int(getattr(row, "occurrences", 0)),
            ]
        )

    poster_catalog: Dict[str, Dict[str, str]] = {}
    for rating, file_name in RATING_POSTER_IMAGE_FILES.items():
        poster_path = root / "images" / file_name
        poster_catalog[rating] = {
            "path": str(poster_path) if poster_path.exists() else "",
            "description": RATING_POSTER_DESCRIPTIONS.get(rating, ""),
        }

    return {
        "run_id": clean_text(payload.get("run_id", "")),
        "generated_at_utc": clean_text(payload.get("generated_at_utc", "")),
        "inspection_rows": int(len(events_df)),
        "restaurant_count": int(events_df["business_id"].nunique()),
        "red_count": red_count,
        "blue_count": blue_count,
        "rating_rows": [
            ["3", "Excellent", "0.00-3.75 average red points over last 4 routine inspections"],
            ["3", "Good", "3.76-16.25 average red points over last 4 routine inspections"],
            ["3", "Okay", "16.26+ average red points over last 4 routine inspections"],
            ["1-2", "Excellent", "0.00 average red points over up to last 2 routine inspections"],
            ["1-2", "Good", "0.01-5.00 average red points over up to last 2 routine inspections"],
            ["1-2", "Okay", "5.01+ average red points over up to last 2 routine inspections"],
            ["Any", "Needs to Improve", "Closed within last 90 days or needed multiple return inspections"],
            ["-", RATING_NOT_AVAILABLE_LABEL, "No routine-based rating is currently available"],
        ],
        "top_code_rows": top_code_rows,
        "poster_catalog": poster_catalog,
    }


def train_models(events_df: pd.DataFrame) -> Dict[str, Any]:
    if not SKLEARN_AVAILABLE:
        return {"available": False, "message": "scikit-learn is not available."}
    model_df = build_next_inspection_dataset(events_df)
    if len(model_df) < 1200 or model_df["target_next_high_risk"].nunique() < 2:
        return {"available": False, "message": "Insufficient data for model training."}

    model_df = model_df.sort_values("inspection_date_dt")
    split_idx = int(len(model_df) * 0.8)
    train_df = model_df.iloc[:split_idx].copy()
    test_df = model_df.iloc[split_idx:].copy()

    if (
        train_df.empty
        or test_df.empty
        or train_df["target_next_high_risk"].nunique() < 2
        or test_df["target_next_high_risk"].nunique() < 2
    ):
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
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
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

    models: Dict[str, Pipeline] = {}
    metric_rows: List[List[Any]] = []
    for model_name, estimator in model_specs.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocess), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metric_rows.append(
            [
                model_name,
                round(float(accuracy_score(y_test, y_pred)), 4),
                round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                round(float(roc_auc_score(y_test, y_proba)), 4),
            ]
        )
        models[model_name] = pipeline

    metric_rows.sort(key=lambda row: (row[5], row[4]), reverse=True)
    best_model_name = metric_rows[0][0]
    best_model = models[best_model_name]

    try:
        pre = best_model.named_steps["preprocessor"]
        model = best_model.named_steps["model"]
        feature_names = pre.get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            coef = model.coef_[0] if getattr(model.coef_, "ndim", 1) > 1 else model.coef_
            importances = abs(coef)
        feat_pairs = sorted(
            zip(feature_names, importances),
            key=lambda item: item[1],
            reverse=True,
        )[:12]
        top_features = [[str(name), round(float(imp), 6)] for name, imp in feat_pairs]
    except Exception:
        top_features = []

    latest_by_business = (
        events_df.sort_values("inspection_date_dt")
        .drop_duplicates(subset=["business_id"], keep="last")
        .copy()
    )
    predict_input = latest_by_business[MODEL_ALL_FEATURES].copy()
    predict_probs = best_model.predict_proba(predict_input)[:, 1]
    prediction_map: Dict[str, List[Any]] = {}
    for business_id, prob in zip(latest_by_business["business_id"].astype(str), predict_probs):
        p = round(float(prob), 4)
        if p >= 0.5:
            band = "High"
        elif p >= 0.25:
            band = "Medium"
        else:
            band = "Low"
        prediction_map[business_id] = [p, band]

    return {
        "available": True,
        "metrics_columns": [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
        ],
        "metrics_rows": metric_rows,
        "best_model_name": best_model_name,
        "top_features_columns": ["Feature", "Importance"],
        "top_features_rows": top_features,
        "predictions_by_business": prediction_map,
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "positive_rate_train": round(float(y_train.mean()), 4),
        "positive_rate_test": round(float(y_test.mean()), 4),
    }


def to_js_safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def build_html(payload: Dict[str, Any]) -> str:
    data_json = to_js_safe_json(payload)
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>King County Restaurant Safety Dashboard (HTML)</title>
  <style>
    :root {
      --bg:#e8f4f5;
      --bg2:#d7ecee;
      --surface:#ffffff;
      --surface-soft:#f5fbfb;
      --ink:#0f2d36;
      --muted:#5d7d85;
      --brand:#0d8a98;
      --brand-strong:#08727f;
      --brand2:#27c2a8;
      --line:#cae0e4;
      --line-strong:#9fc8cf;
      --accent:#00a5c3;
      --good:#2f9e44;
      --warn:#f08c00;
      --bad:#c92a2a;
      --radius-xs:8px;
      --radius-sm:12px;
      --radius-md:18px;
      --radius-lg:24px;
      --shadow-sm:0 2px 8px rgba(15, 45, 54, 0.07);
      --shadow-md:0 10px 30px rgba(15, 45, 54, 0.12);
      --shadow-lg:0 24px 40px rgba(15, 45, 54, 0.14);
    }
    * { box-sizing:border-box; }
    body {
      margin:0;
      font-family: "Avenir Next","IBM Plex Sans","Trebuchet MS","Segoe UI",sans-serif;
      color:var(--ink);
      background:
        radial-gradient(1200px 500px at 8% -5%, rgba(39, 194, 168, 0.26), transparent 60%),
        radial-gradient(900px 500px at 95% 0%, rgba(13, 138, 152, 0.18), transparent 65%),
        linear-gradient(145deg, var(--bg) 0%, var(--bg2) 35%, #eef8f9 100%);
      min-height:100vh;
    }
    .container { max-width:1440px; margin:0 auto; padding:26px 22px 30px; }
    .topbar {
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:12px;
      margin-bottom:12px;
      padding:10px 14px;
      border:1px solid var(--line);
      border-radius:var(--radius-sm);
      background:rgba(255,255,255,0.82);
      backdrop-filter:blur(3px);
      box-shadow:var(--shadow-sm);
    }
    .topbar .hint {
      font-size:12px;
      color:var(--muted);
      letter-spacing:0.01em;
    }
    .locale-wrap {
      display:none;
    }
    .locale-wrap select {
      border:1px solid var(--line);
      border-radius:var(--radius-xs);
      background:#fff;
      color:var(--ink);
      height:34px;
      padding:0 10px;
      font-weight:600;
    }
    .hero {
      background:
        linear-gradient(132deg, rgba(8,114,127,0.95), rgba(13,138,152,0.93) 45%, rgba(39,194,168,0.92)),
        linear-gradient(165deg, rgba(255,255,255,0.08), transparent 60%);
      color:#f5feff;
      border:1px solid rgba(255,255,255,0.2);
      border-radius:var(--radius-lg);
      padding:22px 24px;
      box-shadow:var(--shadow-lg);
      position:relative;
      overflow:hidden;
    }
    .hero::after {
      content:'';
      position:absolute;
      inset:auto -80px -60px auto;
      width:260px;
      height:260px;
      border-radius:999px;
      background:radial-gradient(circle, rgba(255,255,255,0.28) 0%, rgba(255,255,255,0) 70%);
    }
    .hero h1 {
      margin:0;
      font-size:30px;
      line-height:1.08;
      letter-spacing:0.01em;
      position:relative;
      z-index:1;
    }
    .hero p {
      margin:10px 0 0;
      opacity:0.96;
      font-size:14px;
      max-width:840px;
      position:relative;
      z-index:1;
    }
    .tabbar {
      display:flex;
      gap:18px;
      margin:16px 0;
      padding:10px;
      border:1px solid var(--line);
      border-radius:var(--radius-md);
      background:rgba(255,255,255,0.84);
      backdrop-filter:blur(3px);
      align-items:flex-start;
      justify-content:space-between;
    }
    .tabgroup {
      display:flex;
      gap:10px;
      flex-wrap:wrap;
      align-items:flex-start;
    }
    .tabgroup.right {
      justify-content:flex-start;
    }
    .tabbtn {
      border:1px solid var(--line);
      background:#fbfefe;
      padding:10px 16px;
      border-radius:var(--radius-sm);
      cursor:pointer;
      font-weight:700;
      color:var(--ink);
      transition:all 180ms ease;
    }
    .tabbtn:hover { transform:translateY(-1px); border-color:var(--line-strong); box-shadow:var(--shadow-sm); }
    .tabbtn.active {
      background:linear-gradient(135deg, var(--brand-strong), var(--brand));
      color:#fff;
      border-color:var(--brand-strong);
      box-shadow:0 8px 20px rgba(8,114,127,0.28);
    }
    .panel {
      display:none;
      background:var(--surface);
      border:1px solid var(--line);
      border-radius:var(--radius-md);
      padding:22px;
      box-shadow:var(--shadow-md);
    }
    .panel.active { display:block; }
    .hidden { display:none !important; }
    .filters { display:grid; grid-template-columns: 2fr 1fr 1fr 2fr; gap:10px; align-items:end; margin-bottom:12px; }
    .search-filters { grid-template-columns: 2.4fr 0.8fr 0.8fr 3fr; }
    .field label { display:block; font-size:12px; color:var(--muted); margin-bottom:6px; font-weight:600; }
    .field input,.field select {
      width:100%;
      border:1px solid var(--line);
      border-radius:var(--radius-xs);
      padding:9px 10px;
      font-size:14px;
      background:#fff;
      color:var(--ink);
      transition:all 160ms ease;
    }
    .field input:focus,.field select:focus {
      outline:none;
      border-color:var(--accent);
      box-shadow:0 0 0 3px rgba(0,165,195,0.18);
    }
    .search-filters .field label {
      min-height:18px;
      line-height:18px;
      white-space:nowrap;
      margin-bottom:6px;
    }
    .search-filters .field input,
    .search-filters .field select { height:38px; }
    .chips { display:flex; gap:8px; flex-wrap:wrap; }
    .search-filters .chips {
      flex-wrap:nowrap;
      overflow-x:auto;
      scrollbar-width:thin;
      height:38px;
      align-items:stretch;
    }
    .search-filters .chip {
      flex:0 0 auto;
      height:38px;
      display:inline-flex;
      align-items:center;
      padding:0 12px;
    }
    .step-title { margin:14px 0 10px; font-size:16px; font-weight:700; }
    .step-sep { border-top:1px dashed var(--line-strong); margin:22px 0; }
    .mode-buttons { display:flex; gap:8px; flex-wrap:wrap; margin:8px 0 12px; }
    .mode-btn {
      border:1px solid var(--line);
      border-radius:var(--radius-sm);
      padding:10px 14px;
      background:#fff;
      font-size:13px;
      font-weight:700;
      color:var(--ink);
      cursor:pointer;
      transition:all 160ms ease;
    }
    .mode-btn:hover { border-color:var(--line-strong); }
    .mode-btn.active {
      background:rgba(13,138,152,0.12);
      border-color:rgba(13,138,152,0.44);
      color:var(--brand-strong);
    }
    .search-mode-panel { margin-bottom:8px; }
    .map-block {
      margin-top:10px;
      border:1px solid var(--line);
      border-radius:var(--radius-sm);
      background:var(--surface-soft);
      padding:10px;
    }
    .map-canvas {
      width:100%;
      height:420px;
      border:1px solid var(--line);
      border-radius:var(--radius-sm);
      overflow:hidden;
      background:#eef2f7;
    }
    .role-buttons { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; }
    .role-btn {
      border:1px solid var(--line);
      border-radius:999px;
      padding:8px 14px;
      background:#fff;
      font-size:13px;
      font-weight:600;
      color:var(--ink);
      cursor:pointer;
      transition:all 160ms ease;
    }
    .role-btn.active { background:rgba(13,138,152,0.12); border-color:rgba(13,138,152,0.44); color:var(--brand-strong); }
    .chip {
      border:1px solid var(--line);
      border-radius:999px;
      padding:6px 10px;
      background:#fff;
      font-size:12px;
      cursor:pointer;
      transition:all 140ms ease;
    }
    .chip.active { background:rgba(13,138,152,0.12); border-color:rgba(13,138,152,0.44); color:var(--brand-strong); }
    .secondary-btn {
      border:1px solid var(--line);
      border-radius:var(--radius-sm);
      background:#fff;
      color:var(--ink);
      padding:8px 12px;
      font-size:13px;
      font-weight:600;
      cursor:pointer;
      transition:all 160ms ease;
    }
    .secondary-btn:hover { border-color:var(--line-strong); box-shadow:var(--shadow-sm); }
    .meta { margin-top:8px; margin-bottom:12px; font-size:12px; color:var(--muted); }
    .kpis { display:grid; grid-template-columns: repeat(4,1fr); gap:12px; margin:12px 0 14px; }
    .kpi {
      background:linear-gradient(180deg,#ffffff 0%, #f8fcfc 100%);
      border:1px solid var(--line);
      border-radius:var(--radius-sm);
      padding:12px;
      box-shadow:var(--shadow-sm);
    }
    .kpi .k { color:var(--muted); font-size:12px; }
    .kpi .v { font-weight:800; font-size:18px; margin-top:2px; letter-spacing:0.01em; }
    .kpi .note {
      color:var(--muted);
      font-size:12px;
      line-height:1.45;
      margin-top:8px;
    }
    .kpi ul {
      color:var(--muted);
      font-size:12px;
      line-height:1.45;
      margin:8px 0 0 18px;
      padding:0;
    }
    .kpi li { margin:0 0 4px; }
    table { width:100%; border-collapse:collapse; margin-top:8px; }
    th,td { border-bottom:1px solid var(--line); text-align:left; padding:8px 8px; font-size:13px; vertical-align:top; }
    th {
      background:linear-gradient(180deg, #eff7f8 0%, #edf6f7 100%);
      position:sticky;
      top:0;
      z-index:1;
      color:#24454d;
      font-weight:700;
    }
    .table-wrap {
      max-height:360px;
      overflow:auto;
      border:1px solid var(--line);
      border-radius:var(--radius-sm);
      margin-top:6px;
      margin-bottom:14px;
      background:#fff;
      box-shadow:var(--shadow-sm);
    }
    .table-wrap.fixed-320 { height:320px; max-height:320px; }
    .split { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .warn { color:var(--warn); font-weight:700; }
    .small { font-size:12px; color:var(--muted); }
    .i18n-debug {
      margin:8px 0 14px;
      border:1px dashed var(--line-strong);
      border-radius:var(--radius-sm);
      padding:8px 10px;
      background:rgba(255,255,255,0.78);
      display:none;
      white-space:pre-wrap;
    }
    .section-title { margin:18px 0 8px; font-size:16px; font-weight:800; letter-spacing:0.01em; }
    .barrow { display:flex; align-items:center; gap:10px; margin:6px 0; font-size:13px; }
    .bar { height:10px; border-radius:999px; background:#ddecef; flex:1; overflow:hidden; }
    .bar > span { display:block; height:100%; background:linear-gradient(90deg,var(--brand-strong),var(--brand2)); }
    .th-wrap { display:flex; align-items:center; justify-content:space-between; gap:6px; }
    .sort-btn {
      border:1px solid var(--line);
      border-radius:var(--radius-xs);
      background:#fff;
      color:var(--muted);
      cursor:pointer;
      width:22px;
      height:22px;
      font-size:11px;
      line-height:1;
      padding:0;
    }
    .sort-btn.active { background:rgba(13,138,152,0.12); border-color:rgba(13,138,152,0.44); color:var(--brand-strong); }
    tbody tr:hover td { background:#f3fbfc; }
    tr.selected-row td { background:#e6f8fa; font-weight:600; }
    tr.selected-row td:first-child { border-left:3px solid var(--brand-strong); }
    .info-block {
      border:1px dashed var(--line);
      border-radius:var(--radius-sm);
      padding:12px;
      margin-bottom:10px;
      background:var(--surface-soft);
    }
    .poster-box {
      border:1px solid var(--line);
      border-radius:var(--radius-sm);
      padding:10px;
      background:#fff;
      min-height:160px;
      display:flex;
      align-items:center;
      justify-content:center;
    }
    .poster-img {
      max-width:100%;
      max-height:300px;
      width:auto;
      height:auto;
      object-fit:contain;
      border-radius:6px;
    }
    .poster-desc {
      border:1px solid var(--line);
      border-radius:var(--radius-sm);
      padding:12px;
      background:var(--surface-soft);
      min-height:160px;
    }
    .detail-split {
      display:grid;
      grid-template-columns:280px 1fr;
      gap:14px;
      align-items:start;
      margin:8px 0 10px;
    }
    .detail-poster img {
      max-width:100%;
      max-height:340px;
      width:auto;
      height:auto;
      object-fit:contain;
      display:block;
    }
    .detail-kpis { grid-template-columns:1fr 1fr; margin:0 0 8px; }
    .detail-desc { margin-top:2px; }
    .subtabbar { display:flex; gap:8px; margin-bottom:10px; flex-wrap:wrap; }
    .subtabbtn {
      border:1px solid var(--line);
      background:#fff;
      padding:8px 12px;
      border-radius:var(--radius-xs);
      cursor:pointer;
      font-weight:700;
      color:var(--ink);
      font-size:13px;
    }
    .subtabbtn.active { background:rgba(13,138,152,0.12); border-color:rgba(13,138,152,0.44); color:var(--brand-strong); }
    .subpanel { display:none; }
    .subpanel.active { display:block; }
    .essay-card {
      background:linear-gradient(180deg,#ffffff 0%, #f6fbfb 100%);
      border:1px solid var(--line);
      border-radius:var(--radius-sm);
      padding:14px 15px;
      box-shadow:var(--shadow-sm);
      margin-bottom:12px;
    }
    .essay-card h4 {
      margin:0 0 8px;
      font-size:15px;
      color:var(--ink);
    }
    .essay-card p {
      margin:0;
      color:#36545c;
      line-height:1.55;
    }
    .takeaway-box {
      border-left:4px solid var(--brand-strong);
      background:rgba(13,138,152,0.08);
      border-radius:var(--radius-sm);
      padding:12px 14px;
      margin:8px 0 16px;
      color:#27474f;
      line-height:1.5;
    }
    .plot-box {
      border:1px solid var(--line);
      border-radius:var(--radius-sm);
      padding:10px;
      background:#fff;
      box-shadow:var(--shadow-sm);
      margin-bottom:8px;
    }
    .plot-canvas {
      width:100%;
      min-height:320px;
    }
    .dictionary-grid {
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:12px;
    }
    @media (max-width: 980px) {
      .container { padding:16px 12px 24px; }
      .topbar { flex-direction:column; align-items:flex-start; }
      .hero { padding:16px 16px; }
      .hero h1 { font-size:24px; }
      .tabbar { flex-wrap:wrap; padding:8px; gap:10px; }
      .tabgroup { width:100%; }
      .tabbtn { flex:1 1 calc(50% - 6px); }
      .filters { grid-template-columns:1fr; }
      .kpis { grid-template-columns:1fr 1fr; }
      .split { grid-template-columns:1fr; }
      .detail-split { grid-template-columns:1fr; }
      .detail-kpis { grid-template-columns:1fr 1fr; }
      .dictionary-grid { grid-template-columns:1fr; }
    }
    @media (max-width: 640px) {
      .container { padding:12px 8px 18px; }
      .topbar { padding:8px 10px; gap:8px; }
      .hero { border-radius:16px; padding:14px 12px; }
      .hero h1 { font-size:20px; line-height:1.15; }
      .hero p { font-size:12px; margin-top:8px; }
      .tabbar { gap:8px; padding:8px; margin:12px 0; }
      .tabgroup { gap:8px; }
      .tabbtn { flex:1 1 100%; padding:10px 12px; }
      .panel { padding:14px 10px; border-radius:12px; }
      .kpis { grid-template-columns:1fr; gap:8px; }
      .detail-kpis { grid-template-columns:1fr; }
      .mode-buttons { display:grid; grid-template-columns:1fr; }
      .mode-btn { width:100%; }
      .search-filters .chips { flex-wrap:wrap; height:auto; overflow-x:visible; }
      .search-filters .chip { height:auto; min-height:34px; }
      .table-wrap { max-height:300px; }
      th, td { font-size:12px; padding:7px 6px; }
      .section-title { font-size:15px; margin:14px 0 6px; }
      .step-title { font-size:15px; margin:12px 0 8px; }
      .map-canvas { height:320px; }
      .i18n-debug { font-size:11px; }
    }
  </style>
</head>
<body>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <div class="container">
    <div class="topbar">
      <div class="hint" id="topbarHint">Modern SaaS Bento UI · Public Release + MSIS 522 Workflow</div>
      <div class="locale-wrap">
        <label id="localeLabel" for="localeSelect">Language</label>
        <select id="localeSelect"></select>
      </div>
    </div>
    <div class="hero">
      <h1 id="appTitle">King County Restaurant Safety Dashboard (HTML)</h1>
      <p id="metaLine">Public-facing hygiene transparency dashboard plus a display-oriented mirror of the assignment analytics workflow.</p>
    </div>

    <div class="tabbar">
      <div class="tabgroup left">
        <button id="tabBtnOverview" class="tabbtn active" data-tab="overview">Project Overview</button>
        <button id="tabBtnSearch" class="tabbtn" data-tab="search">Restaurant Search</button>
        <button id="tabBtnSummary" class="tabbtn" data-tab="summary">Historical Insights</button>
      </div>
      <div class="tabgroup right">
        <button id="tabBtnExecutive" class="tabbtn" data-tab="executive">Executive Summary</button>
        <button id="tabBtnDescriptive" class="tabbtn" data-tab="descriptive">Descriptive Analytics</button>
        <button id="tabBtnPerformance" class="tabbtn" data-tab="performance">Model Performance</button>
        <button id="tabBtnExplainability" class="tabbtn" data-tab="explainability">Explainability & Interactive Prediction</button>
      </div>
    </div>
    <div id="i18nDebugPanel" class="small i18n-debug"></div>

    <section id="tab-overview" class="panel active">
      <div class="kpis" id="overviewKpis"></div>
      <div class="section-title" id="overviewTitleWhat">What this project is</div>
      <div class="small" id="overviewIntro">
        This MSIS project translates official inspection data into clearer public information. It helps consumers understand risk, helps restaurants prioritize remediation, and helps agencies monitor city-level patterns.
      </div>
      <div class="section-title" id="overviewTitleHow">How to use this dashboard</div>
      <div class="small" id="overviewHowTo">
        1) Use Restaurant Search to find one establishment and inspect detailed history.
        2) Use Historical Insights to compare trends from consumer, owner, regulator, and King County audit views.
        3) Use the four homework panels on the right to review the executive summary, descriptive analytics, model comparison, and explainability workflow.
      </div>
      <div class="section-title" id="overviewTitleRating">Official Rating Levels</div>
      <div class="table-wrap"><table id="overviewRatingTable"></table></div>
      <div class="field">
        <label id="overviewPosterLabel">Select a rating category to view poster and interpretation</label>
        <select id="overviewRatingPosterSelect"></select>
      </div>
      <div class="split">
        <div class="poster-box" id="overviewPosterBox"></div>
        <div class="poster-desc small" id="overviewPosterDesc"></div>
      </div>
      <div class="section-title" id="overviewTitleRisk">How Risk Is Classified in This Project</div>
      <div class="small" id="overviewRiskText">
        This dashboard recalculates food safety ratings from risk level, the most recent routine inspections
        available for that restaurant (up to the official window cap), and the recent closure / return-inspection
        rule shown by King County. The county's published grade is retained separately for audit comparison.
      </div>
      <div class="section-title" id="overviewTitleViolation">Violation Types</div>
      <div class="kpis" id="overviewViolationKpis"></div>
      <div class="small" id="overviewViolationText">
        Red violations indicate direct foodborne illness risks. Blue violations indicate sanitation/process/facility issues.
      </div>
      <div class="section-title" id="overviewTitleTopCode">Most Frequent Violation Codes (Top 12)</div>
      <div class="table-wrap"><table id="overviewTopCodeTable"></table></div>
      <div class="section-title" id="overviewTitlePublic">How the public can participate</div>
      <div class="small" id="overviewPublicText">
        1) Check a restaurant before visiting and compare recent inspection history.
        2) Restaurant owners can use recurring violation patterns to prioritize fixes.
        3) For official resources, use the county portal and open data links below.
      </div>
      <div class="section-title" id="overviewTitlePolicy">Important Policy Notes</div>
      <div class="small" id="overviewPolicyText">
        Closure decisions may be triggered by very high point thresholds (e.g., high Red or total Red+Blue points) and repeat-risk patterns.
      </div>
      <div class="section-title" id="overviewTitleSource">Data Sources</div>
      <div class="small" id="overviewSourceText">
        King County search portal, Open Data dataset (f29f-zza5), and public API endpoint.
      </div>
      <div class="small" id="overviewMetaFooter" style="margin-top:14px;"></div>
    </section>

    <section id="tab-search" class="panel">
      <div class="step-title"><strong id="searchStep1Title">Step 1. Please enter the restaurant details you want to search</strong></div>
      <div class="small" id="searchStep1Hint">Choose one search method. Only filters for the active method are shown.</div>
      <div id="searchModeButtons" class="mode-buttons"></div>
      <div id="searchModeKeywordPanel" class="search-mode-panel">
        <div class="filters search-filters">
          <div class="field">
            <label id="searchLabelKeyword">Restaurant / Address / City / Zip</label>
            <input id="qInput" placeholder="e.g. tutta bella / seatle / 98101" />
          </div>
          <div class="field">
            <label id="searchLabelCity">City</label>
            <select id="citySelect"><option value="All">All</option></select>
          </div>
          <div class="field">
            <label id="searchLabelZip">Zip</label>
            <input id="zipInput" placeholder="98101" />
          </div>
          <div class="field">
            <label id="searchLabelRatingA">Latest Rating</label>
            <div id="ratingChipsKeyword" class="chips"></div>
          </div>
        </div>
      </div>
      <div id="searchModeMapPanel" class="search-mode-panel hidden">
        <div class="filters" style="grid-template-columns:1.1fr 2fr;">
          <div class="field">
            <label id="searchLabelMapCity">City</label>
            <select id="mapCitySelect"><option value="All">All</option></select>
          </div>
          <div class="field">
            <label id="searchLabelRatingB">Latest Rating</label>
            <div id="ratingChipsMap" class="chips"></div>
          </div>
        </div>
      </div>
      <div id="searchModeHybridPanel" class="search-mode-panel hidden">
        <div class="filters" style="grid-template-columns:2fr 2fr;">
          <div class="field">
            <label id="searchLabelCombo">Restaurant / Address</label>
            <input id="comboQueryInput" placeholder="e.g. pho / bellevue way" />
          </div>
          <div class="field">
            <label id="searchLabelRatingC">Latest Rating</label>
            <div id="ratingChipsHybrid" class="chips"></div>
          </div>
        </div>
      </div>
      <div id="searchMapBlock" class="map-block hidden">
        <div class="small" id="searchMapMeta">
          Click one point on the map to choose a restaurant. The detailed inspection view still opens below.
        </div>
        <div id="searchMap" class="map-canvas"></div>
      </div>
      <div style="margin-top:10px;">
        <button id="clearSearchFiltersBtn" class="secondary-btn">Clear all filters</button>
      </div>
      <div class="meta" id="searchMeta"></div>
      <div class="small" id="searchSelectionMeta"></div>
      <div class="step-sep"></div>
      <div class="step-title"><strong id="searchStep2Title">Step 2. Please click any restaurant to view detailed results</strong></div>
      <div class="small" id="searchStep2Hint">Use the small sort button beside each field to toggle ascending / descending.</div>
      <div class="table-wrap"><table id="resultsTable"></table></div>

      <div id="searchDetailEmpty" class="info-block small">
        Click one row in Search Results to open restaurant details and inspection history.
      </div>
      <div id="searchDetailBlock" class="hidden">
        <div class="detail-split">
          <div class="detail-poster" id="detailPosterBox"></div>
          <div>
            <div class="kpis detail-kpis" id="detailKpis"></div>
          </div>
        </div>

        <div class="step-sep"></div>
        <div class="step-title"><strong id="searchStep3Title">Step 3. Please click one inspection to view violation details</strong></div>
        <div class="filters" style="grid-template-columns:1.1fr 1.1fr 1.5fr; margin-bottom:10px;">
          <div class="field">
            <label id="historyLabelStart">Start date</label>
            <input id="eventDateFrom" type="date" />
          </div>
          <div class="field">
            <label id="historyLabelEnd">End date</label>
            <input id="eventDateTo" type="date" />
          </div>
          <div class="field">
            <label id="historyLabelType">Inspection type</label>
            <select id="eventTypeSelect"><option value="">All</option></select>
          </div>
        </div>
        <div style="margin:0 0 10px;">
          <button id="clearHistoryFiltersBtn" class="secondary-btn">Clear filters</button>
        </div>
        <div class="table-wrap fixed-320"><table id="eventsTable"></table></div>

        <div id="violEmpty" class="info-block small">
          Click one inspection row to view Violations & Remediation for that day.
        </div>
        <div id="violBlock" class="hidden">
          <div class="step-sep"></div>
          <div class="step-title"><strong id="searchStep4Title">Step 4. Review violations and remediation recommendations</strong></div>
          <div class="small" id="violMeta"></div>
          <div class="table-wrap fixed-320"><table id="violTable"></table></div>
        </div>
      </div>
    </section>

    <section id="tab-summary" class="panel">
      <div class="step-title"><strong id="summaryStep1Title">Step 1. Please select your role</strong></div>
      <div id="insightRoleButtons" class="role-buttons"></div>
      <div class="step-sep"></div>
      <div class="step-title"><strong id="summaryStep2Title">Step 2. Please select what you care about</strong></div>
      <div class="field">
        <label id="summaryQuestionLabel">I want to know...</label>
        <select id="insightQuestionSelect"></select>
      </div>
      <div class="small" id="insightHint"></div>
      <div id="insightControls" class="filters" style="grid-template-columns:1fr 1fr; margin-top:10px;"></div>
      <div class="kpis" id="insightKpis"></div>
      <div id="insightContent"></div>
    </section>

    <section id="tab-executive" class="panel">
      <div id="executiveBlock"></div>
    </section>

    <section id="tab-descriptive" class="panel">
      <div id="descriptiveBlock"></div>
    </section>

    <section id="tab-performance" class="panel">
      <div id="performanceBlock"></div>
    </section>

    <section id="tab-explainability" class="panel">
      <div id="explainabilityBlock"></div>
    </section>
  </div>

  <script id="app-data" type="application/json">__DATA_JSON__</script>
  <script>
    const APP = JSON.parse(document.getElementById('app-data').textContent);
    const IDX = cols => Object.fromEntries((cols || []).map((c, i) => [c, i]));
    const bIdx = IDX(APP.business_columns || []);
    const eIdx = IDX(APP.event_columns || []);
    const vIdx = IDX(APP.violation_columns || []);
    const mIdx = IDX(APP.movement_columns || []);

    const state = {
      searchMode: 'keyword',
      ratingFilters: {
        keyword: new Set(),
        map: new Set(),
        hybrid: new Set()
      },
      filteredBusinesses: [],
      searchSortField: 'display_name',
      searchSortAsc: true,
      selectedBusinessId: '',
      selectedInspectionEventId: '',
      selectedInspectionDate: '',
      insightRole: 'Consumer',
      locale: 'en'
    };

    const SUPPORTED_LOCALES = ['en', 'zh-CN', 'es', 'fr', 'ja'];
    const I18N_DEBUG = new URLSearchParams(window.location.search).get('i18n_debug') === '1';
    const I18N_MISSING_KEYS = {};
    const LOCALE_LABELS = {
      en: 'English',
      'zh-CN': '中文',
      es: 'Español',
      fr: 'Français',
      ja: '日本語'
    };
    const I18N_MESSAGES = {
      en: {
        'topbar.hint': 'Modern SaaS Bento UI · Public Release + MSIS 522 Workflow',
        'locale.label': 'Language',
        'app.title': 'King County Restaurant Safety Dashboard (HTML)',
        'app.subtitle': 'Public-facing hygiene transparency dashboard plus a display-oriented mirror of the assignment analytics workflow.',
        'tab.overview': 'Project Overview',
        'tab.search': 'Restaurant Search',
        'tab.summary': 'Historical Insights',
        'tab.predict': 'Predictive Modeling',
        'overview.title.what': 'What this project is',
        'overview.title.how': 'How to use this dashboard',
        'overview.title.rating': 'Official Rating Levels',
        'overview.title.risk': 'How Risk Is Classified in This Project',
        'overview.title.violation': 'Violation Types',
        'overview.title.top_code': 'Most Frequent Violation Codes (Top 12)',
        'overview.title.public': 'How the public can participate',
        'overview.title.policy': 'Important Policy Notes',
        'overview.title.source': 'Data Sources',
        'overview.poster.label': 'Select a rating category to view poster and interpretation',
        'overview.intro': 'This MSIS project translates official inspection data into clearer public information. It helps consumers understand risk, helps restaurants prioritize remediation, and helps agencies monitor city-level patterns.',
        'overview.howto': '1) Use Restaurant Search to find one establishment and inspect detailed history.\\n2) Use Historical Insights to compare trends from consumer, owner, regulator, and King County audit views.\\n3) Use the four homework panels on the right to review the executive summary, descriptive analytics, model comparison, and explainability workflow.',
        'overview.risk_text': 'This dashboard recalculates food safety ratings from risk level, the most recent routine inspections available for that restaurant (up to the official window cap), and the recent closure / return-inspection rule shown by King County. The county\\'s published grade is retained separately for audit comparison.',
        'overview.violation_text': 'Red violations indicate direct foodborne illness risks. Blue violations indicate sanitation/process/facility issues.',
        'overview.public_text': '1) Check a restaurant before visiting and compare recent inspection history.\\n2) Restaurant owners can use recurring violation patterns to prioritize fixes.\\n3) For official resources, use the county portal and open data links below.',
        'overview.policy_text': 'Closure decisions may be triggered by very high point thresholds (e.g., high Red or total Red+Blue points) and repeat-risk patterns.',
        'overview.source_text': 'King County search portal, Open Data dataset (f29f-zza5), and public API endpoint.',
        'search.step1.title': 'Step 1. Please enter the restaurant details you want to search',
        'search.step1.hint': 'Choose one search method. Only filters for the active method are shown.',
        'search.step2.title': 'Step 2. Please click any restaurant to view detailed results',
        'search.step2.hint': 'Use the small sort button beside each field to toggle ascending / descending.',
        'search.step3.title': 'Step 3. Please click one inspection to view violation details',
        'search.step4.title': 'Step 4. Review violations and remediation recommendations',
        'search.label.keyword': 'Restaurant / Address / City / Zip',
        'search.label.city': 'City',
        'search.label.zip': 'Zip',
        'search.label.rating': 'Latest Rating',
        'search.label.combo': 'Restaurant / Address',
        'search.label.start': 'Start date',
        'search.label.end': 'End date',
        'search.label.type': 'Inspection type',
        'search.button.clear_all': 'Clear all filters',
        'search.button.clear': 'Clear filters',
        'search.map.hint': 'Click one point on the map to choose a restaurant. The detailed inspection view still opens below.',
        'summary.step1.title': 'Step 1. Please select your role',
        'summary.step2.title': 'Step 2. Please select what you care about',
        'summary.question_label': 'I want to know...',
        'common.all': 'All',
        'common.no_data': 'No data.',
        'common.no_data_table': 'No data',
        'common.na': 'N/A',
        'mode.keyword': 'Keyword Search',
        'mode.map': 'Map Search',
        'mode.hybrid': 'Keyword + Map',
        'role.Consumer': 'Consumer',
        'role.Restaurant Owner': 'Restaurant Owner',
        'role.Regulator': 'Regulator',
        'role.King County': 'King County',
        'rating.Excellent': 'Excellent',
        'rating.Good': 'Good',
        'rating.Okay': 'Okay',
        'rating.Needs to Improve': 'Needs to Improve',
        'rating.Rating not available': 'Rating not available',
        'question.C1': 'I want to know which restaurants got better or worse this month.',
        'question.C2': 'I want to know which restaurants are consistently safer over time.',
        'question.C3': 'I want to know which restaurants had serious red-flag violations recently.',
        'question.C4': 'I want to know how one restaurant compares with similar restaurants nearby.',
        'question.C5': 'I want to know which hygiene problems are most common right now.',
        'question.C6': 'I want to know whether food safety risk is improving or worsening over time.',
        'question.O1': 'I want to know which violations to fix first to reduce risk fastest.',
        'question.O2': 'I want to know which violations keep repeating at my restaurant.',
        'question.O3': 'I want to know how my restaurant compares with peer restaurants in my city.',
        'question.O4': 'I want to know what changed before my rating dropped.',
        'question.O5': 'I want to know which violation categories are increasing in my area.',
        'question.O6': 'I want to know whether my remediation actions are actually working.',
        'question.R1': 'I want to know which cities or zip codes need attention right now.',
        'question.R2': 'I want to know where high-risk rates are rising the fastest.',
        'question.R3': 'I want to know which restaurants show persistent high-risk patterns.',
        'question.R4': 'I want to know which areas have both high risk and high inspection workload.',
        'question.R5': 'I want to know whether risk patterns are seasonal.',
        'question.R6': 'I want to know where data quality issues may affect decisions.',
        'question.K1': 'I want to know which restaurants have differences between King County\\'s published grade and this dashboard\\'s calculated grade.',
        'question.K2': 'I want to know which records are outside King County scope or have locality ambiguity.',
        'question.K3': 'I want to know where city normalization quality is weak.',
        'question.K4': 'I want to know whether identifiers and dates are complete enough for reliable analytics.',
        'question.K5': 'I want to know where semantic nulls and text truncation could mislead users.',
        'question.K6': 'I want to know which quality issues are getting better or worse across data runs.'
      },
      'zh-CN': {
        'topbar.hint': 'Modern SaaS Bento 界面 · 公共数据透明体验',
        'locale.label': '语言',
        'app.title': '金县餐厅食品安全仪表盘（HTML）',
        'app.subtitle': '面向公众的金县餐厅卫生透明化仪表盘。',
        'tab.overview': '项目总览',
        'tab.search': '餐厅搜索',
        'tab.summary': '历史洞察',
        'tab.predict': '预测建模',
        'overview.title.what': '项目简介',
        'overview.title.how': '如何使用本仪表盘',
        'overview.title.rating': '官方评级标准',
        'overview.title.risk': '本项目如何定义风险',
        'overview.title.violation': '违规类型',
        'overview.title.top_code': '最常见违规代码（前12）',
        'overview.title.public': '公众如何参与',
        'overview.title.policy': '重要政策说明',
        'overview.title.source': '数据来源',
        'overview.poster.label': '请选择评级类别以查看海报与解读',
        'overview.intro': '本 MSIS 项目将官方检查数据转化为更易理解的公共信息，帮助消费者理解风险，帮助餐厅确定整改优先级，并帮助监管机构观察城市层面的趋势。',
        'overview.howto': '1) 在餐厅搜索中定位单个餐厅并查看历史。\\n2) 在历史洞察中从消费者、经营者、监管者和金县审计视角比较趋势。\\n3) 在预测建模中查看模型表现与下一次检查风险估计。',
        'overview.risk_text': '本仪表盘基于风险等级、该餐厅最近的常规检查（不超过官方窗口上限）以及金县公布的近期关停/复检规则重新计算食品安全评级。县官方评级被单独保留用于审计对比。',
        'overview.violation_text': '红色违规代表直接的食源性疾病风险；蓝色违规代表卫生、流程或设施问题。',
        'overview.public_text': '1) 就餐前先查看餐厅近期检查历史。\\n2) 餐厅经营者可依据重复违规模式安排整改优先级。\\n3) 如需官方信息，请使用下方县门户与开放数据链接。',
        'overview.policy_text': '当红分或红蓝总分达到较高阈值，或出现重复风险模式时，可能触发关停判定。',
        'overview.source_text': '金县检索门户、开放数据集（f29f-zza5）与公共 API 端点。',
        'search.step1.title': '步骤1：请输入要查询的餐厅信息',
        'search.step1.hint': '请选择一种搜索方式，仅显示当前方式对应的筛选项。',
        'search.step2.title': '步骤2：请点击任一餐厅查看详细结果',
        'search.step2.hint': '可点击每列旁的小排序按钮切换升序/降序。',
        'search.step3.title': '步骤3：请点击一次检查记录查看违规详情',
        'search.step4.title': '步骤4：查看违规与整改建议',
        'search.label.keyword': '餐厅 / 地址 / 城市 / 邮编',
        'search.label.city': '城市',
        'search.label.zip': '邮编',
        'search.label.rating': '最新评级',
        'search.label.combo': '餐厅 / 地址',
        'search.label.start': '开始日期',
        'search.label.end': '结束日期',
        'search.label.type': '检查类型',
        'search.button.clear_all': '清空全部筛选',
        'search.button.clear': '清空筛选',
        'search.map.hint': '请在地图上点击一个点位选择餐厅，详细检查记录将在下方显示。',
        'summary.step1.title': '步骤1：请选择你的角色',
        'summary.step2.title': '步骤2：请选择你关心的问题',
        'summary.question_label': '我想了解...',
        'common.all': '全部',
        'common.no_data': '暂无数据。',
        'common.no_data_table': '暂无数据',
        'common.na': '无',
        'mode.keyword': '关键词搜索',
        'mode.map': '地图搜索',
        'mode.hybrid': '关键词 + 地图',
        'role.Consumer': '消费者',
        'role.Restaurant Owner': '餐厅经营者',
        'role.Regulator': '监管者',
        'role.King County': '金县',
        'rating.Excellent': '优秀',
        'rating.Good': '良好',
        'rating.Okay': '一般',
        'rating.Needs to Improve': '需改进',
        'rating.Rating not available': '暂无评级',
        'question.C1': '我想知道本月哪些餐厅变好或变差了。',
        'question.C2': '我想知道哪些餐厅长期更安全稳定。',
        'question.C3': '我想知道近期哪些餐厅出现了严重红色违规。',
        'question.C4': '我想知道某家餐厅与附近同类餐厅相比如何。',
        'question.C5': '我想知道当前最常见的卫生问题是什么。',
        'question.C6': '我想知道食品安全风险总体是在改善还是恶化。',
        'question.O1': '我想知道优先整改哪些违规能最快降低风险。',
        'question.O2': '我想知道我店里哪些违规在反复出现。',
        'question.O3': '我想知道我店与同城同类餐厅相比如何。',
        'question.O4': '我想知道评级下降前发生了哪些变化。',
        'question.O5': '我想知道我所在区域哪些违规类别在上升。',
        'question.O6': '我想知道整改措施是否真的有效。',
        'question.R1': '我想知道当前哪些城市或邮编区域需要重点关注。',
        'question.R2': '我想知道哪些区域高风险率上升最快。',
        'question.R3': '我想知道哪些餐厅持续呈现高风险模式。',
        'question.R4': '我想知道哪些区域同时具备高风险与高检查工作量。',
        'question.R5': '我想知道风险模式是否具有季节性。',
        'question.R6': '我想知道哪些数据质量问题可能影响决策。',
        'question.K1': '我想知道哪些餐厅在县官方评级与本仪表盘计算评级之间存在差异。',
        'question.K2': '我想知道哪些记录超出金县范围或存在地理归属歧义。',
        'question.K3': '我想知道城市名称标准化质量薄弱在哪里。',
        'question.K4': '我想知道标识符和日期完整性是否足以支持可靠分析。',
        'question.K5': '我想知道语义空值或文本截断会在哪些地方误导用户。',
        'question.K6': '我想知道各类数据质量问题是在改善还是恶化。'
      },
      es: {
        'topbar.hint': 'UI Modern SaaS Bento · Experiencia de datos públicos transparentes',
        'locale.label': 'Idioma',
        'app.title': 'Panel de Seguridad Alimentaria del Condado de King (HTML)',
        'app.subtitle': 'Panel público de transparencia de higiene para restaurantes del condado de King.',
        'tab.overview': 'Resumen del Proyecto',
        'tab.search': 'Búsqueda de Restaurantes',
        'tab.summary': 'Información Histórica',
        'tab.predict': 'Modelado Predictivo',
        'overview.title.what': 'Que es este proyecto',
        'overview.title.how': 'Como usar este panel',
        'overview.title.rating': 'Niveles Oficiales de Calificacion',
        'overview.title.risk': 'Como se clasifica el riesgo en este proyecto',
        'overview.title.violation': 'Tipos de infraccion',
        'overview.title.top_code': 'Codigos de infraccion mas frecuentes (Top 12)',
        'overview.title.public': 'Como puede participar el publico',
        'overview.title.policy': 'Notas importantes de politica',
        'overview.title.source': 'Fuentes de datos',
        'overview.poster.label': 'Seleccione una categoria de calificacion para ver el poster e interpretacion',
        'overview.intro': 'Este proyecto de MSIS convierte datos oficiales de inspeccion en informacion publica mas clara. Ayuda a consumidores a entender el riesgo, a restaurantes a priorizar mejoras y a agencias a monitorear patrones por ciudad.',
        'overview.howto': '1) Use Busqueda de Restaurantes para encontrar un establecimiento y revisar su historial.\\n2) Use Informacion Historica para comparar tendencias desde vistas de consumidor, propietario, regulador y auditoria del condado.\\n3) Use Modelado Predictivo para revisar rendimiento del modelo y riesgo estimado de la proxima inspeccion.',
        'overview.risk_text': 'Este panel recalcula calificaciones de seguridad alimentaria con base en nivel de riesgo, inspecciones rutinarias recientes disponibles (hasta el limite oficial) y la regla de cierre/reinspeccion reciente del Condado de King. La nota oficial publicada se conserva por separado para comparacion de auditoria.',
        'overview.violation_text': 'Las infracciones rojas indican riesgo directo de enfermedades transmitidas por alimentos. Las azules indican problemas de saneamiento, proceso o instalaciones.',
        'overview.public_text': '1) Revise un restaurante antes de visitarlo y compare su historial reciente.\\n2) Los propietarios pueden priorizar acciones con base en patrones repetidos de infraccion.\\n3) Para recursos oficiales, use el portal del condado y enlaces de datos abiertos.',
        'overview.policy_text': 'Las decisiones de cierre pueden activarse por umbrales altos de puntos (por ejemplo, puntos rojos altos o rojo+azul) y por patrones repetidos de riesgo.',
        'overview.source_text': 'Portal de busqueda del Condado de King, conjunto Open Data (f29f-zza5) y endpoint API publico.',
        'search.step1.title': 'Paso 1. Ingrese los datos del restaurante que desea buscar',
        'search.step1.hint': 'Elija un metodo de busqueda. Solo se muestran filtros del metodo activo.',
        'search.step2.title': 'Paso 2. Haga clic en cualquier restaurante para ver resultados detallados',
        'search.step2.hint': 'Use el boton de orden junto a cada campo para alternar ascendente/descendente.',
        'search.step3.title': 'Paso 3. Haga clic en una inspeccion para ver detalles de infracciones',
        'search.step4.title': 'Paso 4. Revise infracciones y recomendaciones de correccion',
        'search.label.keyword': 'Restaurante / Direccion / Ciudad / Codigo postal',
        'search.label.city': 'Ciudad',
        'search.label.zip': 'Codigo postal',
        'search.label.rating': 'Calificacion mas reciente',
        'search.label.combo': 'Restaurante / Direccion',
        'search.label.start': 'Fecha de inicio',
        'search.label.end': 'Fecha de fin',
        'search.label.type': 'Tipo de inspeccion',
        'search.button.clear_all': 'Borrar todos los filtros',
        'search.button.clear': 'Borrar filtros',
        'search.map.hint': 'Haga clic en un punto del mapa para elegir un restaurante. La vista detallada de inspecciones se abre abajo.',
        'summary.step1.title': 'Paso 1. Seleccione su rol',
        'summary.step2.title': 'Paso 2. Seleccione lo que le importa',
        'summary.question_label': 'Quiero saber...',
        'common.all': 'Todos',
        'common.no_data': 'Sin datos.',
        'common.no_data_table': 'Sin datos',
        'common.na': 'N/A',
        'mode.keyword': 'Búsqueda por Palabra',
        'mode.map': 'Búsqueda por Mapa',
        'mode.hybrid': 'Palabra + Mapa',
        'role.Consumer': 'Consumidor',
        'role.Restaurant Owner': 'Propietario del restaurante',
        'role.Regulator': 'Regulador',
        'role.King County': 'Condado de King',
        'rating.Excellent': 'Excelente',
        'rating.Good': 'Bueno',
        'rating.Okay': 'Aceptable',
        'rating.Needs to Improve': 'Necesita mejorar',
        'rating.Rating not available': 'Calificacion no disponible',
        'question.C1': 'Quiero saber que restaurantes mejoraron o empeoraron este mes.',
        'question.C2': 'Quiero saber que restaurantes son consistentemente mas seguros con el tiempo.',
        'question.C3': 'Quiero saber que restaurantes tuvieron infracciones graves recientemente.',
        'question.C4': 'Quiero saber como se compara un restaurante con restaurantes similares cercanos.',
        'question.C5': 'Quiero saber que problemas de higiene son mas comunes ahora.',
        'question.C6': 'Quiero saber si el riesgo de seguridad alimentaria esta mejorando o empeorando con el tiempo.',
        'question.O1': 'Quiero saber que infracciones corregir primero para reducir el riesgo mas rapido.',
        'question.O2': 'Quiero saber que infracciones se repiten en mi restaurante.',
        'question.O3': 'Quiero saber como se compara mi restaurante con restaurantes pares en mi ciudad.',
        'question.O4': 'Quiero saber que cambio antes de que bajara mi calificacion.',
        'question.O5': 'Quiero saber que categorias de infracciones estan aumentando en mi zona.',
        'question.O6': 'Quiero saber si mis acciones correctivas realmente estan funcionando.',
        'question.R1': 'Quiero saber que ciudades o codigos postales necesitan atencion ahora.',
        'question.R2': 'Quiero saber donde las tasas de alto riesgo estan subiendo mas rapido.',
        'question.R3': 'Quiero saber que restaurantes muestran patrones persistentes de alto riesgo.',
        'question.R4': 'Quiero saber que zonas tienen alto riesgo y alta carga de inspeccion.',
        'question.R5': 'Quiero saber si los patrones de riesgo son estacionales.',
        'question.R6': 'Quiero saber donde los problemas de calidad de datos pueden afectar decisiones.',
        'question.K1': 'Quiero saber que restaurantes tienen diferencias entre la calificacion publicada por el condado y la calculada por este panel.',
        'question.K2': 'Quiero saber que registros estan fuera del alcance del condado o tienen ambiguedad de localidad.',
        'question.K3': 'Quiero saber donde es debil la normalizacion de ciudades.',
        'question.K4': 'Quiero saber si identificadores y fechas estan suficientemente completos para analisis confiables.',
        'question.K5': 'Quiero saber donde los nulos semanticos y truncamientos de texto pueden confundir a los usuarios.',
        'question.K6': 'Quiero saber que problemas de calidad estan mejorando o empeorando entre ejecuciones.'
      },
      fr: {
        'topbar.hint': 'UI Modern SaaS Bento · Expérience de transparence des données publiques',
        'locale.label': 'Langue',
        'app.title': 'Tableau de Bord de Sécurité Alimentaire du Comté de King (HTML)',
        'app.subtitle': 'Tableau de bord public de transparence sanitaire pour les restaurants du comté de King.',
        'tab.overview': 'Aperçu du Projet',
        'tab.search': 'Recherche de Restaurants',
        'tab.summary': 'Analyses Historiques',
        'tab.predict': 'Modélisation Prédictive',
        'overview.title.what': 'Presentation du projet',
        'overview.title.how': 'Comment utiliser ce tableau de bord',
        'overview.title.rating': 'Niveaux officiels de note',
        'overview.title.risk': 'Comment le risque est classe dans ce projet',
        'overview.title.violation': 'Types de violation',
        'overview.title.top_code': 'Codes de violation les plus frequents (Top 12)',
        'overview.title.public': 'Comment le public peut participer',
        'overview.title.policy': 'Notes importantes de politique',
        'overview.title.source': 'Sources de donnees',
        'overview.poster.label': 'Selectionnez une categorie de note pour voir le poster et l interpretation',
        'overview.intro': 'Ce projet MSIS transforme les donnees officielles d inspection en information publique plus lisible. Il aide les consommateurs a comprendre le risque, les restaurants a prioriser les corrections et les agences a suivre les tendances par ville.',
        'overview.howto': '1) Utilisez Recherche de Restaurants pour trouver un etablissement et examiner son historique.\\n2) Utilisez Analyses Historiques pour comparer les tendances selon les vues consommateur, proprietaire, regulateur et audit du comte.\\n3) Utilisez Modelisation Predictive pour examiner la performance du modele et le risque estime de la prochaine inspection.',
        'overview.risk_text': 'Ce tableau recalcule les notes de securite alimentaire selon le niveau de risque, les inspections de routine recentes disponibles (dans la limite officielle) et la regle recente de fermeture ou reinspection du Comte de King. La note officielle publiee est conservee separement pour comparaison d audit.',
        'overview.violation_text': 'Les violations rouges indiquent un risque direct de maladie d origine alimentaire. Les violations bleues indiquent des problemes de salubrite, de processus ou d installations.',
        'overview.public_text': '1) Verifiez un restaurant avant votre visite et comparez son historique recent.\\n2) Les proprietaires peuvent prioriser les actions via les motifs repetes de violation.\\n3) Pour les ressources officielles, utilisez le portail du comte et les liens open data ci-dessous.',
        'overview.policy_text': 'Les decisions de fermeture peuvent etre declenchees par des seuils eleves de points (par exemple, points rouges eleves ou total rouge+bleu) et des motifs repetes de risque.',
        'overview.source_text': 'Portail de recherche du Comte de King, jeu Open Data (f29f-zza5) et endpoint API public.',
        'search.step1.title': 'Etape 1. Saisissez les informations du restaurant a rechercher',
        'search.step1.hint': 'Choisissez une methode de recherche. Seuls les filtres de la methode active sont affiches.',
        'search.step2.title': 'Etape 2. Cliquez sur un restaurant pour voir les resultats detailles',
        'search.step2.hint': 'Utilisez le petit bouton de tri a cote de chaque champ pour alterner ascendant/descendant.',
        'search.step3.title': 'Etape 3. Cliquez sur une inspection pour voir les details des violations',
        'search.step4.title': 'Etape 4. Consultez les violations et recommandations de correction',
        'search.label.keyword': 'Restaurant / Adresse / Ville / Code postal',
        'search.label.city': 'Ville',
        'search.label.zip': 'Code postal',
        'search.label.rating': 'Note la plus recente',
        'search.label.combo': 'Restaurant / Adresse',
        'search.label.start': 'Date de debut',
        'search.label.end': 'Date de fin',
        'search.label.type': 'Type d inspection',
        'search.button.clear_all': 'Effacer tous les filtres',
        'search.button.clear': 'Effacer les filtres',
        'search.map.hint': 'Cliquez sur un point de la carte pour choisir un restaurant. La vue detaillee des inspections reste disponible ci-dessous.',
        'summary.step1.title': 'Etape 1. Selectionnez votre role',
        'summary.step2.title': 'Etape 2. Selectionnez ce qui vous interesse',
        'summary.question_label': 'Je veux savoir...',
        'common.all': 'Tous',
        'common.no_data': 'Aucune donnee.',
        'common.no_data_table': 'Aucune donnee',
        'common.na': 'N/A',
        'mode.keyword': 'Recherche par Mot-clé',
        'mode.map': 'Recherche par Carte',
        'mode.hybrid': 'Mot-clé + Carte',
        'role.Consumer': 'Consommateur',
        'role.Restaurant Owner': 'Proprietaire du restaurant',
        'role.Regulator': 'Regulateur',
        'role.King County': 'Comte de King',
        'rating.Excellent': 'Excellent',
        'rating.Good': 'Bon',
        'rating.Okay': 'Acceptable',
        'rating.Needs to Improve': 'A ameliorer',
        'rating.Rating not available': 'Note non disponible',
        'question.C1': 'Je veux savoir quels restaurants se sont ameliores ou deteriorés ce mois-ci.',
        'question.C2': 'Je veux savoir quels restaurants sont regulierement plus surs dans le temps.',
        'question.C3': 'Je veux savoir quels restaurants ont eu des violations graves recemment.',
        'question.C4': 'Je veux savoir comment un restaurant se compare a des restaurants similaires proches.',
        'question.C5': 'Je veux savoir quels problemes d hygiene sont les plus frequents actuellement.',
        'question.C6': 'Je veux savoir si le risque sanitaire alimentaire s ameliore ou se degrade avec le temps.',
        'question.O1': 'Je veux savoir quelles violations corriger en premier pour reduire le risque plus vite.',
        'question.O2': 'Je veux savoir quelles violations se repetent dans mon restaurant.',
        'question.O3': 'Je veux savoir comment mon restaurant se compare aux restaurants pairs dans ma ville.',
        'question.O4': 'Je veux savoir ce qui a change avant la baisse de ma note.',
        'question.O5': 'Je veux savoir quelles categories de violations augmentent dans ma zone.',
        'question.O6': 'Je veux savoir si mes actions correctives fonctionnent vraiment.',
        'question.R1': 'Je veux savoir quelles villes ou quels codes postaux demandent une attention immediate.',
        'question.R2': 'Je veux savoir ou les taux de risque eleve augmentent le plus vite.',
        'question.R3': 'Je veux savoir quels restaurants montrent des motifs persistants de risque eleve.',
        'question.R4': 'Je veux savoir quelles zones combinent risque eleve et forte charge d inspection.',
        'question.R5': 'Je veux savoir si les motifs de risque sont saisonniers.',
        'question.R6': 'Je veux savoir ou les problemes de qualite des donnees peuvent affecter les decisions.',
        'question.K1': 'Je veux savoir quels restaurants presentent des ecarts entre la note publiee par le comte et la note calculee par ce tableau.',
        'question.K2': 'Je veux savoir quels enregistrements sont hors perimetre du comte ou presentent une ambiguite de localite.',
        'question.K3': 'Je veux savoir ou la normalisation des villes est faible.',
        'question.K4': 'Je veux savoir si les identifiants et les dates sont suffisamment complets pour une analyse fiable.',
        'question.K5': 'Je veux savoir ou les valeurs nulles semantiques et les troncatures de texte peuvent induire les utilisateurs en erreur.',
        'question.K6': 'Je veux savoir quels problemes de qualite s ameliorent ou se deteriorent entre les executions.'
      },
      ja: {
        'topbar.hint': 'Modern SaaS Bento UI ・公共データ透明化エクスペリエンス',
        'locale.label': '言語',
        'app.title': 'キング郡レストラン食品安全ダッシュボード（HTML）',
        'app.subtitle': 'キング郡レストラン向けの公開衛生透明化ダッシュボード。',
        'tab.overview': 'プロジェクト概要',
        'tab.search': 'レストラン検索',
        'tab.summary': '履歴インサイト',
        'tab.predict': '予測モデリング',
        'overview.title.what': 'このプロジェクトについて',
        'overview.title.how': 'このダッシュボードの使い方',
        'overview.title.rating': '公式評価レベル',
        'overview.title.risk': '本プロジェクトのリスク分類',
        'overview.title.violation': '違反タイプ',
        'overview.title.top_code': '最頻出の違反コード（上位12件）',
        'overview.title.public': '市民ができること',
        'overview.title.policy': '重要なポリシーノート',
        'overview.title.source': 'データソース',
        'overview.poster.label': '評価カテゴリを選択してポスターと解説を表示',
        'overview.intro': 'この MSIS プロジェクトは公式検査データを分かりやすい公開情報に変換します。消費者のリスク理解、店舗の改善優先順位付け、行政の都市別傾向把握を支援します。',
        'overview.howto': '1) レストラン検索で対象店舗を探し、詳細履歴を確認します。\\n2) 履歴インサイトで消費者・オーナー・規制担当・キング郡監査視点の傾向を比較します。\\n3) 予測モデリングでモデル性能と次回検査の推定リスクを確認します。',
        'overview.risk_text': 'このダッシュボードは、リスクレベル、直近の定期検査（公式の集計上限内）、およびキング郡の最近の営業停止/再検査ルールに基づいて食品安全評価を再計算します。郡の公表評価は監査比較のため別途保持します。',
        'overview.violation_text': '赤の違反は食中毒の直接リスク、青の違反は衛生・工程・施設上の問題を示します。',
        'overview.public_text': '1) 来店前に店舗を確認し、最近の検査履歴を比較します。\\n2) 店舗オーナーは反復違反パターンをもとに改善優先度を決められます。\\n3) 公式情報は郡ポータルと下記オープンデータを参照してください。',
        'overview.policy_text': '高い点数閾値（例：赤点が高い、または赤+青点が高い）や反復リスクパターンにより営業停止判断が行われる場合があります。',
        'overview.source_text': 'キング郡検索ポータル、Open Data データセット（f29f-zza5）、公開 API エンドポイント。',
        'search.step1.title': 'ステップ1：検索したいレストラン情報を入力してください',
        'search.step1.hint': '検索方法を1つ選択してください。選択中の方法に対応するフィルターのみ表示されます。',
        'search.step2.title': 'ステップ2：任意のレストランをクリックして詳細結果を表示',
        'search.step2.hint': '各項目横の小さなソートボタンで昇順/降順を切り替えできます。',
        'search.step3.title': 'ステップ3：検査を1件クリックして違反詳細を表示',
        'search.step4.title': 'ステップ4：違反内容と改善提案を確認',
        'search.label.keyword': 'レストラン / 住所 / 市 / 郵便番号',
        'search.label.city': '市',
        'search.label.zip': '郵便番号',
        'search.label.rating': '最新評価',
        'search.label.combo': 'レストラン / 住所',
        'search.label.start': '開始日',
        'search.label.end': '終了日',
        'search.label.type': '検査タイプ',
        'search.button.clear_all': 'すべてのフィルターをクリア',
        'search.button.clear': 'フィルターをクリア',
        'search.map.hint': '地図上のポイントをクリックしてレストランを選択します。詳細検査ビューは下部に表示されます。',
        'summary.step1.title': 'ステップ1：役割を選択してください',
        'summary.step2.title': 'ステップ2：知りたい内容を選択してください',
        'summary.question_label': '知りたいこと...',
        'common.all': 'すべて',
        'common.no_data': 'データがありません。',
        'common.no_data_table': 'データなし',
        'common.na': 'N/A',
        'mode.keyword': 'キーワード検索',
        'mode.map': '地図検索',
        'mode.hybrid': 'キーワード + 地図',
        'role.Consumer': '消費者',
        'role.Restaurant Owner': '店舗オーナー',
        'role.Regulator': '規制担当',
        'role.King County': 'キング郡',
        'rating.Excellent': '優秀',
        'rating.Good': '良好',
        'rating.Okay': '可',
        'rating.Needs to Improve': '要改善',
        'rating.Rating not available': '評価なし',
        'question.C1': '今月、どのレストランの評価が改善または悪化したかを知りたい。',
        'question.C2': '長期的により安全性が高いレストランを知りたい。',
        'question.C3': '最近、重大な赤色違反があったレストランを知りたい。',
        'question.C4': 'あるレストランが近隣の類似店と比べてどうか知りたい。',
        'question.C5': '現在、最も多い衛生問題を知りたい。',
        'question.C6': '食品安全リスクが時間とともに改善しているか悪化しているか知りたい。',
        'question.O1': 'リスクを最速で下げるために、どの違反から優先的に改善すべきか知りたい。',
        'question.O2': '自店で繰り返し発生している違反を知りたい。',
        'question.O3': '自店が同じ市内の同業店と比べてどうか知りたい。',
        'question.O4': '評価が下がる前に何が変わったか知りたい。',
        'question.O5': '地域で増加している違反カテゴリを知りたい。',
        'question.O6': '改善施策が実際に効果を出しているか知りたい。',
        'question.R1': '今、どの都市または郵便番号エリアに注意が必要か知りたい。',
        'question.R2': '高リスク率が最も速く上昇している場所を知りたい。',
        'question.R3': '継続的な高リスク傾向を示すレストランを知りたい。',
        'question.R4': '高リスクかつ検査負荷が高いエリアを知りたい。',
        'question.R5': 'リスクパターンに季節性があるか知りたい。',
        'question.R6': '意思決定に影響し得るデータ品質問題の場所を知りたい。',
        'question.K1': '郡の公表評価と本ダッシュボード計算評価に差異があるレストランを知りたい。',
        'question.K2': 'キング郡の対象外、または地域帰属が曖昧なレコードを知りたい。',
        'question.K3': '都市名正規化の品質が弱い箇所を知りたい。',
        'question.K4': '識別子と日付の完全性が信頼できる分析に十分か知りたい。',
        'question.K5': '意味的な欠損やテキスト切り捨てがユーザーを誤解させる箇所を知りたい。',
        'question.K6': 'データ品質問題が実行間で改善しているか悪化しているか知りたい。'
      }
    };

    function t(key, params) {
      const locale = state.locale || 'en';
      const map = I18N_MESSAGES[locale] || {};
      const enMap = I18N_MESSAGES.en || {};
      let out = map[key] || enMap[key] || key;
      if (locale !== 'en' && !Object.prototype.hasOwnProperty.call(map, key) && Object.prototype.hasOwnProperty.call(enMap, key)) {
        if (!I18N_MISSING_KEYS[locale]) I18N_MISSING_KEYS[locale] = new Set();
        I18N_MISSING_KEYS[locale].add(key);
        if (I18N_DEBUG) console.warn('[i18n missing][' + locale + '] ' + key);
      }
      const values = params || {};
      Object.keys(values).forEach(name => {
        out = out.replaceAll('{' + name + '}', String(values[name]));
      });
      return out;
    }

    function renderI18nDebugPanel() {
      const panel = document.getElementById('i18nDebugPanel');
      if (!panel) return;
      if (!I18N_DEBUG) {
        panel.style.display = 'none';
        panel.textContent = '';
        return;
      }
      const locale = state.locale || 'en';
      const keys = Array.from(I18N_MISSING_KEYS[locale] || []).sort();
      panel.style.display = 'block';
      panel.textContent = keys.length
        ? ('i18n missing keys (' + locale + '):\\n' + keys.join('\\n'))
        : ('i18n missing keys (' + locale + '): none');
    }

    function getInitialLocale() {
      const fromUrl = new URLSearchParams(window.location.search).get('lang');
      if (fromUrl && SUPPORTED_LOCALES.includes(fromUrl)) return fromUrl;
      const fromStorage = localStorage.getItem('kc_dashboard_locale');
      if (fromStorage && SUPPORTED_LOCALES.includes(fromStorage)) return fromStorage;
      return 'en';
    }

    function setLocale(locale) {
      const next = SUPPORTED_LOCALES.includes(locale) ? locale : 'en';
      state.locale = next;
      localStorage.setItem('kc_dashboard_locale', next);
      const url = new URL(window.location.href);
      url.searchParams.set('lang', next);
      history.replaceState({}, '', url.toString());
    }

    function translateRatingLabel(label) {
      return t('rating.' + String(label || ''));
    }

    const ratingOptions = ['Excellent', 'Good', 'Okay', 'Needs to Improve', 'Rating not available'];
    const ratingRank = {
      'Excellent': 1,
      'Good': 2,
      'Okay': 3,
      'Needs to Improve': 4,
      'Rating not available': 5
    };
    const ratingColors = {
      'Excellent': '#099268',
      'Good': '#82c91e',
      'Okay': '#c0d72f',
      'Needs to Improve': '#adb5bd',
      'Rating not available': '#ced4da'
    };
    const searchModes = [
      { key: 'keyword' },
      { key: 'map' },
      { key: 'hybrid' }
    ];
    const insightQuestionMap = {
      Consumer: [
        { id: 'C1', label: 'I want to know which restaurants got better or worse this month.' },
        { id: 'C2', label: 'I want to know which restaurants are consistently safer over time.' },
        { id: 'C3', label: 'I want to know which restaurants had serious red-flag violations recently.' },
        { id: 'C4', label: 'I want to know how one restaurant compares with similar restaurants nearby.' },
        { id: 'C5', label: 'I want to know which hygiene problems are most common right now.' },
        { id: 'C6', label: 'I want to know whether food safety risk is improving or worsening over time.' }
      ],
      'Restaurant Owner': [
        { id: 'O1', label: 'I want to know which violations to fix first to reduce risk fastest.' },
        { id: 'O2', label: 'I want to know which violations keep repeating at my restaurant.' },
        { id: 'O3', label: 'I want to know how my restaurant compares with peer restaurants in my city.' },
        { id: 'O4', label: 'I want to know what changed before my rating dropped.' },
        { id: 'O5', label: 'I want to know which violation categories are increasing in my area.' },
        { id: 'O6', label: 'I want to know whether my remediation actions are actually working.' }
      ],
      Regulator: [
        { id: 'R1', label: 'I want to know which cities or zip codes need attention right now.' },
        { id: 'R2', label: 'I want to know where high-risk rates are rising the fastest.' },
        { id: 'R3', label: 'I want to know which restaurants show persistent high-risk patterns.' },
        { id: 'R4', label: 'I want to know which areas have both high risk and high inspection workload.' },
        { id: 'R5', label: 'I want to know whether risk patterns are seasonal.' },
        { id: 'R6', label: 'I want to know where data quality issues may affect decisions.' }
      ],
      'King County': [
        { id: 'K1', label: 'I want to know which restaurants have differences between King County\\'s published grade and this dashboard\\'s calculated grade.' },
        { id: 'K2', label: 'I want to know which records are outside King County scope or have locality ambiguity.' },
        { id: 'K3', label: 'I want to know where city normalization quality is weak.' },
        { id: 'K4', label: 'I want to know whether identifiers and dates are complete enough for reliable analytics.' },
        { id: 'K5', label: 'I want to know where semantic nulls and text truncation could mislead users.' },
        { id: 'K6', label: 'I want to know which quality issues are getting better or worse across data runs.' }
      ]
    };
    const officialRatingExplanationMap = {
      'Excellent': 'Consistently followed high standards for safe food handling.',
      'Good': 'Exceeded the minimum requirements for safe food handling.',
      'Okay': 'Met the minimum requirements for safe food handling.',
      'Needs to Improve': 'Was either closed within the last 90 days or needed multiple return inspections to correct unsafe food handling.',
      'Rating not available': 'No published food safety rating is currently available in the current record.'
    };
    const riskCategoryCardMap = {
      '1': {
        label: 'Risk Category 1',
        bullets: [
          'Scopes: Cold holding, limited food prep',
          'Examples: Coffee stands, hot dog stands',
          'Cook Step Exceptions: Commercially processed microwave dinners'
        ]
      },
      '2': {
        label: 'Risk Category 2',
        bullets: [
          'Scopes: No Cook Step, Food Preparation',
          'Examples: Ice cream shop, grocery store, some bakeries',
          'Cook Step Exceptions: Pre-packed raw meat or seafood'
        ]
      },
      '3': {
        label: 'Risk Category 3',
        bullets: [
          'Scopes: Same Day Service or Complex Food Preparation, Meat or Seafood Market, Overnight Cooking, Time as a Control, Approved HACCP',
          'Examples: Restaurant, meat or seafood markets'
        ]
      }
    };
    const resultFields = [
      { key: 'display_name', label: 'Restaurant', type: 'text' },
      { key: 'address', label: 'Address', type: 'text' },
      { key: 'city', label: 'City', type: 'text' },
      { key: 'zip', label: 'Zip', type: 'text' }
    ];

    function resultFieldLabel(fieldKey) {
      if (fieldKey === 'display_name') return 'Restaurant';
      if (fieldKey === 'address') return 'Address';
      if (fieldKey === 'city') return t('search.label.city');
      if (fieldKey === 'zip') return t('search.label.zip');
      return fieldKey;
    }

    function esc(value) {
      return String(value == null ? '' : value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function num(value) {
      const n = Number(value);
      return Number.isFinite(n) ? n : null;
    }

    function pct(value) {
      const n = Number(value);
      return Number.isFinite(n) ? (n * 100).toFixed(1) + '%' : '-';
    }

    function norm(text) {
      return String(text || '')
        .toUpperCase()
        .replace(/[^A-Z0-9 ]+/g, ' ')
        .replace(/\\s+/g, ' ')
        .trim();
    }

    function levenshtein(a, b) {
      if (a === b) return 0;
      if (!a) return b.length;
      if (!b) return a.length;
      const dp = Array.from({ length: a.length + 1 }, () => Array(b.length + 1).fill(0));
      for (let i = 0; i <= a.length; i += 1) dp[i][0] = i;
      for (let j = 0; j <= b.length; j += 1) dp[0][j] = j;
      for (let i = 1; i <= a.length; i += 1) {
        for (let j = 1; j <= b.length; j += 1) {
          const cost = a[i - 1] === b[j - 1] ? 0 : 1;
          dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
        }
      }
      return dp[a.length][b.length];
    }

    function fuzzyMatch(query, target) {
      if (!query || !target) return false;
      if (target.includes(query)) return true;
      const qTokens = query.split(' ').filter(token => token.length >= 4);
      const tTokens = target.split(' ');
      for (const qToken of qTokens) {
        for (const tToken of tTokens) {
          if (Math.abs(qToken.length - tToken.length) > 2) continue;
          const distance = levenshtein(qToken, tToken);
          const ratio = 1 - distance / Math.max(qToken.length, tToken.length);
          if (ratio >= 0.88) return true;
        }
      }
      return false;
    }

    function applyStaticTranslations() {
      const textMap = {
        topbarHint: 'topbar.hint',
        localeLabel: 'locale.label',
        appTitle: 'app.title',
        metaLine: 'app.subtitle',
        tabBtnOverview: 'tab.overview',
        tabBtnSearch: 'tab.search',
        tabBtnSummary: 'tab.summary',
        tabBtnPredict: 'tab.predict',
        overviewTitleWhat: 'overview.title.what',
        overviewTitleHow: 'overview.title.how',
        overviewTitleRating: 'overview.title.rating',
        overviewTitleRisk: 'overview.title.risk',
        overviewTitleViolation: 'overview.title.violation',
        overviewTitleTopCode: 'overview.title.top_code',
        overviewTitlePublic: 'overview.title.public',
        overviewTitlePolicy: 'overview.title.policy',
        overviewTitleSource: 'overview.title.source',
        overviewPosterLabel: 'overview.poster.label',
        overviewIntro: 'overview.intro',
        overviewHowTo: 'overview.howto',
        overviewRiskText: 'overview.risk_text',
        overviewViolationText: 'overview.violation_text',
        overviewPublicText: 'overview.public_text',
        overviewPolicyText: 'overview.policy_text',
        overviewSourceText: 'overview.source_text',
        searchStep1Title: 'search.step1.title',
        searchStep1Hint: 'search.step1.hint',
        searchStep2Title: 'search.step2.title',
        searchStep2Hint: 'search.step2.hint',
        searchStep3Title: 'search.step3.title',
        searchStep4Title: 'search.step4.title',
        searchLabelKeyword: 'search.label.keyword',
        searchLabelCity: 'search.label.city',
        searchLabelMapCity: 'search.label.city',
        searchLabelZip: 'search.label.zip',
        searchLabelRatingA: 'search.label.rating',
        searchLabelRatingB: 'search.label.rating',
        searchLabelRatingC: 'search.label.rating',
        searchLabelCombo: 'search.label.combo',
        historyLabelStart: 'search.label.start',
        historyLabelEnd: 'search.label.end',
        historyLabelType: 'search.label.type',
        summaryStep1Title: 'summary.step1.title',
        summaryStep2Title: 'summary.step2.title',
        summaryQuestionLabel: 'summary.question_label'
      };
      Object.entries(textMap).forEach(([id, key]) => {
        const node = document.getElementById(id);
        if (node) node.textContent = t(key);
      });

      const qInput = document.getElementById('qInput');
      if (qInput) qInput.placeholder = 'e.g. tutta bella / seatle / 98101';
      const zipInput = document.getElementById('zipInput');
      if (zipInput) zipInput.placeholder = '98101';
      const comboInput = document.getElementById('comboQueryInput');
      if (comboInput) comboInput.placeholder = 'e.g. pho / bellevue way';

      const clearSearch = document.getElementById('clearSearchFiltersBtn');
      if (clearSearch) clearSearch.textContent = t('search.button.clear_all');
      const clearHistory = document.getElementById('clearHistoryFiltersBtn');
      if (clearHistory) clearHistory.textContent = t('search.button.clear');
      const mapMeta = document.getElementById('searchMapMeta');
      if (mapMeta) mapMeta.textContent = t('search.map.hint');

      const localeSelect = document.getElementById('localeSelect');
      if (localeSelect) localeSelect.value = state.locale;
      renderI18nDebugPanel();

      const cityOpt = document.querySelector('#citySelect option');
      if (cityOpt) cityOpt.textContent = t('common.all');
      const mapCityOpt = document.querySelector('#mapCitySelect option');
      if (mapCityOpt) mapCityOpt.textContent = t('common.all');
      const eventTypeOpt = document.querySelector('#eventTypeSelect option');
      if (eventTypeOpt) eventTypeOpt.textContent = t('common.all');
    }

    function initLocaleSelector() {
      const localeSelect = document.getElementById('localeSelect');
      if (!localeSelect) return;
      localeSelect.innerHTML = SUPPORTED_LOCALES.map(locale =>
        '<option value="' + esc(locale) + '">' + esc(LOCALE_LABELS[locale] || locale) + '</option>'
      ).join('');
      localeSelect.value = state.locale;
      localeSelect.addEventListener('change', () => {
        setLocale(localeSelect.value || 'en');
        applyStaticTranslations();
        renderSearchModeButtons();
        renderRatingChipsForMode('ratingChipsKeyword', 'keyword');
        renderRatingChipsForMode('ratingChipsMap', 'map');
        renderRatingChipsForMode('ratingChipsHybrid', 'hybrid');
        renderOverview();
        refreshSearch(true);
        renderInsightsSummary();
        renderHomeworkPanels();
        renderI18nDebugPanel();
      });
    }

    function setTable(tableId, headers, rows, emptyMessage) {
      const table = document.getElementById(tableId);
      const safeRows = rows || [];
      let html = '<thead><tr>' + headers.map(h => '<th>' + esc(h) + '</th>').join('') + '</tr></thead>';
      html += '<tbody>';
      if (!safeRows.length) {
        html += '<tr><td colspan="' + headers.length + '">' + esc(emptyMessage || t('common.no_data_table')) + '</td></tr>';
      } else {
        for (const row of safeRows) {
          html += '<tr>' + row.map(cell => '<td>' + esc(cell) + '</td>').join('') + '</tr>';
        }
      }
      html += '</tbody>';
      table.innerHTML = html;
    }

    function renderSearchModeButtons() {
      const wrap = document.getElementById('searchModeButtons');
      wrap.innerHTML = '';
      for (const mode of searchModes) {
        const btn = document.createElement('button');
        btn.className = 'mode-btn' + (state.searchMode === mode.key ? ' active' : '');
        btn.textContent = t('mode.' + mode.key);
        btn.addEventListener('click', () => {
          if (state.searchMode === mode.key) return;
          state.searchMode = mode.key;
          updateSearchModePanels();
          refreshSearch(true);
        });
        wrap.appendChild(btn);
      }
    }

    function updateSearchModePanels() {
      setHidden('searchModeKeywordPanel', state.searchMode !== 'keyword');
      setHidden('searchModeMapPanel', state.searchMode !== 'map');
      setHidden('searchModeHybridPanel', state.searchMode !== 'hybrid');
      setHidden('searchMapBlock', !(state.searchMode === 'map' || state.searchMode === 'hybrid'));
      renderSearchModeButtons();
    }

    function renderRatingChipsForMode(wrapperId, modeKey) {
      const wrap = document.getElementById(wrapperId);
      if (!wrap) return;
      wrap.innerHTML = '';
      for (const rating of ratingOptions) {
        const btn = document.createElement('button');
        btn.className = 'chip' + (state.ratingFilters[modeKey].has(rating) ? ' active' : '');
        btn.textContent = translateRatingLabel(rating);
        btn.addEventListener('click', () => {
          const activeSet = state.ratingFilters[modeKey];
          if (activeSet.has(rating)) activeSet.delete(rating);
          else activeSet.add(rating);
          btn.classList.toggle('active');
          refreshSearch(true);
        });
        wrap.appendChild(btn);
      }
    }

    function populateCityOptions() {
      const cities = new Set();
      for (const b of APP.businesses || []) {
        const city = b[bIdx.city];
        if (city) cities.add(city);
      }
      for (const selectId of ['citySelect', 'mapCitySelect']) {
        const sel = document.getElementById(selectId);
        if (!sel) continue;
        for (const city of Array.from(cities).sort()) {
          const opt = document.createElement('option');
          opt.value = city;
          opt.textContent = city;
          sel.appendChild(opt);
        }
      }
    }

    function clearSearchFilters() {
      document.getElementById('qInput').value = '';
      document.getElementById('citySelect').value = 'All';
      document.getElementById('zipInput').value = '';
      document.getElementById('mapCitySelect').value = 'All';
      document.getElementById('comboQueryInput').value = '';
      state.ratingFilters.keyword = new Set();
      state.ratingFilters.map = new Set();
      state.ratingFilters.hybrid = new Set();
      document.querySelectorAll('#ratingChipsKeyword .chip, #ratingChipsMap .chip, #ratingChipsHybrid .chip').forEach(btn => btn.classList.remove('active'));
      state.selectedBusinessId = '';
      state.selectedInspectionEventId = '';
      state.selectedInspectionDate = '';
      document.getElementById('eventDateFrom').value = '';
      document.getElementById('eventDateTo').value = '';
      document.getElementById('eventTypeSelect').innerHTML = '<option value=\"\">' + esc(t('common.all')) + '</option>';
      refreshSearch(false);
    }

    function getActiveSearchFilters() {
      if (state.searchMode === 'map') {
        return {
          query: '',
          city: document.getElementById('mapCitySelect').value || 'All',
          zip: '',
          selectedRatings: state.ratingFilters.map
        };
      }
      if (state.searchMode === 'hybrid') {
        return {
          query: document.getElementById('comboQueryInput').value.trim(),
          city: 'All',
          zip: '',
          selectedRatings: state.ratingFilters.hybrid
        };
      }
      return {
        query: document.getElementById('qInput').value.trim(),
        city: document.getElementById('citySelect').value || 'All',
        zip: document.getElementById('zipInput').value.trim(),
        selectedRatings: state.ratingFilters.keyword
      };
    }

    function filterBusinesses() {
      const filters = getActiveSearchFilters();
      const q = filters.query;
      const qNorm = norm(q);
      const city = filters.city;
      const zip = filters.zip;
      const selectedRatings = filters.selectedRatings;

      let out = (APP.businesses || []).filter(b => {
        if (city !== 'All' && b[bIdx.city] !== city) return false;
        if (zip && !String(b[bIdx.zip] || '').includes(zip)) return false;
        if (selectedRatings.size && !selectedRatings.has(b[bIdx.latest_rating])) return false;
        return true;
      });

      if (qNorm) {
        let direct = out.filter(b => {
          const blob = norm(b[bIdx.search_blob] || '');
          return blob.includes(qNorm);
        });
        if (direct.length < 60) {
          const fuzzy = out.filter(b => fuzzyMatch(qNorm, norm(b[bIdx.search_blob] || '')));
          const seen = new Set(direct.map(b => b[bIdx.business_id]));
          for (const row of fuzzy) {
            const bid = row[bIdx.business_id];
            if (!seen.has(bid)) direct.push(row);
          }
        }
        out = direct;
      }
      return out;
    }

    function compareByField(a, b, fieldMeta) {
      const idx = bIdx[fieldMeta.key];
      const av = a[idx];
      const bv = b[idx];

      if (fieldMeta.type === 'number') {
        const an = num(av);
        const bn = num(bv);
        if (an == null && bn == null) return 0;
        if (an == null) return 1;
        if (bn == null) return -1;
        return an - bn;
      }
      if (fieldMeta.type === 'rating') {
        const ar = ratingRank[String(av || '')] || 999;
        const br = ratingRank[String(bv || '')] || 999;
        return ar - br;
      }
      return String(av || '').localeCompare(String(bv || ''), undefined, {
        numeric: true,
        sensitivity: 'base'
      });
    }

    function sortBusinesses(rows) {
      const out = [...rows];
      const fieldMeta = resultFields.find(f => f.key === state.searchSortField) || resultFields[0];
      out.sort((a, b) => {
        const cmp = compareByField(a, b, fieldMeta);
        return state.searchSortAsc ? cmp : -cmp;
      });
      return out;
    }

    function setHidden(id, hidden) {
      const el = document.getElementById(id);
      if (!el) return;
      if (hidden) el.classList.add('hidden');
      else el.classList.remove('hidden');
    }

    function computeSearchMapView(rows) {
      if (!rows.length) return { lat: 47.5480, lon: -121.9836, zoom: 8.8 };
      const lats = rows.map(row => num(row[bIdx.latitude])).filter(value => value != null);
      const lons = rows.map(row => num(row[bIdx.longitude])).filter(value => value != null);
      if (!lats.length || !lons.length) return { lat: 47.5480, lon: -121.9836, zoom: 8.8 };
      let lat = lats.reduce((sum, value) => sum + value, 0) / lats.length;
      let lon = lons.reduce((sum, value) => sum + value, 0) / lons.length;
      let zoom = 9.2;
      if (state.selectedBusinessId) {
        const selected = rows.find(row => String(row[bIdx.business_id] || '') === String(state.selectedBusinessId || ''));
        if (selected) {
          const sLat = num(selected[bIdx.latitude]);
          const sLon = num(selected[bIdx.longitude]);
          if (sLat != null && sLon != null) {
            lat = sLat;
            lon = sLon;
            zoom = rows.length === 1 ? 13.8 : 12.2;
          }
        }
      }
      const latSpan = Math.max(...lats) - Math.min(...lats);
      const lonSpan = Math.max(...lons) - Math.min(...lons);
      const span = Math.max(latSpan, lonSpan);
      if (span > 0.9) zoom = Math.min(zoom, 8.4);
      else if (span > 0.45) zoom = Math.min(zoom, 9.0);
      else if (span > 0.22) zoom = Math.min(zoom, 10.0);
      else if (span > 0.08) zoom = Math.min(zoom, 11.2);
      else if (span > 0.03) zoom = Math.min(zoom, 12.4);
      else zoom = Math.max(zoom, 13.0);
      return { lat, lon, zoom };
    }

    function renderSearchMap(rows) {
      const mapWrap = document.getElementById('searchMap');
      const meta = document.getElementById('searchMapMeta');
      if (state.searchMode !== 'map' && state.searchMode !== 'hybrid') {
        mapWrap.innerHTML = '';
        return;
      }
      if (typeof Plotly === 'undefined') {
        mapWrap.innerHTML = '<div class="small" style="padding:12px;">Plotly could not be loaded, so map search is unavailable.</div>';
        return;
      }
      const mapRows = (rows || []).filter(row => num(row[bIdx.latitude]) != null && num(row[bIdx.longitude]) != null);
      meta.textContent =
        t('search.map.hint') + ' ' +
        'Mapped restaurants in current results: ' + mapRows.length.toLocaleString() + '.';
      if (!mapRows.length) {
        mapWrap.innerHTML = '<div class="small" style="padding:12px;">No matched restaurants with valid coordinates are available for map search.</div>';
        return;
      }

      const colors = mapRows.map(row => ratingColors[String(row[bIdx.latest_rating] || '')] || '#ced4da');
      const selectedIndex = mapRows.findIndex(row => String(row[bIdx.business_id] || '') === String(state.selectedBusinessId || ''));
      const view = computeSearchMapView(mapRows);
      const traces = [];
      const boundary = APP.map_boundary || { lon: [], lat: [] };
      if ((boundary.lon || []).length && (boundary.lat || []).length) {
        traces.push({
          type: 'scattermapbox',
          mode: 'lines',
          lon: boundary.lon,
          lat: boundary.lat,
          hoverinfo: 'skip',
          line: {
            color: '#5f6c7b',
            width: 2.2
          },
          opacity: 0.95,
          showlegend: false
        });
      }
      const trace = {
        type: 'scattermapbox',
        mode: 'markers',
        lat: mapRows.map(row => num(row[bIdx.latitude])),
        lon: mapRows.map(row => num(row[bIdx.longitude])),
        customdata: mapRows.map(row => [
          String(row[bIdx.business_id] || ''),
          String(row[bIdx.display_name] || ''),
          String(row[bIdx.address] || ''),
          String(row[bIdx.latest_rating] || '')
        ]),
        marker: {
          size: 11,
          color: colors,
          opacity: 0.82
        },
        selectedpoints: selectedIndex >= 0 ? [selectedIndex] : [],
        selected: { marker: { size: 16, opacity: 1.0 } },
        unselected: { marker: { opacity: 0.36 } },
        hovertemplate:
          '<b>%{customdata[1]}</b><br>' +
          '%{customdata[2]}<br>' +
          'Rating: %{customdata[3]}<extra></extra>'
      };
      traces.push(trace);
      const layout = {
        margin: { l: 0, r: 0, t: 0, b: 0 },
        dragmode: 'pan',
        clickmode: 'event+select',
        mapbox: {
          style: 'carto-positron',
          center: { lat: view.lat, lon: view.lon },
          zoom: view.zoom
        }
      };
      Plotly.newPlot(mapWrap, traces, layout, {
        responsive: true,
        displaylogo: false,
        scrollZoom: true
      }).then(() => {
        mapWrap.on('plotly_click', evt => {
          const point = (((evt || {}).points || [])[0]) || null;
          const custom = point && point.customdata ? point.customdata : [];
          const businessId = custom && custom.length ? String(custom[0] || '') : '';
          if (!businessId) return;
          state.selectedBusinessId = businessId;
          state.selectedInspectionEventId = '';
          state.selectedInspectionDate = '';
          renderResultsTable(state.filteredBusinesses);
          renderSearchMap(state.filteredBusinesses);
          renderRestaurantDetail(state.selectedBusinessId);
        });
      });
    }

    function clearSearchDetail(message) {
      state.selectedBusinessId = '';
      state.selectedInspectionEventId = '';
      state.selectedInspectionDate = '';

      const emptyBlock = document.getElementById('searchDetailEmpty');
      const defaultMessage = (state.searchMode === 'map' || state.searchMode === 'hybrid')
        ? 'Click one row in Search Results or one point on the map to open restaurant details and inspection history.'
        : 'Click one row in Search Results to open restaurant details and inspection history.';
      if (emptyBlock) {
        emptyBlock.textContent = message || defaultMessage;
        emptyBlock.classList.remove('hidden');
      }
      setHidden('searchDetailBlock', true);
      setHidden('violBlock', true);
      setHidden('violEmpty', false);
      document.getElementById('detailKpis').innerHTML = '';
      document.getElementById('detailPosterBox').innerHTML = '';
      document.getElementById('violMeta').textContent = '';
      setTable('eventsTable', ['Date', 'Type', 'Result', 'Score', 'Rating', 'Violations', 'Red Points', 'Blue Points'], [], 'No inspection history.');
      setTable('violTable', ['Type', 'Code', 'Points', 'Violation', 'Priority', 'Category', 'Remediation summary'], [], 'No violation details found.');
    }

    function formatAvgRedTitle(riskLevelNum) {
      if (String(riskLevelNum || '') === '3') return 'Avg Red Points in Recent 4 Inspections';
      if (String(riskLevelNum || '') === '1' || String(riskLevelNum || '') === '2') {
        return 'Avg Red Points in Recent 2 Inspections';
      }
      return 'Avg Red Points in Recent Inspections';
    }

    function buildKpiCard(title, value, noteText, bulletItems) {
      const note = String(noteText || '').trim();
      const bullets = Array.isArray(bulletItems) ? bulletItems.filter(Boolean) : [];
      const noteHtml = note ? '<div class="note">' + esc(note) + '</div>' : '';
      const listHtml = bullets.length
        ? '<ul>' + bullets.map(item => '<li>' + esc(item) + '</li>').join('') + '</ul>'
        : '';
      return '<div class="kpi"><div class="k">' + esc(title) + '</div><div class="v">' + esc(value) + '</div>' + noteHtml + listHtml + '</div>';
    }

    function formatScoreValue(value) {
      const n = num(value);
      if (n == null) return '-';
      return Number.isInteger(n) ? String(n) : n.toFixed(1);
    }

    function formatAvgRedValue(value) {
      const n = num(value);
      if (n == null) return '-';
      return n.toFixed(2);
    }

    function getFilteredEvents(events) {
      const from = document.getElementById('eventDateFrom').value || '';
      const to = document.getElementById('eventDateTo').value || '';
      const eventType = document.getElementById('eventTypeSelect').value || '';
      return (events || []).filter(row => {
        const date = String(row[eIdx.date] || '');
        const type = String(row[eIdx.type] || '');
        if (from && date && date < from) return false;
        if (to && date && date > to) return false;
        if (eventType && type !== eventType) return false;
        return true;
      });
    }

    function syncEventFilters(events) {
      const dateInputFrom = document.getElementById('eventDateFrom');
      const dateInputTo = document.getElementById('eventDateTo');
      const typeSelect = document.getElementById('eventTypeSelect');
      const dates = (events || []).map(row => String(row[eIdx.date] || '')).filter(Boolean).sort();
      const types = Array.from(new Set((events || []).map(row => String(row[eIdx.type] || '')).filter(Boolean))).sort();
      const minDate = dates[0] || '';
      const maxDate = dates[dates.length - 1] || '';
      dateInputFrom.min = minDate;
      dateInputFrom.max = maxDate;
      dateInputTo.min = minDate;
      dateInputTo.max = maxDate;
      if (!dateInputFrom.value || dateInputFrom.value < minDate || dateInputFrom.value > maxDate) dateInputFrom.value = minDate;
      if (!dateInputTo.value || dateInputTo.value < minDate || dateInputTo.value > maxDate) dateInputTo.value = maxDate;
      const previousType = typeSelect.value || '';
      typeSelect.innerHTML = '<option value=\"\">' + esc(t('common.all')) + '</option>' + types.map(v => '<option value=\"' + esc(v) + '\">' + esc(v) + '</option>').join('');
      if (previousType && types.includes(previousType)) typeSelect.value = previousType;
      else typeSelect.value = '';
    }

    function clearHistoryFilters() {
      const events = ((APP.events_by_business || {})[state.selectedBusinessId] || []);
      const dates = events.map(row => String(row[eIdx.date] || '')).filter(Boolean).sort();
      const minDate = dates[0] || '';
      const maxDate = dates[dates.length - 1] || '';
      document.getElementById('eventDateFrom').value = minDate;
      document.getElementById('eventDateTo').value = maxDate;
      document.getElementById('eventTypeSelect').value = '';
      if (state.selectedBusinessId) renderRestaurantDetail(state.selectedBusinessId);
    }

    function renderResultsTable(rows) {
      const table = document.getElementById('resultsTable');
      const orderedRows = [...rows];
      if (state.selectedBusinessId) {
        const selectedIndex = orderedRows.findIndex(
          row => String(row[bIdx.business_id] || '') === String(state.selectedBusinessId || '')
        );
        if (selectedIndex > 0) {
          const selectedRow = orderedRows.splice(selectedIndex, 1)[0];
          orderedRows.unshift(selectedRow);
        }
      }
      const displayRows = orderedRows.slice(0, 200);
      let html = '<thead><tr>';
      for (const field of resultFields) {
        const active = state.searchSortField === field.key;
        const indicator = active ? (state.searchSortAsc ? '▲' : '▼') : '↕';
        html += '<th><div class="th-wrap"><span>' + esc(resultFieldLabel(field.key)) + '</span>' +
          '<button class="sort-btn ' + (active ? 'active' : '') + '" data-field="' + esc(field.key) + '">' +
          esc(indicator) + '</button></div></th>';
      }
      html += '</tr></thead><tbody>';
      if (!displayRows.length) {
        html += '<tr><td colspan="' + String(resultFields.length) + '">No matched restaurants.</td></tr>';
      } else {
        for (const b of displayRows) {
          const businessId = String(b[bIdx.business_id] || '');
          const selected = businessId && businessId === state.selectedBusinessId;
          const cells = resultFields.map(field => '<td>' + esc(b[bIdx[field.key]]) + '</td>').join('');
          html += '<tr class="' + (selected ? 'selected-row' : '') + '" data-business-id="' + esc(businessId) + '">' +
            cells + '</tr>';
        }
      }
      html += '</tbody>';
      table.innerHTML = html;

      table.querySelectorAll('.sort-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const field = btn.dataset.field;
          if (state.searchSortField === field) state.searchSortAsc = !state.searchSortAsc;
          else {
            state.searchSortField = field;
            state.searchSortAsc = true;
          }
          refreshSearch(true);
        });
      });

      table.querySelectorAll('tbody tr[data-business-id]').forEach(tr => {
        tr.addEventListener('click', () => {
          state.selectedBusinessId = tr.dataset.businessId || '';
          state.selectedInspectionEventId = '';
          state.selectedInspectionDate = '';
          renderResultsTable(rows);
          renderSearchMap(state.filteredBusinesses);
          renderRestaurantDetail(state.selectedBusinessId);
        });
      });

      const selectedMeta = document.getElementById('searchSelectionMeta');
      if (selectedMeta) {
        if (displayRows.length && state.selectedBusinessId) {
          const selectedRow = displayRows.find(
            row => String(row[bIdx.business_id] || '') === String(state.selectedBusinessId || '')
          );
          if (selectedRow) {
            selectedMeta.textContent =
              'Selected restaurant: ' + String(selectedRow[bIdx.display_name] || '') +
              ' | Pinned to the top of the list.';
          } else {
            selectedMeta.textContent = '';
          }
        } else {
          selectedMeta.textContent = '';
        }
      }

      const wrap = table.closest('.table-wrap');
      if (wrap && state.selectedBusinessId) wrap.scrollTop = 0;
    }

    function refreshSearch(preserveRestaurantSelection) {
      const filtered = sortBusinesses(filterBusinesses());
      state.filteredBusinesses = filtered;
      const currentField = resultFields.find(f => f.key === state.searchSortField);
      const direction = state.searchSortAsc ? 'ASC' : 'DESC';
      const activeMode = searchModes.find(mode => mode.key === state.searchMode);
      document.getElementById('searchMeta').textContent =
        'Mode: ' + (activeMode ? t('mode.' + activeMode.key) : state.searchMode) + ' | ' +
        'Matched restaurants: ' + filtered.length.toLocaleString() +
        ' (showing up to 200 rows) | Sorted by: ' +
        (currentField ? resultFieldLabel(currentField.key) : state.searchSortField) +
        ' (' + direction + ')';
      renderSearchMap(filtered);

      if (!filtered.length) {
        renderResultsTable(filtered);
        clearSearchDetail('No matched restaurants. Adjust filters and try again.');
        return;
      }

      const previous = preserveRestaurantSelection ? state.selectedBusinessId : '';
      if (previous && filtered.some(b => String(b[bIdx.business_id] || '') === previous)) {
        state.selectedBusinessId = previous;
      } else if (!filtered.some(b => String(b[bIdx.business_id] || '') === state.selectedBusinessId)) {
        state.selectedBusinessId = '';
        state.selectedInspectionEventId = '';
        state.selectedInspectionDate = '';
      }

      renderResultsTable(filtered);
      if (state.selectedBusinessId) {
        renderRestaurantDetail(state.selectedBusinessId);
      } else {
        clearSearchDetail();
      }
    }

    function renderRestaurantDetail(businessId) {
      state.selectedBusinessId = businessId;
      const b = (APP.businesses || []).find(x => String(x[bIdx.business_id] || '') === String(businessId || ''));
      if (!b) {
        clearSearchDetail();
        return;
      }

      document.getElementById('searchDetailEmpty').classList.add('hidden');
      setHidden('searchDetailBlock', false);

      const riskLevelNum = String(b[bIdx.latest_risk_level_num] || '');
      const riskMeta = riskCategoryCardMap[riskLevelNum] || {};
      const kpis = [
        {
          title: 'Restaurant Name',
          value: b[bIdx.display_name] || '-'
        },
        {
          title: 'Restaurant Address',
          value: b[bIdx.address] || '-'
        },
        {
          title: 'Latest Inspection Date',
          value: b[bIdx.latest_date] || '-'
        },
        {
          title: 'Latest Inspection Score',
          value: formatScoreValue(b[bIdx.latest_score])
        },
        {
          title: 'Inspection Count',
          value: b[bIdx.inspection_count] || '0'
        },
        {
          title: formatAvgRedTitle(riskLevelNum),
          value: formatAvgRedValue(b[bIdx.latest_avg_red_points])
        },
        {
          title: 'Risk Category',
          value: riskMeta.label || '-',
          bullets: riskMeta.bullets || []
        },
        {
          title: 'Food Safety Rating',
          value: translateRatingLabel(b[bIdx.latest_rating] || ''),
          note: officialRatingExplanationMap[String(b[bIdx.latest_rating] || '')] || ''
        }
      ];
      document.getElementById('detailKpis').innerHTML = kpis.map(card =>
        buildKpiCard(card.title, card.value, card.note || '', card.bullets || [])
      ).join('');

      const ratingLabel = String(b[bIdx.latest_rating] || '');
      const posterCatalog = ((APP.overview || {}).poster_catalog || {});
      const poster = posterCatalog[ratingLabel] || {};
      const posterPath = String(poster.path || '');
      if (posterPath) {
        const src = posterPath.startsWith('/') ? ('file://' + posterPath) : posterPath;
        document.getElementById('detailPosterBox').innerHTML =
          '<img src="' + esc(src) + '" class="poster-img" alt="' + esc(ratingLabel + ' poster') + '" />';
      } else {
        document.getElementById('detailPosterBox').innerHTML =
          '<div class="small">Poster image not available in local images folder.</div>';
      }
      const events = (APP.events_by_business || {})[businessId] || [];
      syncEventFilters(events);
      const filteredEvents = getFilteredEvents(events);
      const hasSelected = filteredEvents.some(e => String(e[eIdx.event_id] || '') === String(state.selectedInspectionEventId || ''));
      if (!hasSelected) {
        state.selectedInspectionEventId = '';
        state.selectedInspectionDate = '';
      }

      renderEventsTable(filteredEvents);
      renderViolationsTable(businessId);
    }

    function renderEventsTable(events) {
      const table = document.getElementById('eventsTable');
      let html = '<thead><tr>' +
        '<th>Date</th><th>Type</th><th>Result</th><th>Score</th><th>Rating</th><th>Violations</th><th>Red Points</th><th>Blue Points</th>' +
        '</tr></thead><tbody>';

      if (!events.length) {
        state.selectedInspectionEventId = '';
        state.selectedInspectionDate = '';
        html += '<tr><td colspan="8">No inspection history.</td></tr>';
      } else {
        for (const row of events) {
          const eventId = String(row[eIdx.event_id] || '');
          const eventDate = String(row[eIdx.date] || '');
          const active = eventId && eventId === state.selectedInspectionEventId;
          html += '<tr class="' + (active ? 'selected-row' : '') + '" data-event-id="' + esc(eventId) +
            '" data-event-date="' + esc(eventDate) + '">' +
            '<td>' + esc(eventDate) + '</td>' +
            '<td>' + esc(row[eIdx.type]) + '</td>' +
            '<td>' + esc(row[eIdx.result]) + '</td>' +
            '<td>' + esc(row[eIdx.score]) + '</td>' +
            '<td>' + esc(translateRatingLabel(row[eIdx.rating])) + '</td>' +
            '<td>' + esc(row[eIdx.violations]) + '</td>' +
            '<td>' + esc(row[eIdx.red]) + '</td>' +
            '<td>' + esc(row[eIdx.blue]) + '</td>' +
            '</tr>';
        }
      }
      html += '</tbody>';
      table.innerHTML = html;

      table.querySelectorAll('tbody tr[data-event-id]').forEach(tr => {
        tr.addEventListener('click', () => {
          state.selectedInspectionEventId = tr.dataset.eventId || '';
          state.selectedInspectionDate = tr.dataset.eventDate || '';
          renderEventsTable(events);
          renderViolationsTable(state.selectedBusinessId);
        });
      });
    }

    function renderViolationsTable(businessId) {
      const allRows = ((APP.violations_by_business || {})[businessId] || []);
      const selectedEventId = String(state.selectedInspectionEventId || '');
      const selectedDate = String(state.selectedInspectionDate || '');
      if (!selectedEventId && !selectedDate) {
        setHidden('violBlock', true);
        setHidden('violEmpty', false);
        document.getElementById('violMeta').textContent = '';
        return;
      }

      let rows = [];
      if (selectedEventId) {
        rows = allRows.filter(v => String(v[vIdx.event_id] || '') === selectedEventId);
      }
      if (!rows.length && selectedDate) {
        rows = allRows.filter(v => String(v[vIdx.date] || '') === selectedDate);
      }

      const meta = 'Selected inspection: ' + (selectedDate || 'N/A') + ' | ID: ' + (selectedEventId || 'N/A');
      document.getElementById('violMeta').textContent = meta;
      setHidden('violEmpty', true);
      setHidden('violBlock', false);

      setTable(
        'violTable',
        ['Type', 'Code', 'Points', 'Violation', 'Priority', 'Category', 'Remediation summary'],
        rows.map(v => [
          v[vIdx.type],
          v[vIdx.code],
          v[vIdx.points],
          v[vIdx.violation],
          v[vIdx.priority],
          v[vIdx.category],
          v[vIdx.remediation_summary]
        ]),
        'No violation details found for the selected inspection.'
      );
    }

    function renderBarRows(containerId, rows, labelIndex, valueIndex, valueFormatter) {
      const host = document.getElementById(containerId);
      if (!rows || !rows.length) {
        host.innerHTML = '<div class="small">No data.</div>';
        return;
      }
      const maxValue = Math.max(...rows.map(r => Number(r[valueIndex] || 0)), 1);
      host.innerHTML = rows.map(r => {
        const label = String(r[labelIndex] || '');
        const value = Number(r[valueIndex] || 0);
        const width = Math.max(2, (value / maxValue) * 100);
        const display = valueFormatter ? valueFormatter(value) : String(value);
        return '<div class="barrow"><div style="width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' +
          esc(label) + '</div><div class="bar"><span style="width:' + width.toFixed(1) +
          '%"></span></div><div style="width:90px">' + esc(display) + '</div></div>';
      }).join('');
    }

    let _businessMetricsCache = null;
    let _eventsFlatCache = null;
    let _violationsFlatCache = null;

    function setInsightKpis(pairs) {
      const host = document.getElementById('insightKpis');
      const rows = pairs || [];
      if (!rows.length) {
        host.innerHTML = '';
        return;
      }
      host.innerHTML = rows.map(pair =>
        '<div class="kpi"><div class="k">' + esc(pair[0]) + '</div><div class="v">' + esc(pair[1]) + '</div></div>'
      ).join('');
    }

    function clearInsightPanels() {
      document.getElementById('insightControls').innerHTML = '';
      document.getElementById('insightContent').innerHTML = '';
      setInsightKpis([]);
    }

    function getRestaurantOptions() {
      return (APP.businesses || []).map(b => ({
        id: String(b[bIdx.business_id] || ''),
        name: String(b[bIdx.display_name] || ''),
        city: String(b[bIdx.city] || ''),
        address: String(b[bIdx.address] || ''),
        zip: String(b[bIdx.zip] || ''),
        latestRating: String(b[bIdx.latest_rating] || ''),
        latestOfficialRating: String(b[bIdx.latest_official_rating] || ''),
        latestDate: String(b[bIdx.latest_date] || ''),
        latestScore: num(b[bIdx.latest_score]),
        inspections: Number(b[bIdx.inspection_count] || 0),
        riskCategory: String(b[bIdx.latest_risk_level] || ''),
        riskCategoryNum: String(b[bIdx.latest_risk_level_num] || ''),
        avgRedPointsUsed: num(b[bIdx.latest_avg_red_points]),
        routineInspectionsUsed: num(b[bIdx.latest_rating_routine_n]),
        cityScope: String(b[bIdx.city_scope] || ''),
        cityCleaningReason: String(b[bIdx.city_cleaning_reason] || '')
      })).filter(r => r.id);
    }

    function getEventsForBusiness(id) {
      return ((APP.events_by_business || {})[id] || []).map(row => ({
        eventId: String(row[eIdx.event_id] || ''),
        date: String(row[eIdx.date] || ''),
        type: String(row[eIdx.type] || ''),
        result: String(row[eIdx.result] || ''),
        score: num(row[eIdx.score]),
        rating: String(row[eIdx.rating] || ''),
        red: num(row[eIdx.red]) || 0,
        blue: num(row[eIdx.blue]) || 0,
        violations: num(row[eIdx.violations]) || 0
      }));
    }

    function getViolationsForBusiness(id) {
      return ((APP.violations_by_business || {})[id] || []).map(row => ({
        eventId: String(row[vIdx.event_id] || ''),
        date: String(row[vIdx.date] || ''),
        type: String(row[vIdx.type] || ''),
        code: String(row[vIdx.code] || ''),
        points: num(row[vIdx.points]) || 0,
        violation: String(row[vIdx.violation] || ''),
        priority: String(row[vIdx.priority] || ''),
        category: String(row[vIdx.category] || ''),
        remediation: String(row[vIdx.remediation_summary] || '')
      }));
    }

    function getBusinessMetricsRows() {
      if (_businessMetricsCache) return _businessMetricsCache;
      const restaurants = getRestaurantOptions();
      const rows = [];
      for (const r of restaurants) {
        const events = getEventsForBusiness(r.id);
        if (!events.length) continue;
        let highCount = 0;
        let scoreSum = 0;
        let scoreN = 0;
        let redSum = 0;
        let blueSum = 0;
        for (const ev of events) {
          const isHigh = ev.rating === 'Needs to Improve' || (Number(ev.red || 0) >= 25);
          if (isHigh) highCount += 1;
          if (Number.isFinite(ev.score)) {
            scoreSum += ev.score;
            scoreN += 1;
          }
          redSum += Number(ev.red || 0);
          blueSum += Number(ev.blue || 0);
        }
        const count = events.length;
        const highRate = count ? highCount / count : 0;
        const avgScore = scoreN ? scoreSum / scoreN : null;
        const avgRed = count ? redSum / count : 0;
        const avgBlue = count ? blueSum / count : 0;
        const safetyScore = (1 - highRate) * 100 - avgRed * 1.2 - avgBlue * 0.35;
        rows.push({
          id: r.id,
          restaurant: r.name,
          city: r.city,
          address: r.address,
          zip: r.zip,
          inspections: count,
          latestRating: r.latestRating,
          latestDate: r.latestDate,
          latestScore: r.latestScore,
          highRiskRate: highRate,
          avgScore,
          avgRed,
          avgBlue,
          safetyScore
        });
      }
      _businessMetricsCache = rows;
      return rows;
    }

    function getEventsFlat() {
      if (_eventsFlatCache) return _eventsFlatCache;
      const businessById = new Map(getRestaurantOptions().map(r => [r.id, r]));
      const out = [];
      for (const [id, rows] of Object.entries(APP.events_by_business || {})) {
        const b = businessById.get(String(id)) || { name: id, city: '', address: '', zip: '' };
        for (const row of rows || []) {
          const date = String(row[eIdx.date] || '');
          const red = num(row[eIdx.red]) || 0;
          const rating = String(row[eIdx.rating] || '');
          out.push({
            businessId: String(id),
            restaurant: b.name,
            city: b.city,
            address: b.address,
            zip: b.zip,
            date,
            month: date ? date.slice(0, 7) : '',
            year: date ? date.slice(0, 4) : '',
            monthNum: date ? Number(date.slice(5, 7)) : null,
            score: num(row[eIdx.score]),
            red,
            blue: num(row[eIdx.blue]) || 0,
            violations: num(row[eIdx.violations]) || 0,
            rating,
            isHighRisk: (rating === 'Needs to Improve' || red >= 25) ? 1 : 0
          });
        }
      }
      _eventsFlatCache = out;
      return out;
    }

    function getViolationsFlat() {
      if (_violationsFlatCache) return _violationsFlatCache;
      const businessById = new Map(getRestaurantOptions().map(r => [r.id, r]));
      const out = [];
      for (const [id, rows] of Object.entries(APP.violations_by_business || {})) {
        const b = businessById.get(String(id)) || { name: id, city: '', address: '', zip: '' };
        for (const row of rows || []) {
          const date = String(row[vIdx.date] || '');
          out.push({
            businessId: String(id),
            restaurant: b.name,
            city: b.city,
            address: b.address,
            zip: b.zip,
            date,
            month: date ? date.slice(0, 7) : '',
            type: String(row[vIdx.type] || ''),
            code: String(row[vIdx.code] || ''),
            points: num(row[vIdx.points]) || 0,
            violation: String(row[vIdx.violation] || ''),
            priority: String(row[vIdx.priority] || ''),
            category: String(row[vIdx.category] || ''),
            remediation: String(row[vIdx.remediation_summary] || '')
          });
        }
      }
      _violationsFlatCache = out;
      return out;
    }

    function getQualityIssueCatalog() {
      const quality = APP.quality || {};
      const cols = quality.issue_catalog_columns || [];
      const rows = quality.issue_catalog_rows || [];
      const idx = IDX(cols);
      return rows.map(row => ({
        category: String(row[idx.category] || ''),
        issue: String(row[idx.issue] || ''),
        count: num(row[idx.count]),
        denominator: num(row[idx.denominator]),
        sharePct: num(row[idx.share_pct]),
        severity: String(row[idx.severity] || ''),
        why: String(row[idx.why_it_matters] || ''),
        action: String(row[idx.suggested_action] || ''),
        owner: String(row[idx.owner] || '')
      }));
    }

    function getQualityHistory() {
      const quality = APP.quality || {};
      const cols = quality.history_columns || [];
      const rows = quality.history_rows || [];
      const idx = IDX(cols);
      return rows.map(row => ({
        runId: String(row[idx.run_id] || ''),
        rowsRaw: num(row[idx.rows_raw]),
        cityOutsideRate: num(row[idx.city_outside_rate]),
        cityUnknownRate: num(row[idx.city_unknown_rate]),
        dateParseFailRate: num(row[idx.date_parse_fail_rate]),
        violationTruncatedRate: num(row[idx.violation_truncated_rate]),
        trueMismatchRate: num(row[idx.true_mismatch_rate])
      })).sort((a, b) => String(a.runId).localeCompare(String(b.runId)));
    }

    function getQualitySamplesByPrefix(prefixes) {
      const out = [];
      const samples = (APP.quality || {}).samples || {};
      for (const [groupName, records] of Object.entries(samples)) {
        const matched = (prefixes || []).some(prefix => String(groupName).startsWith(String(prefix)));
        if (!matched || !Array.isArray(records)) continue;
        for (const rec of records) {
          if (!rec || typeof rec !== 'object') continue;
          out.push([
            String(groupName || ''),
            String(rec.name || ''),
            String(rec.address || ''),
            String(rec.city || ''),
            String(rec.zip_code || ''),
            String(rec.inspection_date || ''),
            String(rec.latitude || ''),
            String(rec.longitude || '')
          ]);
        }
      }
      return out;
    }

    function fmtPctRate(value) {
      const n = Number(value);
      return Number.isFinite(n) ? (n * 100).toFixed(2) + '%' : 'N/A';
    }

    function roleQuestionById(role, qid) {
      return (insightQuestionMap[role] || []).find(q => q.id === qid);
    }

    function getSelectedRoleAndQuestion() {
      const role = state.insightRole || Object.keys(insightQuestionMap)[0];
      const qid = document.getElementById('insightQuestionSelect').value;
      return { role, qid };
    }

    function renderC1() {
      const months = Object.keys(APP.movement_by_month || {}).sort();
      if (!months.length) {
        document.getElementById('insightContent').innerHTML = '<div class="small">No monthly movement data.</div>';
        return;
      }
      const latestMonth = months[months.length - 1];
      const bucket = (APP.movement_by_month || {})[latestMonth] || { official: [], proxy: [] };
      let rows = bucket.official || [];
      let source = 'Dashboard calculated rating';
      if (!rows.length) {
        rows = bucket.proxy || [];
        source = 'Score-band proxy rating';
      }
      const improved = rows.filter(r => r[mIdx.change_type] === 'Improved');
      const declined = rows.filter(r => r[mIdx.change_type] === 'Declined');
      setInsightKpis([
        ['Month', latestMonth],
        ['Movement source', source],
        ['Improved restaurants', improved.length.toLocaleString()],
        ['Declined restaurants', declined.length.toLocaleString()]
      ]);
      document.getElementById('insightContent').innerHTML =
        '<div class="section-title">Improved Restaurants</div><div class="table-wrap fixed-320"><table id="insightC1Improved"></table></div>' +
        '<div class="section-title">Declined Restaurants</div><div class="table-wrap fixed-320"><table id="insightC1Declined"></table></div>';
      setTable(
        'insightC1Improved',
        ['Restaurant', 'City', 'Address', 'Previous date', 'Current date', 'Rating change'],
        improved
          .sort((a, b) => String(b[mIdx.current_date] || '').localeCompare(String(a[mIdx.current_date] || '')))
          .map(r => [r[mIdx.display_name], r[mIdx.city], r[mIdx.address], r[mIdx.prev_date], r[mIdx.current_date], r[mIdx.transition]]),
        'No improvements.'
      );
      setTable(
        'insightC1Declined',
        ['Restaurant', 'City', 'Address', 'Previous date', 'Current date', 'Rating change'],
        declined
          .sort((a, b) => String(b[mIdx.current_date] || '').localeCompare(String(a[mIdx.current_date] || '')))
          .map(r => [r[mIdx.display_name], r[mIdx.city], r[mIdx.address], r[mIdx.prev_date], r[mIdx.current_date], r[mIdx.transition]]),
        'No declines.'
      );
    }

    function renderC2() {
      const metrics = getBusinessMetricsRows().filter(r => r.inspections >= 6);
      const citySet = Array.from(new Set(metrics.map(r => r.city).filter(Boolean))).sort();
      document.getElementById('insightControls').innerHTML =
        '<div class="field"><label>Area (city)</label><select id="c2City"><option>All</option>' +
        citySet.map(c => '<option>' + esc(c) + '</option>').join('') +
        '</select></div>' +
        '<div class="field"><label>Min inspections</label><input id="c2MinInspections" type="number" min="1" max="50" value="6" /></div>';

      const render = () => {
        const city = document.getElementById('c2City').value;
        const minIns = Math.max(1, Number(document.getElementById('c2MinInspections').value || 6));
        let rows = metrics.filter(r => r.inspections >= minIns);
        if (city !== 'All') rows = rows.filter(r => r.city === city);
        rows = rows.sort((a, b) => (b.safetyScore - a.safetyScore) || (b.inspections - a.inspections));
        setInsightKpis([
          ['Restaurants shown', rows.length.toLocaleString()],
          ['Area', city],
          ['Min inspections', String(minIns)],
          ['Top safety score', rows.length ? rows[0].safetyScore.toFixed(1) : 'N/A']
        ]);
        document.getElementById('insightContent').innerHTML =
          '<div class="table-wrap"><table id="insightC2Table"></table></div><div id="insightC2Bars"></div>';
        setTable(
          'insightC2Table',
          ['Restaurant', 'City', 'Address', 'Inspections', 'High-risk rate', 'Avg red points', 'Latest rating', 'Safety score'],
          rows.slice(0, 25).map(r => [
            r.restaurant,
            r.city,
            r.address,
            r.inspections,
            (r.highRiskRate * 100).toFixed(1) + '%',
            r.avgRed.toFixed(2),
            r.latestRating,
            r.safetyScore.toFixed(2)
          ]),
          'No restaurants under current filter.'
        );
        renderBarRows('insightC2Bars', rows.slice(0, 12).map(r => [r.restaurant, r.safetyScore]), 0, 1, v => Number(v || 0).toFixed(1));
      };

      document.getElementById('c2City').addEventListener('change', render);
      document.getElementById('c2MinInspections').addEventListener('input', render);
      render();
    }

    function renderC3() {
      const all = getViolationsFlat();
      document.getElementById('insightControls').innerHTML =
        '<div class="field"><label>Recent window (days)</label><select id="c3Days">' +
        '<option value="30">30</option><option value="60">60</option><option value="90" selected>90</option><option value="180">180</option>' +
        '</select></div>';
      const render = () => {
        const days = Number(document.getElementById('c3Days').value || 90);
        const maxDate = all.reduce((acc, r) => (r.date > acc ? r.date : acc), '');
        if (!maxDate) {
          document.getElementById('insightContent').innerHTML = '<div class="small">No violation date data.</div>';
          return;
        }
        const cutoff = new Date(maxDate + 'T00:00:00Z');
        cutoff.setUTCDate(cutoff.getUTCDate() - days);
        const map = new Map();
        for (const row of all) {
          if (String(row.type || '').toUpperCase() !== 'RED') continue;
          if (!row.date) continue;
          const d = new Date(row.date + 'T00:00:00Z');
          if (d < cutoff) continue;
          const key = row.businessId;
          const prev = map.get(key) || { restaurant: row.restaurant, city: row.city, address: row.address, redRecords: 0, totalRedPoints: 0, latestDate: '' };
          prev.redRecords += 1;
          prev.totalRedPoints += Number(row.points || 0);
          if (row.date > prev.latestDate) prev.latestDate = row.date;
          map.set(key, prev);
        }
        const rows = Array.from(map.values()).sort((a, b) =>
          (b.totalRedPoints - a.totalRedPoints) || (b.redRecords - a.redRecords)
        );
        setInsightKpis([
          ['Window', days + ' days'],
          ['Restaurants with red issues', rows.length.toLocaleString()],
          ['Top total red points', rows.length ? String(rows[0].totalRedPoints.toFixed(0)) : '0'],
          ['Most recent date in window', maxDate]
        ]);
        document.getElementById('insightContent').innerHTML = '<div class="table-wrap"><table id="insightC3Table"></table></div>';
        setTable(
          'insightC3Table',
          ['Restaurant', 'City', 'Address', 'Red records', 'Total red points', 'Most recent red date'],
          rows.slice(0, 25).map(r => [r.restaurant, r.city, r.address, r.redRecords, r.totalRedPoints.toFixed(0), r.latestDate]),
          'No red violations in selected window.'
        );
      };
      document.getElementById('c3Days').addEventListener('change', render);
      render();
    }

    function renderC4OrO3(prefix) {
      const metrics = getBusinessMetricsRows();
      const opts = metrics
        .slice()
        .sort((a, b) => String(a.restaurant).localeCompare(String(b.restaurant)))
        .map(r => '<option value="' + esc(r.id) + '">' + esc(r.restaurant + ' | ' + r.address + ' | ' + r.city + ' (' + r.id + ')') + '</option>')
        .join('');
      document.getElementById('insightControls').innerHTML =
        '<div class="field" style="grid-column:1 / span 2;"><label>Select restaurant</label><select id="' + prefix + 'Restaurant">' + opts + '</select></div>';
      const render = () => {
        const id = document.getElementById(prefix + 'Restaurant').value;
        const selected = metrics.find(r => r.id === id);
        if (!selected) {
          document.getElementById('insightContent').innerHTML = '<div class="small">Restaurant not found.</div>';
          return;
        }
        const peers = metrics.filter(r => r.city === selected.city && r.inspections >= 3);
        if (!peers.length) {
          document.getElementById('insightContent').innerHTML = '<div class="small">No peer group in this city.</div>';
          return;
        }
        const saferScorePct = peers.filter(r => Number(r.avgScore || 0) >= Number(selected.avgScore || 0)).length / peers.length;
        const saferRedPct = peers.filter(r => Number(r.avgRed || 0) >= Number(selected.avgRed || 0)).length / peers.length;
        const saferRiskPct = peers.filter(r => Number(r.highRiskRate || 0) >= Number(selected.highRiskRate || 0)).length / peers.length;
        setInsightKpis([
          ['City', selected.city || 'N/A'],
          ['Peer restaurants', peers.length.toLocaleString()],
          ['Safer than peers (score)', (saferScorePct * 100).toFixed(1) + '%'],
          ['Safer than peers (risk rate)', (saferRiskPct * 100).toFixed(1) + '%']
        ]);
        const med = arr => {
          const v = arr.slice().map(Number).filter(Number.isFinite).sort((a, b) => a - b);
          if (!v.length) return NaN;
          const m = Math.floor(v.length / 2);
          return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
        };
        document.getElementById('insightContent').innerHTML =
          '<div class="table-wrap"><table id="' + prefix + 'CompareTable"></table></div>' +
          '<div class="section-title">Top safer peers in same city</div><div class="table-wrap fixed-320"><table id="' + prefix + 'PeersTable"></table></div>';
        setTable(
          prefix + 'CompareTable',
          ['Metric', 'Selected restaurant', 'City median'],
          [
            ['Average inspection score', Number(selected.avgScore || 0).toFixed(2), med(peers.map(r => r.avgScore)).toFixed(2)],
            ['Average red points', Number(selected.avgRed || 0).toFixed(2), med(peers.map(r => r.avgRed)).toFixed(2)],
            ['High-risk rate', (Number(selected.highRiskRate || 0) * 100).toFixed(1) + '%', (med(peers.map(r => r.highRiskRate)) * 100).toFixed(1) + '%'],
            ['Inspection count', String(selected.inspections || 0), med(peers.map(r => r.inspections)).toFixed(1)]
          ],
          'No comparison data.'
        );
        const topPeers = peers.slice().sort((a, b) => (b.safetyScore - a.safetyScore) || (b.inspections - a.inspections)).slice(0, 15);
        setTable(
          prefix + 'PeersTable',
          ['Restaurant', 'Address', 'Inspections', 'High-risk rate', 'Avg red points', 'Safety score'],
          topPeers.map(r => [r.restaurant, r.address, r.inspections, (r.highRiskRate * 100).toFixed(1) + '%', r.avgRed.toFixed(2), r.safetyScore.toFixed(2)]),
          'No peer table data.'
        );
      };
      document.getElementById(prefix + 'Restaurant').addEventListener('change', render);
      render();
    }

    function renderC5() {
      const all = getViolationsFlat();
      document.getElementById('insightControls').innerHTML =
        '<div class="field"><label>Recent window (days)</label><select id="c5Days">' +
        '<option value="30">30</option><option value="60">60</option><option value="90" selected>90</option><option value="180">180</option>' +
        '</select></div>';
      const render = () => {
        const days = Number(document.getElementById('c5Days').value || 90);
        const maxDate = all.reduce((acc, r) => (r.date > acc ? r.date : acc), '');
        if (!maxDate) {
          document.getElementById('insightContent').innerHTML = '<div class="small">No violation date data.</div>';
          return;
        }
        const cutoff = new Date(maxDate + 'T00:00:00Z');
        cutoff.setUTCDate(cutoff.getUTCDate() - days);
        const map = new Map();
        for (const row of all) {
          if (!row.code) continue;
          if (!row.date) continue;
          const d = new Date(row.date + 'T00:00:00Z');
          if (d < cutoff) continue;
          const key = [row.code, row.type, row.violation, row.category].join('|');
          const prev = map.get(key) || {
            code: row.code,
            type: row.type,
            violation: row.violation,
            category: row.category,
            occurrences: 0,
            restaurants: new Set(),
            pointsSum: 0
          };
          prev.occurrences += 1;
          prev.pointsSum += Number(row.points || 0);
          prev.restaurants.add(row.businessId);
          map.set(key, prev);
        }
        const rows = Array.from(map.values()).map(r => ({
          code: r.code,
          type: r.type,
          violation: r.violation,
          category: r.category,
          occurrences: r.occurrences,
          restaurantsAffected: r.restaurants.size,
          avgPoints: r.occurrences ? r.pointsSum / r.occurrences : 0
        })).sort((a, b) => (b.occurrences - a.occurrences) || (b.restaurantsAffected - a.restaurantsAffected));
        setInsightKpis([
          ['Window', days + ' days'],
          ['Unique issue patterns', rows.length.toLocaleString()],
          ['Top occurrence count', rows.length ? String(rows[0].occurrences) : '0'],
          ['Most recent date in window', maxDate]
        ]);
        document.getElementById('insightContent').innerHTML =
          '<div class="table-wrap"><table id="insightC5Table"></table></div><div id="insightC5Bars"></div>';
        setTable(
          'insightC5Table',
          ['Code', 'Type', 'Violation', 'Category', 'Occurrences', 'Restaurants affected', 'Avg points'],
          rows.slice(0, 25).map(r => [r.code, r.type, r.violation, r.category, r.occurrences, r.restaurantsAffected, r.avgPoints.toFixed(2)]),
          'No coded issues in selected window.'
        );
        renderBarRows('insightC5Bars', rows.slice(0, 12).map(r => [r.code + '-' + r.type, r.occurrences]), 0, 1, v => String(Number(v || 0).toFixed(0)));
      };
      document.getElementById('c5Days').addEventListener('change', render);
      render();
    }

    function renderC6() {
      const trend = APP.monthly_trend_rows || [];
      const trendRows = trend.slice(-24);
      setInsightKpis([
        ['Months shown', trendRows.length.toLocaleString()],
        ['Latest month', trendRows.length ? String(trendRows[trendRows.length - 1][0] || '') : 'N/A'],
        ['Latest high-risk rate', trendRows.length ? (Number(trendRows[trendRows.length - 1][3] || 0) * 100).toFixed(1) + '%' : 'N/A'],
        ['Latest average score', trendRows.length ? Number(trendRows[trendRows.length - 1][4] || 0).toFixed(1) : 'N/A']
      ]);
      document.getElementById('insightContent').innerHTML =
        '<div class="section-title">Monthly High-Risk Rate Trend</div><div id="insightC6TrendBars"></div>' +
        '<div class="section-title">Monthly Average Score Trend</div><div id="insightC6ScoreBars"></div>' +
        '<div class="table-wrap"><table id="insightC6Table"></table></div>';
      document.getElementById('insightC6TrendBars').innerHTML = trendRows.length
        ? trendRows.map(r => {
            const month = String(r[0] || '');
            const rate = Number(r[3] || 0);
            const width = Math.max(2, rate * 100);
            return '<div class="barrow"><div style="width:80px">' + esc(month) +
              '</div><div class="bar"><span style="width:' + width.toFixed(1) +
              '%"></span></div><div style="width:90px">' + esc((rate * 100).toFixed(1) + '%') + '</div></div>';
          }).join('')
        : '<div class="small">No trend rows.</div>';
      document.getElementById('insightC6ScoreBars').innerHTML = trendRows.length
        ? trendRows.map(r => {
            const month = String(r[0] || '');
            const score = Number(r[4] || 0);
            const width = Math.max(2, Math.min(100, (score / 60) * 100));
            return '<div class="barrow"><div style="width:80px">' + esc(month) +
              '</div><div class="bar"><span style="width:' + width.toFixed(1) +
              '%"></span></div><div style="width:90px">' + esc(score.toFixed(1)) + '</div></div>';
          }).join('')
        : '<div class="small">No score rows.</div>';
      setTable(
        'insightC6Table',
        ['Month', 'Inspections', 'High-risk inspections', 'High-risk rate', 'Average score'],
        trendRows.map(r => [r[0], r[1], r[2], (Number(r[3] || 0) * 100).toFixed(1) + '%', Number(r[4] || 0).toFixed(1)]),
        'No monthly trend table.'
      );
    }

    function renderO1() {
      const owner = APP.owner || {};
      const rows = owner.ranking_rows || [];
      setInsightKpis([
        ['Rows shown', Math.min(25, rows.length).toLocaleString()],
        ['Unique code-type patterns', rows.length.toLocaleString()],
        ['Top priority', rows.length ? String(rows[0][5] || '') : 'N/A'],
        ['Top occurrences', rows.length ? String(rows[0][2] || 0) : '0']
      ]);
      document.getElementById('insightContent').innerHTML =
        '<div class="table-wrap"><table id="insightO1Table"></table></div>' +
        '<div class="section-title">Top codes by occurrence</div><div id="insightO1TopCodes"></div>' +
        '<div class="section-title">Priority distribution</div><div id="insightO1Priority"></div>';
      setTable(
        'insightO1Table',
        ['Code', 'Type', 'Occurrences', 'Restaurants affected', 'Avg points', 'Priority', 'Category', 'Remediation summary'],
        rows.slice(0, 25).map(r => [r[0], r[1], r[2], r[3], Number(r[4] || 0).toFixed(2), r[5], r[6], r[7]]),
        'No violation ranking rows.'
      );
      renderBarRows('insightO1TopCodes', (owner.top_code_rows || []).map(r => [r[0], Number(r[1] || 0)]), 0, 1, v => String(Number(v || 0).toFixed(0)));
      renderBarRows('insightO1Priority', (owner.priority_rows || []).map(r => [r[0], Number(r[1] || 0)]), 0, 1, v => String(Number(v || 0).toFixed(0)));
    }

    function renderO2() {
      const metrics = getBusinessMetricsRows();
      const opts = metrics
        .slice()
        .sort((a, b) => String(a.restaurant).localeCompare(String(b.restaurant)))
        .map(r => '<option value="' + esc(r.id) + '">' + esc(r.restaurant + ' | ' + r.address + ' | ' + r.city + ' (' + r.id + ')') + '</option>')
        .join('');
      document.getElementById('insightControls').innerHTML =
        '<div class="field" style="grid-column:1 / span 2;"><label>Select restaurant</label><select id="o2Restaurant">' + opts + '</select></div>';
      const render = () => {
        const id = document.getElementById('o2Restaurant').value;
        const rows = getViolationsForBusiness(id).filter(v => v.code);
        if (!rows.length) {
          setInsightKpis([]);
          document.getElementById('insightContent').innerHTML = '<div class="small">No coded violations for selected restaurant.</div>';
          return;
        }
        const monthCodeMap = new Map();
        const codeCounts = new Map();
        for (const v of rows) {
          const month = v.date ? v.date.slice(0, 7) : '';
          const code = v.code;
          if (!month || !code) continue;
          const key = code + '|' + month;
          monthCodeMap.set(key, (monthCodeMap.get(key) || 0) + 1);
          codeCounts.set(code, (codeCounts.get(code) || 0) + 1);
        }
        const topCodes = Array.from(codeCounts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 12).map(x => x[0]);
        const months = Array.from(new Set(rows.map(v => v.date ? v.date.slice(0, 7) : '').filter(Boolean))).sort().slice(-12);
        const tableRows = topCodes.map(code => {
          const values = months.map(month => monthCodeMap.get(code + '|' + month) || 0);
          return [code, ...values];
        });
        const repeated = topCodes.map(code => {
          const monthsWithIssue = months.filter(month => (monthCodeMap.get(code + '|' + month) || 0) > 0).length;
          return { code, monthsWithIssue, total: codeCounts.get(code) || 0 };
        }).filter(x => x.monthsWithIssue >= 3).sort((a, b) => (b.monthsWithIssue - a.monthsWithIssue) || (b.total - a.total));

        setInsightKpis([
          ['Months shown', months.length.toLocaleString()],
          ['Top codes tracked', topCodes.length.toLocaleString()],
          ['Rows in table', tableRows.length.toLocaleString()],
          ['Repeated (3+ months)', repeated.length.toLocaleString()]
        ]);
        document.getElementById('insightContent').innerHTML =
          '<div class="table-wrap"><table id="insightO2HeatTable"></table></div>' +
          '<div class="section-title">Repeated issues (3+ months)</div><div class="table-wrap"><table id="insightO2RepeatedTable"></table></div>';
        setTable('insightO2HeatTable', ['Code', ...months], tableRows, 'No monthly repeat table.');
        setTable(
          'insightO2RepeatedTable',
          ['Code', 'Months with issue', 'Total records'],
          repeated.map(r => [r.code, r.monthsWithIssue, r.total]),
          'No code repeated in 3+ months.'
        );
      };
      document.getElementById('o2Restaurant').addEventListener('change', render);
      render();
    }

    function renderO4() {
      const metrics = getBusinessMetricsRows();
      const opts = metrics
        .slice()
        .sort((a, b) => String(a.restaurant).localeCompare(String(b.restaurant)))
        .map(r => '<option value="' + esc(r.id) + '">' + esc(r.restaurant + ' | ' + r.address + ' | ' + r.city + ' (' + r.id + ')') + '</option>')
        .join('');
      document.getElementById('insightControls').innerHTML =
        '<div class="field" style="grid-column:1 / span 2;"><label>Select restaurant</label><select id="o4Restaurant">' + opts + '</select></div>';
      const rank = { 'Excellent': 1, 'Good': 2, 'Okay': 3, 'Needs to Improve': 4 };
      const render = () => {
        const id = document.getElementById('o4Restaurant').value;
        const events = getEventsForBusiness(id).slice().sort((a, b) => String(a.date).localeCompare(String(b.date)));
        if (events.length < 2) {
          setInsightKpis([]);
          document.getElementById('insightContent').innerHTML = '<div class="small">At least two inspections are required.</div>';
          return;
        }
        const drops = [];
        for (let i = 1; i < events.length; i += 1) {
          const prev = events[i - 1];
          const cur = events[i];
          const prevRank = rank[prev.rating] || null;
          const curRank = rank[cur.rating] || null;
          if (prevRank == null || curRank == null) continue;
          if (curRank > prevRank) {
            drops.push({
              date: cur.date,
              change: String(prev.rating || '') + ' -> ' + String(cur.rating || ''),
              prevScore: prev.score,
              curScore: cur.score,
              scoreDelta: (Number(cur.score || 0) - Number(prev.score || 0)),
              prevRed: prev.red,
              curRed: cur.red,
              redDelta: (Number(cur.red || 0) - Number(prev.red || 0)),
              prevBlue: prev.blue,
              curBlue: cur.blue,
              blueDelta: (Number(cur.blue || 0) - Number(prev.blue || 0)),
              prevViol: prev.violations,
              curViol: cur.violations,
              violDelta: (Number(cur.violations || 0) - Number(prev.violations || 0))
            });
          }
        }
        setInsightKpis([
          ['Inspection rows', events.length.toLocaleString()],
          ['Detected drops', drops.length.toLocaleString()],
          ['Latest inspection', events.length ? events[events.length - 1].date : 'N/A'],
          ['Current rating', events.length ? events[events.length - 1].rating : 'N/A']
        ]);
        document.getElementById('insightContent').innerHTML = '<div class="table-wrap"><table id="insightO4Table"></table></div>';
        setTable(
          'insightO4Table',
          ['Drop date', 'Rating change', 'Prev score', 'Current score', 'Score delta', 'Prev red', 'Current red', 'Red delta', 'Prev blue', 'Current blue', 'Blue delta', 'Prev violations', 'Current violations', 'Violations delta'],
          drops.map(d => [d.date, d.change, d.prevScore, d.curScore, d.scoreDelta.toFixed(1), d.prevRed, d.curRed, d.redDelta.toFixed(1), d.prevBlue, d.curBlue, d.blueDelta.toFixed(1), d.prevViol, d.curViol, d.violDelta.toFixed(1)]),
          'No rating drop detected for selected restaurant.'
        );
      };
      document.getElementById('o4Restaurant').addEventListener('change', render);
      render();
    }

    function renderO5() {
      const all = getViolationsFlat();
      const cities = Array.from(new Set(all.map(r => r.city).filter(Boolean))).sort();
      document.getElementById('insightControls').innerHTML =
        '<div class="field"><label>Area (city)</label><select id="o5City"><option>All</option>' +
        cities.map(c => '<option>' + esc(c) + '</option>').join('') +
        '</select></div>';
      const render = () => {
        const city = document.getElementById('o5City').value;
        let rows = all.slice();
        if (city !== 'All') rows = rows.filter(r => r.city === city);
        const maxDate = rows.reduce((acc, r) => (r.date > acc ? r.date : acc), '');
        if (!maxDate) {
          document.getElementById('insightContent').innerHTML = '<div class="small">No violation dates for selected area.</div>';
          return;
        }
        const recentEnd = new Date(maxDate + 'T00:00:00Z');
        const recentStart = new Date(recentEnd.getTime());
        recentStart.setUTCDate(recentStart.getUTCDate() - 90);
        const priorStart = new Date(recentStart.getTime());
        priorStart.setUTCDate(priorStart.getUTCDate() - 90);

        const recentCount = new Map();
        const priorCount = new Map();
        for (const r of rows) {
          const cat = String(r.category || '');
          if (!cat) continue;
          if (!r.date) continue;
          const d = new Date(r.date + 'T00:00:00Z');
          if (d >= recentStart && d <= recentEnd) recentCount.set(cat, (recentCount.get(cat) || 0) + 1);
          else if (d >= priorStart && d < recentStart) priorCount.set(cat, (priorCount.get(cat) || 0) + 1);
        }
        const cats = new Set([...recentCount.keys(), ...priorCount.keys()]);
        const growth = Array.from(cats).map(cat => {
          const recent = recentCount.get(cat) || 0;
          const prior = priorCount.get(cat) || 0;
          const delta = recent - prior;
          const growthRate = prior > 0 ? delta / prior : null;
          return { cat, recent, prior, delta, growthRate };
        }).sort((a, b) => (b.delta - a.delta) || (b.recent - a.recent));

        setInsightKpis([
          ['Area', city],
          ['Categories tracked', growth.length.toLocaleString()],
          ['Top increase', growth.length ? String(growth[0].delta) : '0'],
          ['Recent window end', maxDate]
        ]);
        document.getElementById('insightContent').innerHTML = '<div class="table-wrap"><table id="insightO5Table"></table></div><div id="insightO5Bars"></div>';
        setTable(
          'insightO5Table',
          ['Category', 'Recent 90d', 'Previous 90d', 'Change', 'Growth rate'],
          growth.map(r => [r.cat, r.recent, r.prior, r.delta, r.growthRate == null ? 'N/A' : (r.growthRate * 100).toFixed(1) + '%']),
          'No category trend data.'
        );
        renderBarRows('insightO5Bars', growth.slice(0, 12).map(r => [r.cat, r.delta]), 0, 1, v => String(Number(v || 0).toFixed(0)));
      };
      document.getElementById('o5City').addEventListener('change', render);
      render();
    }

    function renderO6() {
      const metrics = getBusinessMetricsRows();
      const opts = metrics
        .slice()
        .sort((a, b) => String(a.restaurant).localeCompare(String(b.restaurant)))
        .map(r => '<option value="' + esc(r.id) + '">' + esc(r.restaurant + ' | ' + r.address + ' | ' + r.city + ' (' + r.id + ')') + '</option>')
        .join('');
      document.getElementById('insightControls').innerHTML =
        '<div class="field"><label>Select restaurant</label><select id="o6Restaurant">' + opts + '</select></div>' +
        '<div class="field"><label>Lookback inspections</label><input id="o6Lookback" type="number" min="2" max="20" value="8" /></div>';
      const render = () => {
        const id = document.getElementById('o6Restaurant').value;
        const lookback = Math.max(2, Number(document.getElementById('o6Lookback').value || 8));
        const events = getEventsForBusiness(id).slice().sort((a, b) => String(a.date).localeCompare(String(b.date)));
        if (events.length < 2) {
          setInsightKpis([]);
          document.getElementById('insightContent').innerHTML = '<div class="small">At least two inspections are required.</div>';
          return;
        }
        const view = events.slice(-Math.min(lookback, events.length));
        const first = view[0];
        const last = view[view.length - 1];
        const highRiskCount = view.filter(v => v.rating === 'Needs to Improve' || Number(v.red || 0) >= 25).length;
        setInsightKpis([
          ['Inspections in window', view.length.toLocaleString()],
          ['Score change', (Number(last.score || 0) - Number(first.score || 0)).toFixed(1)],
          ['Red-point change', (Number(last.red || 0) - Number(first.red || 0)).toFixed(1)],
          ['High-risk count in window', highRiskCount.toLocaleString()]
        ]);
        document.getElementById('insightContent').innerHTML =
          '<div class="table-wrap"><table id="insightO6Table"></table></div>' +
          '<div class="section-title">Score trend</div><div id="insightO6ScoreBars"></div>' +
          '<div class="section-title">Red-point trend</div><div id="insightO6RedBars"></div>';
        setTable(
          'insightO6Table',
          ['Date', 'Score', 'Red points', 'Blue points', 'Violations', 'Rating'],
          view.map(v => [v.date, v.score, v.red, v.blue, v.violations, v.rating]),
          'No inspection rows.'
        );
        renderBarRows('insightO6ScoreBars', view.map(v => [v.date, Number(v.score || 0)]), 0, 1, v => Number(v || 0).toFixed(1));
        renderBarRows('insightO6RedBars', view.map(v => [v.date, Number(v.red || 0)]), 0, 1, v => Number(v || 0).toFixed(1));
      };
      document.getElementById('o6Restaurant').addEventListener('change', render);
      document.getElementById('o6Lookback').addEventListener('input', render);
      render();
    }

    function renderR1() {
      const metrics = getBusinessMetricsRows();
      document.getElementById('insightControls').innerHTML =
        '<div class="field"><label>Area level</label><select id="r1Level"><option>City</option><option>Zip code</option></select></div>' +
        '<div class="field"><label>Min inspections</label><input id="r1MinIns" type="number" min="5" max="500" value="30" /></div>';
      const render = () => {
        const level = document.getElementById('r1Level').value;
        const minIns = Math.max(5, Number(document.getElementById('r1MinIns').value || 30));
        const keyName = level === 'City' ? 'city' : 'zip';
        const bucket = new Map();
        for (const r of metrics) {
          const key = String(r[keyName] || '').trim();
          if (!key) continue;
          const prev = bucket.get(key) || { area: key, inspections: 0, restaurants: 0, highRiskInspections: 0, redTotal: 0 };
          prev.inspections += Number(r.inspections || 0);
          prev.restaurants += 1;
          prev.highRiskInspections += Number(r.highRiskRate || 0) * Number(r.inspections || 0);
          prev.redTotal += Number(r.avgRed || 0) * Number(r.inspections || 0);
          bucket.set(key, prev);
        }
        let rows = Array.from(bucket.values()).map(r => ({
          area: r.area,
          inspections: r.inspections,
          restaurants: r.restaurants,
          highRiskInspections: r.highRiskInspections,
          highRiskRate: r.inspections ? r.highRiskInspections / r.inspections : 0,
          avgRedPoints: r.inspections ? r.redTotal / r.inspections : 0
        }));
        rows = rows.filter(r => r.inspections >= minIns).sort((a, b) =>
          (b.highRiskRate - a.highRiskRate) || (b.highRiskInspections - a.highRiskInspections)
        );
        setInsightKpis([
          ['Area level', level],
          ['Rows shown', rows.length.toLocaleString()],
          ['Min inspections', String(minIns)],
          ['Highest high-risk rate', rows.length ? (rows[0].highRiskRate * 100).toFixed(1) + '%' : 'N/A']
        ]);
        document.getElementById('insightContent').innerHTML =
          '<div class="table-wrap"><table id="insightR1Table"></table></div><div id="insightR1Bars"></div>';
        setTable(
          'insightR1Table',
          [level, 'Inspections', 'Restaurants', 'High-risk inspections', 'High-risk rate', 'Avg red points'],
          rows.slice(0, 50).map(r => [r.area, r.inspections.toFixed(0), r.restaurants.toFixed(0), r.highRiskInspections.toFixed(0), (r.highRiskRate * 100).toFixed(1) + '%', r.avgRedPoints.toFixed(2)]),
          'No area rows pass current threshold.'
        );
        renderBarRows('insightR1Bars', rows.slice(0, 15).map(r => [r.area, r.highRiskRate * 100]), 0, 1, v => Number(v || 0).toFixed(1) + '%');
      };
      document.getElementById('r1Level').addEventListener('change', render);
      document.getElementById('r1MinIns').addEventListener('input', render);
      render();
    }

    function renderR2() {
      const events = getEventsFlat().filter(r => r.city && r.month);
      const cityMonth = new Map();
      for (const e of events) {
        const key = e.city + '|' + e.month;
        const prev = cityMonth.get(key) || { city: e.city, month: e.month, inspections: 0, high: 0 };
        prev.inspections += 1;
        prev.high += Number(e.isHighRisk || 0);
        cityMonth.set(key, prev);
      }
      const perCity = new Map();
      for (const row of cityMonth.values()) {
        const prev = perCity.get(row.city) || [];
        prev.push({
          month: row.month,
          inspections: row.inspections,
          highRiskRate: row.inspections ? row.high / row.inspections : 0
        });
        perCity.set(row.city, prev);
      }
      const slopes = [];
      for (const [city, rows] of perCity.entries()) {
        const sorted = rows.slice().sort((a, b) => String(a.month).localeCompare(String(b.month)));
        if (sorted.length < 6) continue;
        const x = sorted.map((_, i) => i);
        const y = sorted.map(r => Number(r.highRiskRate || 0));
        const n = x.length;
        const xMean = x.reduce((a, b) => a + b, 0) / n;
        const yMean = y.reduce((a, b) => a + b, 0) / n;
        let nume = 0;
        let deno = 0;
        for (let i = 0; i < n; i += 1) {
          nume += (x[i] - xMean) * (y[i] - yMean);
          deno += (x[i] - xMean) * (x[i] - xMean);
        }
        const slope = deno === 0 ? 0 : nume / deno;
        slopes.push({
          city,
          monthsUsed: sorted.length,
          slope,
          latestRate: sorted[sorted.length - 1].highRiskRate,
          totalInspections: sorted.reduce((a, b) => a + Number(b.inspections || 0), 0)
        });
      }
      slopes.sort((a, b) => (b.slope - a.slope) || (b.latestRate - a.latestRate));
      setInsightKpis([
        ['Cities with trends', slopes.length.toLocaleString()],
        ['Top slope', slopes.length ? slopes[0].slope.toFixed(4) : 'N/A'],
        ['Top city latest rate', slopes.length ? (slopes[0].latestRate * 100).toFixed(1) + '%' : 'N/A'],
        ['Minimum months required', '6']
      ]);
      document.getElementById('insightContent').innerHTML =
        '<div class="table-wrap"><table id="insightR2Table"></table></div><div id="insightR2Bars"></div>';
      setTable(
        'insightR2Table',
        ['City', 'Months used', 'Trend slope (per month)', 'Latest high-risk rate', 'Total inspections'],
        slopes.slice(0, 30).map(r => [r.city, r.monthsUsed, r.slope.toFixed(4), (r.latestRate * 100).toFixed(1) + '%', r.totalInspections]),
        'No city trend slope rows.'
      );
      renderBarRows('insightR2Bars', slopes.slice(0, 15).map(r => [r.city, r.slope]), 0, 1, v => Number(v || 0).toFixed(4));
    }

    function renderR3() {
      const restaurants = getRestaurantOptions();
      const infoMap = new Map(restaurants.map(r => [r.id, r]));
      const out = [];
      for (const [id, rows] of Object.entries(APP.events_by_business || {})) {
        const sorted = (rows || []).slice().sort((a, b) => String(b[eIdx.date] || '').localeCompare(String(a[eIdx.date] || '')));
        if (!sorted.length) continue;
        let streak = 0;
        let highCount = 0;
        for (const row of sorted) {
          const rating = String(row[eIdx.rating] || '');
          const red = Number(num(row[eIdx.red]) || 0);
          const isHigh = rating === 'Needs to Improve' || red >= 25;
          if (isHigh) highCount += 1;
          if (streak >= 0 && isHigh && streak === highCount - 1) streak += 1;
          else if (streak >= 0 && !isHigh && streak === highCount) streak = -999;
        }
        if (streak < 2) continue;
        const info = infoMap.get(String(id)) || { name: id, city: '', address: '', latestRating: '', latestDate: '' };
        out.push({
          restaurant: info.name,
          city: info.city,
          address: info.address,
          streak,
          overallRate: sorted.length ? highCount / sorted.length : 0,
          latestRating: info.latestRating,
          latestDate: info.latestDate
        });
      }
      out.sort((a, b) => (b.streak - a.streak) || (b.overallRate - a.overallRate));
      setInsightKpis([
        ['Restaurants flagged', out.length.toLocaleString()],
        ['Threshold', '2+ consecutive'],
        ['Top streak', out.length ? String(out[0].streak) : '0'],
        ['Top overall high-risk rate', out.length ? (out[0].overallRate * 100).toFixed(1) + '%' : 'N/A']
      ]);
      document.getElementById('insightContent').innerHTML = '<div class="table-wrap"><table id="insightR3Table"></table></div>';
      setTable(
        'insightR3Table',
        ['Restaurant', 'City', 'Address', 'Current consecutive high-risk count', 'Overall high-risk rate', 'Latest rating', 'Latest inspection'],
        out.slice(0, 50).map(r => [r.restaurant, r.city, r.address, r.streak, (r.overallRate * 100).toFixed(1) + '%', r.latestRating, r.latestDate]),
        'No restaurant reaches 2+ consecutive high-risk inspections.'
      );
    }

    function renderR4() {
      const cityRows = (APP.regulator || {}).city_rows || [];
      if (!cityRows.length) {
        document.getElementById('insightContent').innerHTML = '<div class="small">No city rows available.</div>';
        return;
      }
      const rows = cityRows.map(r => ({
        city: r[0],
        inspections: Number(r[1] || 0),
        restaurants: Number(r[2] || 0),
        highRiskInspections: Number(r[3] || 0),
        highRiskRate: Number(r[4] || 0),
        avgRedPoints: Number(r[5] || 0)
      }));
      const riskValues = rows.map(r => r.highRiskRate).sort((a, b) => a - b);
      const loadValues = rows.map(r => r.inspections).sort((a, b) => a - b);
      const mid = arr => arr.length ? arr[Math.floor(arr.length / 2)] : 0;
      const riskMedian = mid(riskValues);
      const loadMedian = mid(loadValues);
      rows.forEach(r => {
        const highRisk = r.highRiskRate >= riskMedian;
        const highLoad = r.inspections >= loadMedian;
        if (highRisk && highLoad) r.quadrant = 'High Risk / High Workload';
        else if (highRisk) r.quadrant = 'High Risk / Lower Workload';
        else if (highLoad) r.quadrant = 'Lower Risk / High Workload';
        else r.quadrant = 'Lower Risk / Lower Workload';
      });
      rows.sort((a, b) => (b.highRiskRate - a.highRiskRate) || (b.inspections - a.inspections));
      setInsightKpis([
        ['Cities shown', rows.length.toLocaleString()],
        ['Risk median', (riskMedian * 100).toFixed(1) + '%'],
        ['Workload median', loadMedian.toLocaleString()],
        ['Top city risk', rows.length ? (rows[0].highRiskRate * 100).toFixed(1) + '%' : 'N/A']
      ]);
      document.getElementById('insightContent').innerHTML = '<div class="table-wrap"><table id="insightR4Table"></table></div>';
      setTable(
        'insightR4Table',
        ['City', 'Inspections', 'Restaurants', 'High-risk inspections', 'High-risk rate', 'Avg red points', 'Quadrant'],
        rows.map(r => [r.city, r.inspections, r.restaurants, r.highRiskInspections, (r.highRiskRate * 100).toFixed(1) + '%', r.avgRedPoints.toFixed(2), r.quadrant]),
        'No city workload-risk rows.'
      );
    }

    function renderR5() {
      const events = getEventsFlat().filter(r => Number.isFinite(r.monthNum));
      if (!events.length) {
        document.getElementById('insightContent').innerHTML = '<div class="small">No seasonal records found.</div>';
        return;
      }
      const monthMap = new Map();
      const yearMonthMap = new Map();
      for (const e of events) {
        const m = Number(e.monthNum);
        if (!Number.isFinite(m) || m < 1 || m > 12) continue;
        const mk = String(m);
        const prev = monthMap.get(mk) || { monthNum: m, inspections: 0, high: 0, scoreSum: 0, scoreN: 0 };
        prev.inspections += 1;
        prev.high += Number(e.isHighRisk || 0);
        if (Number.isFinite(e.score)) {
          prev.scoreSum += e.score;
          prev.scoreN += 1;
        }
        monthMap.set(mk, prev);
        const ymk = String(e.year) + '|' + mk;
        const ymPrev = yearMonthMap.get(ymk) || { year: String(e.year), monthNum: m, inspections: 0, high: 0 };
        ymPrev.inspections += 1;
        ymPrev.high += Number(e.isHighRisk || 0);
        yearMonthMap.set(ymk, ymPrev);
      }
      const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
      const monthRows = Array.from(monthMap.values())
        .map(r => ({
          monthNum: r.monthNum,
          month: monthNames[r.monthNum - 1],
          inspections: r.inspections,
          highRiskRate: r.inspections ? r.high / r.inspections : 0,
          avgScore: r.scoreN ? r.scoreSum / r.scoreN : 0
        }))
        .sort((a, b) => a.monthNum - b.monthNum);

      const years = Array.from(new Set(Array.from(yearMonthMap.values()).map(r => r.year))).sort();
      const heatRows = years.map(y => {
        const row = [y];
        for (let m = 1; m <= 12; m += 1) {
          const key = y + '|' + String(m);
          const v = yearMonthMap.get(key);
          row.push(v && v.inspections ? ((v.high / v.inspections) * 100).toFixed(1) + '%' : '');
        }
        return row;
      });

      setInsightKpis([
        ['Events used', events.length.toLocaleString()],
        ['Years covered', years.length.toLocaleString()],
        ['Highest month risk', monthRows.length ? monthRows.slice().sort((a, b) => b.highRiskRate - a.highRiskRate)[0].month : 'N/A'],
        ['Lowest month risk', monthRows.length ? monthRows.slice().sort((a, b) => a.highRiskRate - b.highRiskRate)[0].month : 'N/A']
      ]);
      document.getElementById('insightContent').innerHTML =
        '<div id="insightR5RiskBars"></div><div id="insightR5ScoreBars"></div>' +
        '<div class="table-wrap"><table id="insightR5MonthTable"></table></div>' +
        '<div class="section-title">Year x Month High-Risk Matrix</div><div class="table-wrap"><table id="insightR5HeatTable"></table></div>';
      renderBarRows('insightR5RiskBars', monthRows.map(r => [r.month, r.highRiskRate * 100]), 0, 1, v => Number(v || 0).toFixed(1) + '%');
      renderBarRows('insightR5ScoreBars', monthRows.map(r => [r.month, r.avgScore]), 0, 1, v => Number(v || 0).toFixed(1));
      setTable(
        'insightR5MonthTable',
        ['Month', 'Inspections', 'High-risk rate', 'Average score'],
        monthRows.map(r => [r.month, r.inspections, (r.highRiskRate * 100).toFixed(1) + '%', r.avgScore.toFixed(2)]),
        'No monthly seasonality rows.'
      );
      setTable(
        'insightR5HeatTable',
        ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        heatRows,
        'No year-month matrix rows.'
      );
    }

    function renderR6() {
      const restaurants = getRestaurantOptions();
      const events = getEventsFlat();
      const violations = getViolationsFlat();
      const missingCity = restaurants.filter(r => !String(r.city || '').trim()).length;
      const missingZip = restaurants.filter(r => !String(r.zip || '').trim()).length;
      const missingLatestDate = restaurants.filter(r => !String(r.latestDate || '').trim()).length;
      const ratingNA = restaurants.filter(r => String(r.latestRating || '').trim() === '' || String(r.latestRating || '').trim() === 'Rating not available').length;
      const missingEventDate = events.filter(e => !String(e.date || '').trim()).length;
      const missingScore = events.filter(e => !Number.isFinite(e.score)).length;
      const missingViolationIdentity = violations.filter(v => !String(v.code || '').trim() && !String(v.violation || '').trim()).length;
      const missingPriority = violations.filter(v => !String(v.priority || '').trim()).length;
      const missingRemediation = violations.filter(v => !String(v.remediation || '').trim()).length;

      const qualityRows = [
        ['Business profiles', 'Missing city', missingCity, restaurants.length],
        ['Business profiles', 'Missing zip', missingZip, restaurants.length],
        ['Business profiles', 'Missing latest inspection date', missingLatestDate, restaurants.length],
        ['Business profiles', 'Rating not available', ratingNA, restaurants.length],
        ['Events', 'Missing inspection date', missingEventDate, events.length],
        ['Events', 'Missing inspection score', missingScore, events.length],
        ['Violations', 'Missing violation identity (code+text)', missingViolationIdentity, violations.length],
        ['Violations', 'Missing priority', missingPriority, violations.length],
        ['Violations', 'Missing remediation summary', missingRemediation, violations.length]
      ].map(r => [
        r[0],
        r[1],
        r[2],
        r[3],
        r[3] > 0 ? ((r[2] / r[3]) * 100).toFixed(2) + '%' : 'N/A'
      ]).sort((a, b) => Number(String(b[4]).replace('%', '')) - Number(String(a[4]).replace('%', '')));

      setInsightKpis([
        ['Business profiles', restaurants.length.toLocaleString()],
        ['Event rows', events.length.toLocaleString()],
        ['Violation rows', violations.length.toLocaleString()],
        ['Top issue share', qualityRows.length ? qualityRows[0][4] : 'N/A']
      ]);
      document.getElementById('insightContent').innerHTML = '<div class="table-wrap"><table id="insightR6Table"></table></div>';
      setTable(
        'insightR6Table',
        ['Dataset', 'Issue', 'Rows affected', 'Denominator', 'Share'],
        qualityRows,
        'No quality rows.'
      );
    }

    function renderK1() {
      const rows = getRestaurantOptions()
        .filter(r =>
          String(r.latestOfficialRating || '').trim() &&
          String(r.latestRating || '').trim() &&
          String(r.latestOfficialRating || '') !== String(r.latestRating || '')
        )
        .sort((a, b) => String(b.latestDate || '').localeCompare(String(a.latestDate || '')) || String(a.name || '').localeCompare(String(b.name || '')));
      const patternMap = new Map();
      rows.forEach(r => {
        const key = String(r.latestRating || '') + ' || ' + String(r.latestOfficialRating || '');
        patternMap.set(key, (patternMap.get(key) || 0) + 1);
      });
      const patternRows = Array.from(patternMap.entries())
        .map(([key, count]) => {
          const [dashboard, official] = key.split(' || ');
          return [dashboard, official, String(count)];
        })
        .sort((a, b) => Number(b[2]) - Number(a[2]));
      const trueMismatchRows = rows.filter(r => {
        const dashboard = String(r.latestRating || '').trim();
        const official = String(r.latestOfficialRating || '').trim();
        if (!dashboard || !official) return false;
        if (dashboard === 'Rating not available' || official === 'Rating not available') return false;
        return dashboard !== official;
      });
      setInsightKpis([
        ['Restaurants with differences', rows.length.toLocaleString()],
        ['Mismatch patterns', patternRows.length.toLocaleString()],
        ['True mismatches (excluding N/A)', trueMismatchRows.length.toLocaleString()]
      ]);
      document.getElementById('insightContent').innerHTML =
        '<div class="table-wrap"><table id="insightK1PatternTable"></table></div>' +
        '<div class="table-wrap"><table id="insightK1Table"></table></div>';
      setTable(
        'insightK1PatternTable',
        ['Dashboard rating', 'Official county grade', 'Restaurants'],
        patternRows,
        'No mismatch patterns.'
      );
      setTable(
        'insightK1Table',
        ['Restaurant', 'City', 'Address', 'Latest inspection', 'Risk category', 'Official county grade', 'Dashboard rating', 'Avg red points used', 'Routine inspections used', 'City scope', 'City cleaning reason'],
        rows.map(r => [
          r.name,
          r.city,
          r.address,
          r.latestDate,
          r.riskCategory || '-',
          r.latestOfficialRating,
          r.latestRating,
          Number.isFinite(r.avgRedPointsUsed) ? r.avgRedPointsUsed.toFixed(2) : '-',
          Number.isFinite(r.routineInspectionsUsed) ? String(Math.round(r.routineInspectionsUsed)) : '-',
          r.cityScope || '-',
          r.cityCleaningReason || '-'
        ]),
        'No restaurants currently differ between official and dashboard grades.'
      );
    }

    function renderK2() {
      const qa = ((APP.quality || {}).audit_snapshot || {});
      const outsideRows = Object.entries(qa.top_known_outside_tokens || {})
        .map(([token, count]) => [token, Number(count || 0)])
        .sort((a, b) => b[1] - a[1]);
      const unknownRows = Object.entries(qa.top_unknown_city_tokens || {})
        .map(([token, count]) => [token, Number(count || 0)])
        .sort((a, b) => b[1] - a[1]);
      const sampleRows = getQualitySamplesByPrefix(['outside_', 'variant_']);

      setInsightKpis([
        ['Outside-city rows', Number(qa.city_known_outside_rows || 0).toLocaleString()],
        ['Unknown city-token rows', Number(qa.city_unknown_rows || 0).toLocaleString()],
        ['Geo out-of-bounds rows', Number(qa.geo_out_of_bounds_rows || 0).toLocaleString()],
        ['Distinct outside-city tokens', Number(qa.city_known_outside_distinct || 0).toLocaleString()]
      ]);
      document.getElementById('insightContent').innerHTML =
        '<div class="table-wrap"><table id="insightK2OutsideTable"></table></div>' +
        '<div class="table-wrap"><table id="insightK2UnknownTable"></table></div>' +
        '<div class="section-title">Sample records for scope/locality issues</div><div class="table-wrap"><table id="insightK2SampleTable"></table></div>';
      setTable(
        'insightK2OutsideTable',
        ['Outside-city token', 'Rows'],
        outsideRows,
        'No outside-city tokens.'
      );
      setTable(
        'insightK2UnknownTable',
        ['Unknown token', 'Rows'],
        unknownRows,
        'No unknown city tokens.'
      );
      setTable(
        'insightK2SampleTable',
        ['Issue group', 'Restaurant', 'Address', 'City token', 'Zip', 'Inspection date', 'Latitude', 'Longitude'],
        sampleRows,
        'No sample rows for scope/locality issues.'
      );
    }

    function renderK3() {
      const qa = ((APP.quality || {}).audit_snapshot || {});
      const restaurants = getRestaurantOptions();
      const reasonMap = new Map();
      const scopeMap = new Map();
      let correctedBusinesses = 0;
      let outsideScopeBusinesses = 0;
      for (const r of restaurants) {
        const reason = String(r.cityCleaningReason || '').trim() || '(empty)';
        reasonMap.set(reason, (reasonMap.get(reason) || 0) + 1);
        if (reason && reason !== 'exact') correctedBusinesses += 1;

        const scope = String(r.cityScope || '').trim() || '(empty)';
        scopeMap.set(scope, (scopeMap.get(scope) || 0) + 1);
        if (scope.toLowerCase().includes('outside')) outsideScopeBusinesses += 1;
      }
      const reasonRows = Array.from(reasonMap.entries())
        .map(([reason, count]) => [reason, count, restaurants.length ? ((count / restaurants.length) * 100).toFixed(2) + '%' : 'N/A'])
        .sort((a, b) => Number(String(b[1])) - Number(String(a[1])));
      const scopeRows = Array.from(scopeMap.entries())
        .map(([scope, count]) => [scope, count])
        .sort((a, b) => Number(String(b[1])) - Number(String(a[1])));

      setInsightKpis([
        ['City casing-inconsistent rows', Number(qa.city_case_issue_rows || 0).toLocaleString()],
        ['ZIPs with 3+ city variants', Number(qa.zip_with_3plus_city_variants || 0).toLocaleString()],
        ['Businesses with city correction', correctedBusinesses.toLocaleString()],
        ['Businesses outside scope', outsideScopeBusinesses.toLocaleString()]
      ]);
      document.getElementById('insightContent').innerHTML =
        '<div class="table-wrap"><table id="insightK3ReasonTable"></table></div>' +
        '<div class="table-wrap"><table id="insightK3ScopeTable"></table></div>';
      setTable(
        'insightK3ReasonTable',
        ['City cleaning reason', 'Businesses', 'Share'],
        reasonRows,
        'No city cleaning reason rows.'
      );
      setTable(
        'insightK3ScopeTable',
        ['City scope', 'Businesses'],
        scopeRows,
        'No city scope rows.'
      );
    }

    function renderK4() {
      const qa = ((APP.quality || {}).audit_snapshot || {});
      const issueRows = getQualityIssueCatalog()
        .filter(r => String(r.category || '') === 'Identifier Completeness')
        .map(r => [
          r.issue,
          Number.isFinite(r.count) ? Number(r.count).toLocaleString() : 'N/A',
          Number.isFinite(r.denominator) ? Number(r.denominator).toLocaleString() : 'N/A',
          Number.isFinite(r.sharePct) ? Number(r.sharePct).toFixed(2) + '%' : 'N/A',
          r.severity || '',
          r.action || ''
        ]);
      const sampleRows = getQualitySamplesByPrefix(['date_parse_fail']);

      setInsightKpis([
        ['Missing inspection_serial_num', Number(qa.missing_inspection_serial_rows || 0).toLocaleString()],
        ['Unparseable inspection_date rows', Number(qa.raw_date_parse_fail_rows || 0).toLocaleString()],
        ['Missing violation_record_id', Number(qa.missing_violation_record_id_rows || 0).toLocaleString()]
      ]);
      document.getElementById('insightContent').innerHTML =
        '<div class="table-wrap"><table id="insightK4IssueTable"></table></div>' +
        '<div class="section-title">Sample rows for date/key integrity issues</div><div class="table-wrap"><table id="insightK4SampleTable"></table></div>';
      setTable(
        'insightK4IssueTable',
        ['Issue', 'Rows affected', 'Denominator', 'Share', 'Severity', 'Suggested action'],
        issueRows,
        'No identifier-completeness issue rows.'
      );
      setTable(
        'insightK4SampleTable',
        ['Issue group', 'Restaurant', 'Address', 'City token', 'Zip', 'Inspection date', 'Latitude', 'Longitude'],
        sampleRows,
        'No sample rows for date/key integrity issues.'
      );
    }

    function renderK5() {
      const qa = ((APP.quality || {}).audit_snapshot || {});
      const issueRows = getQualityIssueCatalog()
        .filter(r => ['Semantic Completeness', 'Text Quality', 'Numeric Validity'].includes(String(r.category || '')))
        .map(r => [
          r.category,
          r.issue,
          Number.isFinite(r.count) ? Number(r.count).toLocaleString() : 'N/A',
          Number.isFinite(r.denominator) ? Number(r.denominator).toLocaleString() : 'N/A',
          Number.isFinite(r.sharePct) ? Number(r.sharePct).toFixed(2) + '%' : 'N/A',
          r.severity || '',
          r.action || ''
        ]);
      const sampleRows = getQualitySamplesByPrefix(['truncated_violation', 'grade_missing_with_risk_text', 'negative_score']);

      setInsightKpis([
        ['Missing violation_type rows', Number(qa.violation_type_missing_rows || 0).toLocaleString()],
        ['Missing official grade rows', Number(qa.grade_missing_rows || 0).toLocaleString()],
        ['Truncated violation-text rows', Number(qa.violation_desc_truncated_rows || 0).toLocaleString()],
        ['Negative-score rows', Number(qa.score_negative_rows || 0).toLocaleString()]
      ]);
      document.getElementById('insightContent').innerHTML =
        '<div class="table-wrap"><table id="insightK5IssueTable"></table></div>' +
        '<div class="section-title">Sample rows for semantic/text/numeric issues</div><div class="table-wrap"><table id="insightK5SampleTable"></table></div>';
      setTable(
        'insightK5IssueTable',
        ['Category', 'Issue', 'Rows affected', 'Denominator', 'Share', 'Severity', 'Suggested action'],
        issueRows,
        'No semantic/text issue rows.'
      );
      setTable(
        'insightK5SampleTable',
        ['Issue group', 'Restaurant', 'Address', 'City token', 'Zip', 'Inspection date', 'Latitude', 'Longitude'],
        sampleRows,
        'No sample rows for semantic/text/numeric issues.'
      );
    }

    function renderK6() {
      const history = getQualityHistory();
      if (!history.length) {
        setInsightKpis([]);
        document.getElementById('insightContent').innerHTML =
          '<div class="small">No cross-run quality history is available yet. Keep daily snapshots in outputs/analysis.</div>';
        return;
      }
      const latest = history[history.length - 1];
      setInsightKpis([
        ['Snapshots', history.length.toLocaleString()],
        ['Latest run', String(latest.runId || 'N/A')],
        ['Latest outside-city rate', fmtPctRate(latest.cityOutsideRate)],
        ['Latest true-mismatch rate', fmtPctRate(latest.trueMismatchRate)]
      ]);
      document.getElementById('insightContent').innerHTML =
        '<div id="insightK6OutsideBars"></div>' +
        '<div id="insightK6MismatchBars"></div>' +
        '<div class="table-wrap"><table id="insightK6Table"></table></div>' +
        (history.length < 2
          ? '<div class="small">Only one snapshot is currently available; trend direction becomes meaningful after 2+ runs.</div>'
          : '');
      renderBarRows(
        'insightK6OutsideBars',
        history.map(r => [r.runId, Number(r.cityOutsideRate || 0) * 100]),
        0,
        1,
        v => Number(v || 0).toFixed(2) + '%'
      );
      renderBarRows(
        'insightK6MismatchBars',
        history.map(r => [r.runId, Number.isFinite(r.trueMismatchRate) ? Number(r.trueMismatchRate || 0) * 100 : null]).filter(r => r[1] != null),
        0,
        1,
        v => Number(v || 0).toFixed(2) + '%'
      );
      setTable(
        'insightK6Table',
        ['Run ID', 'Raw rows', 'Outside-city rate', 'Unknown-city-token rate', 'Date-parse-fail rate', 'Truncated-violation-text rate', 'True-rating-mismatch rate'],
        history.map(r => [
          r.runId,
          Number.isFinite(r.rowsRaw) ? Number(r.rowsRaw).toLocaleString() : 'N/A',
          fmtPctRate(r.cityOutsideRate),
          fmtPctRate(r.cityUnknownRate),
          fmtPctRate(r.dateParseFailRate),
          fmtPctRate(r.violationTruncatedRate),
          fmtPctRate(r.trueMismatchRate)
        ]),
        'No quality history rows.'
      );
    }

    function renderInsightView() {
      clearInsightPanels();
      const { role, qid } = getSelectedRoleAndQuestion();
      const q = roleQuestionById(role, qid);
      document.getElementById('insightHint').textContent = q ? t('question.' + q.id) : '';

      if (qid === 'C1') return renderC1();
      if (qid === 'C2') return renderC2();
      if (qid === 'C3') return renderC3();
      if (qid === 'C4') return renderC4OrO3('c4');
      if (qid === 'C5') return renderC5();
      if (qid === 'C6') return renderC6();

      if (qid === 'O1') return renderO1();
      if (qid === 'O2') return renderO2();
      if (qid === 'O3') return renderC4OrO3('o3');
      if (qid === 'O4') return renderO4();
      if (qid === 'O5') return renderO5();
      if (qid === 'O6') return renderO6();

      if (qid === 'R1') return renderR1();
      if (qid === 'R2') return renderR2();
      if (qid === 'R3') return renderR3();
      if (qid === 'R4') return renderR4();
      if (qid === 'R5') return renderR5();
      if (qid === 'R6') return renderR6();
      if (qid === 'K1') return renderK1();
      if (qid === 'K2') return renderK2();
      if (qid === 'K3') return renderK3();
      if (qid === 'K4') return renderK4();
      if (qid === 'K5') return renderK5();
      if (qid === 'K6') return renderK6();
    }

    function renderInsightsSummary() {
      const roleButtons = document.getElementById('insightRoleButtons');
      const qSel = document.getElementById('insightQuestionSelect');
      const roles = Object.keys(insightQuestionMap);
      if (!roles.includes(state.insightRole)) state.insightRole = roles[0] || 'Consumer';

      const refreshRoleButtons = () => {
        roleButtons.innerHTML = roles.map(role =>
          '<button class="role-btn ' + (role === state.insightRole ? 'active' : '') + '" data-role="' + esc(role) + '">' +
          esc(t('role.' + role)) + '</button>'
        ).join('');
        roleButtons.querySelectorAll('[data-role]').forEach(btn => {
          btn.addEventListener('click', () => {
            state.insightRole = btn.dataset.role || roles[0];
            refreshRoleButtons();
            refreshQuestions();
          });
        });
      };

      const refreshQuestions = () => {
        const role = state.insightRole;
        const questions = insightQuestionMap[role] || [];
        qSel.innerHTML = questions.map(q => '<option value="' + esc(q.id) + '">' + esc(t('question.' + q.id)) + '</option>').join('');
        renderInsightView();
      };

      qSel.onchange = renderInsightView;
      refreshRoleButtons();
      refreshQuestions();
    }

    function renderOverview() {
      const ov = APP.overview || {};
      const kpis = [
        ['Inspection rows', Number(ov.inspection_rows || 0).toLocaleString()],
        ['Restaurants', Number(ov.restaurant_count || 0).toLocaleString()],
        ['Red violation records', Number(ov.red_count || 0).toLocaleString()],
        ['Blue violation records', Number(ov.blue_count || 0).toLocaleString()]
      ];
      document.getElementById('overviewKpis').innerHTML = kpis.map(pair =>
        '<div class="kpi"><div class="k">' + esc(pair[0]) + '</div><div class="v">' + esc(pair[1]) + '</div></div>'
      ).join('');

      setTable(
        'overviewRatingTable',
        ['Risk Level', 'Rating', 'Rule'],
        (ov.rating_rows || []).map(r => [r[0], translateRatingLabel(r[1]), r[2]]),
        'No rating metadata.'
      );

      const vkpis = [
        ['Red violation records', Number(ov.red_count || 0).toLocaleString()],
        ['Blue violation records', Number(ov.blue_count || 0).toLocaleString()]
      ];
      document.getElementById('overviewViolationKpis').innerHTML = vkpis.map(pair =>
        '<div class="kpi"><div class="k">' + esc(pair[0]) + '</div><div class="v">' + esc(pair[1]) + '</div></div>'
      ).join('');

      setTable(
        'overviewTopCodeTable',
        ['Code', 'Type', 'Description', 'Occurrences'],
        (ov.top_code_rows || []).map(r => [r[0], r[1], r[2], r[3]]),
        'No violation code data.'
      );

      const posterCatalog = ov.poster_catalog || {};
      const posterOptions = Object.keys(posterCatalog);
      const posterSelect = document.getElementById('overviewRatingPosterSelect');
      posterSelect.innerHTML = '';
      posterOptions.forEach(label => {
        const opt = document.createElement('option');
        opt.value = label;
        opt.textContent = translateRatingLabel(label);
        posterSelect.appendChild(opt);
      });

      const renderOverviewPoster = () => {
        const selected = posterSelect.value || posterOptions[0] || '';
        const poster = posterCatalog[selected] || {};
        const path = String(poster.path || '');
        const desc = String(poster.description || '');
        const posterBox = document.getElementById('overviewPosterBox');
        const posterDesc = document.getElementById('overviewPosterDesc');
        if (path) {
          const src = path.startsWith('/') ? ('file://' + path) : path;
          posterBox.innerHTML =
            '<img src="' + esc(src) + '" class="poster-img" alt="' + esc(selected + ' poster') + '" />';
        } else {
          posterBox.innerHTML = '<div class="small">Poster image not available in local images folder.</div>';
        }
        posterDesc.innerHTML =
          '<div><strong>' + esc(translateRatingLabel(selected || 'Rating not available')) + '</strong></div>' +
          '<div style="margin-top:6px;">' + esc(desc) + '</div>';
      };

      if (posterOptions.length) {
        const defaultRating = posterOptions.includes('Excellent') ? 'Excellent' : posterOptions[0];
        posterSelect.value = defaultRating;
        posterSelect.addEventListener('change', renderOverviewPoster);
        renderOverviewPoster();
      } else {
        document.getElementById('overviewPosterBox').innerHTML =
          '<div class="small">No poster metadata available.</div>';
        document.getElementById('overviewPosterDesc').textContent = '';
      }

      const meta = APP.metadata || {};
      document.getElementById('overviewMetaFooter').textContent =
        'Data batch run_id: ' + String(ov.run_id || '-') +
        ' | Generated (UTC): ' + String(ov.generated_at_utc || '-') +
        ' | HTML exported: ' + String(meta.exported_at_utc || '-');
    }

    function htmlTable(headers, rows, emptyMessage) {
      const safeRows = rows || [];
      let html = '<div class="table-wrap"><table><thead><tr>' +
        headers.map(h => '<th>' + esc(h) + '</th>').join('') +
        '</tr></thead><tbody>';
      if (!safeRows.length) {
        html += '<tr><td colspan="' + headers.length + '">' + esc(emptyMessage || 'No data available.') + '</td></tr>';
      } else {
        for (const row of safeRows) {
          html += '<tr>' + row.map(cell => '<td>' + esc(cell) + '</td>').join('') + '</tr>';
        }
      }
      html += '</tbody></table></div>';
      return html;
    }

    function essayCard(title, body) {
      return '<div class="essay-card"><h4>' + esc(title) + '</h4><p>' + esc(body) + '</p></div>';
    }

    function takeawayBox(text) {
      return '<div class="takeaway-box"><strong>Takeaway.</strong> ' + esc(text) + '</div>';
    }

    function imageTag(path, caption) {
      if (!path) return '';
      const src = String(path).startsWith('/') ? ('file://' + path) : String(path);
      return '<div class="small">' + esc(caption || '') + '</div><img src="' + esc(src) + '" style="max-width:100%;max-height:440px;width:auto;object-fit:contain;display:block;border:1px solid var(--line);border-radius:8px;margin:6px 0 12px;" />';
    }

    function plotLayout(title, yTitle, extra) {
      return Object.assign({
        title,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '#ffffff',
        margin: { l: 52, r: 20, t: 46, b: 60 },
        font: { family: '"Avenir Next","IBM Plex Sans","Trebuchet MS","Segoe UI",sans-serif', color: '#0f2d36' },
        yaxis: yTitle ? { title: yTitle, gridcolor: '#e2eff1', zerolinecolor: '#d7e9ec' } : { gridcolor: '#e2eff1', zerolinecolor: '#d7e9ec' },
        xaxis: { gridcolor: '#f3f8f9', zerolinecolor: '#d7e9ec' }
      }, extra || {});
    }

    function renderExecutive() {
      const block = document.getElementById('executiveBlock');
      const hw = APP.homework || {};
      if (!hw.available) {
        block.innerHTML = '<div class="warn">' + esc(hw.message || 'Homework payload unavailable.') + '</div>';
        return;
      }
      const stats = hw.stats || {};
      const executive = hw.executive || {};
      const bestRow = executive.best_model_row || [];
      const fieldGroups = hw.field_groups || [];
      const fieldHtml = fieldGroups.map(group =>
        '<div class="essay-card"><h4>' + esc(group.title || '') + '</h4>' +
        htmlTable(['Field', 'Meaning'], group.rows || [], 'No fields available.') +
        '</div>'
      ).join('');

      block.innerHTML =
        '<div class="kpis">' +
          '<div class="kpi"><div class="k">Inspection events</div><div class="v">' + esc((stats.event_rows || 0).toLocaleString()) + '</div></div>' +
          '<div class="kpi"><div class="k">Violation rows</div><div class="v">' + esc((stats.violation_rows || 0).toLocaleString()) + '</div></div>' +
          '<div class="kpi"><div class="k">Model rows</div><div class="v">' + esc((stats.model_rows || 0).toLocaleString()) + '</div></div>' +
          '<div class="kpi"><div class="k">Restaurants</div><div class="v">' + esc((stats.restaurant_count || 0).toLocaleString()) + '</div></div>' +
        '</div>' +
        essayCard(
          'Dataset and database purpose',
          'This project uses King County restaurant inspection records to turn public food-safety data into an owner-facing decision tool. The database combines an inspection-event table, which records each inspection visit and its score, grade, result, and risk signals, with a violation-level table that stores individual findings, point severity, and remediation guidance.'
        ) +
        essayCard(
          'Prediction target',
          'The target is whether the same restaurant\'s next inspection becomes high risk. In this project, an inspection is treated as high risk when the published grade is Needs to Improve or when red points reach 25 or more.'
        ) +
        essayCard(
          'Why this matters',
          'For a restaurant owner, the value is operational rather than academic. If the latest inspection profile can estimate next-inspection risk early enough, the owner can focus on the controllable signals most likely to trigger a poor next outcome before the next county visit happens.'
        ) +
        essayCard(
          'Approach and key findings',
          'The analysis converts repeated inspections into a next-inspection prediction dataset and compares Logistic Regression, Decision Tree, Random Forest, XGBoost, and MLP. The current saved artifacts identify ' +
          esc(executive.best_model_name || '-') + ' as the best overall model; the assignment explainability view uses ' +
          esc(executive.best_tree_model_name || '-') + ' as the tree-based SHAP model.'
        ) +
        '<div class="takeaway-box"><strong>Core homework question.</strong> ' + esc(hw.owner_question || '') + '</div>' +
        htmlTable(
          ['Item', 'Value'],
          [
            ['Coverage window', (stats.min_date || '-') + ' to ' + (stats.max_date || '-')],
            ['Target positive rate', stats.positive_rate == null ? '-' : ((Number(stats.positive_rate) * 100).toFixed(2) + '%')],
            ['Model features', String(stats.feature_count || 0) + ' total (' + String(stats.numeric_feature_count || 0) + ' numeric, ' + String(stats.categorical_feature_count || 0) + ' categorical)'],
            ['Best overall model', String(executive.best_model_name || '-') + (bestRow.length ? (' | F1=' + bestRow[4] + ' | ROC_AUC=' + bestRow[5]) : '')]
          ],
          'No summary statistics available.'
        ) +
        '<div class="section-title">Field dictionary for the two source tables</div>' +
        '<div class="dictionary-grid">' + fieldHtml + '</div>';
    }

    function renderDescriptive() {
      const block = document.getElementById('descriptiveBlock');
      const hw = APP.homework || {};
      const desc = hw.descriptive || {};
      if (!hw.available) {
        block.innerHTML = '<div class="warn">' + esc(hw.message || 'Homework payload unavailable.') + '</div>';
        return;
      }

      block.innerHTML =
        essayCard(
          'Dataset introduction',
          'The modeling table predicts a restaurant\'s next inspection high-risk flag from the current inspection profile. The descriptive visuals below are selected to show imbalance, severity, and operational patterns that matter for the owner-facing prediction problem.'
        ) +
        takeawayBox(
          'The dataset is large enough for a meaningful classification workflow, but the target is imbalanced, so F1, recall, and ROC-AUC matter more than accuracy alone.'
        ) +
        '<div class="section-title">Target distribution of target_next_high_risk</div>' +
        '<div class="plot-box"><div id="descTargetPlot" class="plot-canvas"></div></div>' +
        '<div class="small">Most rows lead to a low-risk next inspection, while only a small minority lead to a future high-risk outcome. That imbalance is exactly why this project emphasizes class-aware metrics and avoids reading accuracy as the primary quality signal.</div>' +
        takeawayBox('The target is rare but operationally important, so the workflow is built around detection quality rather than raw hit rate.') +
        '<div class="section-title">Inspection score by future high-risk target</div>' +
        '<div class="plot-box"><div id="descScorePlot" class="plot-canvas"></div></div>' +
        '<div class="small">The high-risk-next group has a worse current inspection score distribution, which suggests the current inspection already contains usable early warning information. The overlap across groups shows why a single threshold rule is not enough.</div>' +
        takeawayBox('Inspection score matters, but it is only one component of a broader risk profile.') +
        '<div class="section-title">Red points by future high-risk target</div>' +
        '<div class="plot-box"><div id="descRedPlot" class="plot-canvas"></div></div>' +
        '<div class="small">Rows that later return as high risk carry materially higher red-point severity in the current inspection. That makes red points one of the most actionable warning signals for an owner.</div>' +
        takeawayBox('Red-point severity should be triaged first when an owner wants to reduce next-inspection risk quickly.') +
        '<div class="section-title">Violation count by future high-risk target</div>' +
        '<div class="plot-box"><div id="descViolationPlot" class="plot-canvas"></div></div>' +
        '<div class="small">A larger number of findings today is associated with a higher probability of a high-risk next visit. This suggests persistent process-control weakness, not just one isolated finding.</div>' +
        takeawayBox('High violation volume is itself a warning that the restaurant has not stabilized its operating process.') +
        '<div class="section-title">Inspection result by future high-risk rate</div>' +
        '<div class="plot-box"><div id="descResultPlot" class="plot-canvas"></div></div>' +
        '<div class="small">Published result categories do not carry the same downstream implication. Unsatisfactory or return-oriented outcomes show materially higher future high-risk rates, which justifies using inspection_result as a categorical model feature.</div>' +
        takeawayBox('Current inspection result acts like a process-state label and should be treated as a strong contextual signal.') +
        '<div class="section-title">City-level future high-risk pattern</div>' +
        '<div class="plot-box"><div id="descCityPlot" class="plot-canvas"></div></div>' +
        '<div class="small">Future high-risk rates are not uniform across higher-volume cities. Geography should not be treated as causal by itself, but it captures operating context and restaurant mix that still help the model.</div>' +
        takeawayBox('City context is useful as a supporting feature, not as a standalone explanation.') +
        '<div class="section-title">Correlation heatmap</div>' +
        '<div class="plot-box"><div id="descCorrPlot" class="plot-canvas"></div></div>' +
        '<div class="small">Score, point totals, violation count, and current high-risk status are related severity dimensions. None of them fully subsume the target, which supports combining them in a predictive model.</div>' +
        takeawayBox('The heatmap supports the modeling choice: related signals move together, but no single field fully explains next-inspection risk.');

      if (typeof Plotly === 'undefined') return;
      const targetCounts = desc.target_counts || [];
      Plotly.newPlot('descTargetPlot', [{
        type: 'bar',
        x: targetCounts.map(r => r[0]),
        y: targetCounts.map(r => r[1]),
        marker: { color: ['#8cd3dd', '#0d8a98'] }
      }], plotLayout('Target distribution', 'Inspection rows'), { displayModeBar: false, responsive: true });

      const samplePlots = [
        ['descScorePlot', desc.inspection_score_samples || {}, 'Inspection score'],
        ['descRedPlot', desc.red_points_samples || {}, 'Red points'],
        ['descViolationPlot', desc.violation_count_samples || {}, 'Violation count']
      ];
      for (const [plotId, series, yTitle] of samplePlots) {
        Plotly.newPlot(plotId, [
          { type: 'box', y: series.low || [], name: 'Next low risk', marker: { color: '#8cd3dd' }, boxmean: true },
          { type: 'box', y: series.high || [], name: 'Next high risk', marker: { color: '#0d8a98' }, boxmean: true }
        ], plotLayout('', yTitle), { displayModeBar: false, responsive: true });
      }

      const resultRows = desc.result_rate_rows || [];
      Plotly.newPlot('descResultPlot', [{
        type: 'bar',
        x: resultRows.map(r => r[0]),
        y: resultRows.map(r => r[1]),
        marker: { color: '#0d8a98' }
      }], plotLayout('Current inspection result vs. future high-risk rate', 'Future high-risk rate', {
        xaxis: { tickangle: -30 }
      }), { displayModeBar: false, responsive: true });

      const cityRows = desc.city_rate_rows || [];
      Plotly.newPlot('descCityPlot', [{
        type: 'bar',
        x: cityRows.map(r => r[0]),
        y: cityRows.map(r => r[1]),
        marker: { color: '#27c2a8' }
      }], plotLayout('Higher-volume cities by future high-risk rate', 'Future high-risk rate', {
        xaxis: { tickangle: -30 }
      }), { displayModeBar: false, responsive: true });

      Plotly.newPlot('descCorrPlot', [{
        type: 'heatmap',
        x: desc.corr_columns || [],
        y: desc.corr_columns || [],
        z: desc.corr_values || [],
        colorscale: 'RdBu',
        zmid: 0
      }], plotLayout('Correlation matrix', ''), { displayModeBar: false, responsive: true });
    }

    function renderPerformance() {
      const block = document.getElementById('performanceBlock');
      const model = APP.model || {};
      if (!model.available) {
        block.innerHTML = '<div class="warn">' + esc(model.message || 'Predict data unavailable.') + '</div>';
        return;
      }
      const metricsColumns = model.metrics_columns || [];
      const metricsRows = model.metrics_rows || [];
      const modelDetails = model.model_details || {};
      const modelNames = metricsRows.length
        ? metricsRows.map(r => String(r[0] || ''))
        : Object.keys(modelDetails || {});
      const bestModel = model.best_model_name && modelNames.includes(model.best_model_name)
        ? model.best_model_name
        : (modelNames[0] || '');

      const rocGallery = modelNames.map(name => {
        const detail = modelDetails[name] || {};
        return imageTag(detail.roc_plot_path || '', 'ROC curve — ' + name);
      }).join('');

      block.innerHTML =
        essayCard(
          'Data preparation and evaluation design',
          'The workflow builds X and y from the event-level table, preserves inspection order, keeps a held-out test set, and applies preprocessing inside each saved pipeline. Numerical fields are imputed, categorical fields are encoded, and the final evaluation emphasizes F1 and ROC-AUC because the target is imbalanced.'
        ) +
        '<div class="section-title">Model comparison summary table</div>' +
        htmlTable(metricsColumns, metricsRows, 'No model comparison rows available.') +
        '<div class="section-title">Key metric comparison (F1)</div><div class="plot-box"><div id="perfF1Plot" class="plot-canvas"></div></div>' +
        takeawayBox('The comparison should be read through F1, recall, and ROC-AUC because missing a future high-risk case is more costly than treating too many rows as potentially risky.') +
        '<div class="section-title">Best hyperparameters by model</div>' +
        htmlTable(
          ['Model', 'Best hyperparameters'],
          modelNames.map(name => [name, JSON.stringify((modelDetails[name] || {}).best_params || {}) || 'Baseline defaults']),
          'No hyperparameter metadata available.'
        ) +
        '<div class="section-title">ROC curves for all models</div>' +
        rocGallery +
        '<div class="section-title">Inspect one model in detail</div>' +
        '<div class="filters"><div class="field"><label>Select model</label><select id="performanceModelSelect">' +
          modelNames.map(name => '<option value="' + esc(name) + '"' + (name === bestModel ? ' selected' : '') + '>' + esc(name) + '</option>').join('') +
        '</select></div></div>' +
        '<div id="performanceModelDetail"></div>' +
        essayCard(
          'Model trade-offs',
          'Logistic Regression is the strongest overall model in the current saved artifacts, which means a relatively interpretable baseline remains competitive. Tree-based models remain useful because they capture threshold effects and support SHAP explanation, while the MLP satisfies the neural-network and tuning requirements even though it does not beat the baseline on F1.'
        );

      if (typeof Plotly !== 'undefined') {
        Plotly.newPlot('perfF1Plot', [{
          type: 'bar',
          x: metricsRows.map(r => r[0]),
          y: metricsRows.map(r => r[4]),
          marker: { color: '#0d8a98' }
        }], plotLayout('F1 across models', 'F1'), { displayModeBar: false, responsive: true });
      }

      const modelSel = document.getElementById('performanceModelSelect');
      const updatePerformanceDetail = () => {
        const modelName = modelSel.value;
        const detail = modelDetails[modelName] || {};
        const metrics = detail.metrics || {};
        const params = detail.best_params || {};
        const rows = detail.top_features_rows || [];
        const tuningRows = detail.tuning_top_rows || [];
        const bars = rows.map(r => {
          const width = Math.max(1, Math.min(100, Number(r[1] || 0) * 100));
          return '<div class="barrow"><div style="width:360px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' +
            esc(r[0]) + '</div><div class="bar"><span style="width:' + width.toFixed(1) + '%"></span></div><div style="width:90px">' +
            esc(Number(r[1] || 0).toFixed(4)) + '</div></div>';
        }).join('');
        let html =
          '<div class="kpis">' +
            '<div class="kpi"><div class="k">Accuracy</div><div class="v">' + esc(Number(metrics.Accuracy || 0).toFixed(4)) + '</div></div>' +
            '<div class="kpi"><div class="k">Precision</div><div class="v">' + esc(Number(metrics.Precision || 0).toFixed(4)) + '</div></div>' +
            '<div class="kpi"><div class="k">Recall</div><div class="v">' + esc(Number(metrics.Recall || 0).toFixed(4)) + '</div></div>' +
            '<div class="kpi"><div class="k">F1</div><div class="v">' + esc(Number(metrics.F1 || 0).toFixed(4)) + '</div></div>' +
          '</div>';
        html += '<div class="small">Best parameters</div><pre style="white-space:pre-wrap;background:#f8fbfd;border:1px solid var(--line);border-radius:8px;padding:8px;">' +
          esc(JSON.stringify(params, null, 2)) + '</pre>';
        html += '<div class="small">Top feature signals</div>' + (bars || '<div class="small">No feature table available.</div>');
        html += imageTag(detail.roc_plot_path || '', 'ROC curve — ' + modelName);
        html += imageTag(detail.tree_plot_path || '', 'Best tuned decision tree');
        html += imageTag(detail.history_plot_path || '', 'MLP training history');
        html += imageTag(detail.tuning_plot_path || '', 'MLP tuning top configurations');
        if (tuningRows.length) {
          html += '<div class="small">Bonus — MLP tuning results</div>' +
            htmlTable(
              ['Hidden Layers', 'Learning Rate', 'Dropout', 'Epochs', 'Val F1', 'Val ROC_AUC'],
              tuningRows,
              'No MLP tuning rows available.'
            );
        }
        document.getElementById('performanceModelDetail').innerHTML = html;
      };
      modelSel.addEventListener('change', updatePerformanceDetail);
      updatePerformanceDetail();
    }

    function renderExplainability() {
      const block = document.getElementById('explainabilityBlock');
      const model = APP.model || {};
      const hw = APP.homework || {};
      if (!model.available) {
        block.innerHTML = '<div class="warn">' + esc(model.message || 'Predict data unavailable.') + '</div>';
        return;
      }
      const shap = model.shap || {};
      const shapModel = ((hw.executive || {}).best_tree_model_name) || model.best_tree_model_name || shap.best_tree_model_name || 'Decision Tree';
      const modelNames = (model.metrics_rows || []).map(r => String(r[0] || ''));
      const defaultModel = model.best_model_name && modelNames.includes(model.best_model_name)
        ? model.best_model_name
        : (modelNames[0] || '');
      const predictOptions = (APP.businesses || []).slice().sort((a, b) =>
        String(a[bIdx.display_name] || '').localeCompare(String(b[bIdx.display_name] || ''))
      ).map(b => {
        const id = b[bIdx.business_id];
        return '<option value="' + esc(id) + '">' + esc(b[bIdx.display_name] + ' | ' + b[bIdx.city] + ' (' + id + ')') + '</option>';
      }).join('');

      block.innerHTML =
        essayCard(
          'Explainability setup',
          'Prediction can use any saved model, but the HTML mirror anchors SHAP interpretation to ' + String(shapModel) + '. This keeps the public-facing HTML simple while preserving the tree-based explanation required for the assignment.'
        ) +
        imageTag(shap.summary_plot_path || '', 'SHAP summary plot — ' + shapModel) +
        '<div class="small">The beeswarm plot shows both importance and direction: points to the right push the prediction toward the positive class, while points to the left pull it down. Features near the top have the strongest overall influence on model output.</div>' +
        takeawayBox('Global SHAP confirms that the tree-based model is reacting to a compact set of operational risk signals.') +
        imageTag(shap.bar_plot_path || '', 'Mean absolute SHAP values — ' + shapModel) +
        '<div class="small">The bar plot removes direction and ranks features by average absolute impact. It helps a decision-maker see which signals deserve the most attention before the next inspection occurs.</div>' +
        takeawayBox('The highest-ranked SHAP features define the inspection profile an owner should review first.') +
        htmlTable(['Feature', 'Mean |SHAP|'], hw.shap_rows || [], 'No SHAP ranking rows available.') +
        '<div class="section-title">Display-oriented prediction mirror</div>' +
        '<div class="filters" style="grid-template-columns:1fr 1.4fr;">' +
          '<div class="field"><label>Prediction model</label><select id="explainModelSelect">' +
            modelNames.map(name => '<option value="' + esc(name) + '"' + (name === defaultModel ? ' selected' : '') + '>' + esc(name) + '</option>').join('') +
          '</select></div>' +
          '<div class="field"><label>Restaurant template</label><select id="explainRestaurantSelect">' + predictOptions + '</select></div>' +
        '</div>' +
        '<div class="kpis" id="explainPredictKpis"></div>' +
        '<div class="section-title">Default input snapshot</div>' +
        '<div class="table-wrap"><table id="explainInputTable"></table></div>' +
        '<div class="small">For the full homework-grade custom input controls and custom SHAP waterfall, use the Streamlit app. The HTML page mirrors the same saved artifacts and per-restaurant prediction outputs without running custom inference in the browser.</div>';

      const modelSel = document.getElementById('explainModelSelect');
      const restaurantSel = document.getElementById('explainRestaurantSelect');
      const updateExplainabilityPrediction = () => {
        const businessId = restaurantSel.value;
        const modelName = modelSel.value;
        const pair = ((model.predictions_by_model_business || {})[modelName] || {})[businessId] || [null, 'N/A'];
        const prob = pair[0];
        const band = pair[1];
        const probText = prob == null ? 'N/A' : (Number(prob) * 100).toFixed(1) + '%';
        document.getElementById('explainPredictKpis').innerHTML =
          '<div class="kpi"><div class="k">Predicted probability</div><div class="v">' + esc(probText) + '</div></div>' +
          '<div class="kpi"><div class="k">Risk band</div><div class="v">' + esc(String(band || 'N/A')) + '</div></div>' +
          '<div class="kpi"><div class="k">Prediction model</div><div class="v">' + esc(modelName) + '</div></div>' +
          '<div class="kpi"><div class="k">SHAP model</div><div class="v">' + esc(String(shapModel || '-')) + '</div></div>';
        const defaults = (model.default_inputs_by_business || {})[businessId] || {};
        const rows = Object.keys(defaults).sort().map(key => [key, defaults[key]]);
        setTable('explainInputTable', ['Feature', 'Value'], rows, 'No default input values.');
      };
      modelSel.addEventListener('change', updateExplainabilityPrediction);
      restaurantSel.addEventListener('change', updateExplainabilityPrediction);
      updateExplainabilityPrediction();
    }

    function renderHomeworkPanels() {
      renderExecutive();
      renderDescriptive();
      renderPerformance();
      renderExplainability();
    }

    function initTabs() {
      document.querySelectorAll('.tabbtn').forEach(btn => {
        btn.addEventListener('click', () => {
          document.querySelectorAll('.tabbtn').forEach(x => x.classList.remove('active'));
          document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
          btn.classList.add('active');
          const panel = document.getElementById('tab-' + btn.dataset.tab);
          if (panel) panel.classList.add('active');
          renderI18nDebugPanel();
        });
      });
    }

    function init() {
      state.locale = 'en';

      initTabs();
      applyStaticTranslations();
      renderSearchModeButtons();
      updateSearchModePanels();
      renderRatingChipsForMode('ratingChipsKeyword', 'keyword');
      renderRatingChipsForMode('ratingChipsMap', 'map');
      renderRatingChipsForMode('ratingChipsHybrid', 'hybrid');
      populateCityOptions();
      applyStaticTranslations();

      document.getElementById('qInput').addEventListener('input', () => refreshSearch(true));
      document.getElementById('citySelect').addEventListener('change', () => refreshSearch(true));
      document.getElementById('zipInput').addEventListener('input', () => refreshSearch(true));
      document.getElementById('mapCitySelect').addEventListener('change', () => refreshSearch(true));
      document.getElementById('comboQueryInput').addEventListener('input', () => refreshSearch(true));
      document.getElementById('clearSearchFiltersBtn').addEventListener('click', clearSearchFilters);
      document.getElementById('clearHistoryFiltersBtn').addEventListener('click', clearHistoryFilters);
      document.getElementById('eventDateFrom').addEventListener('change', () => {
        if (state.selectedBusinessId) renderRestaurantDetail(state.selectedBusinessId);
      });
      document.getElementById('eventDateTo').addEventListener('change', () => {
        if (state.selectedBusinessId) renderRestaurantDetail(state.selectedBusinessId);
      });
      document.getElementById('eventTypeSelect').addEventListener('change', () => {
        if (state.selectedBusinessId) renderRestaurantDetail(state.selectedBusinessId);
      });

      renderOverview();
      refreshSearch(false);
      renderInsightsSummary();
      renderHomeworkPanels();
      renderI18nDebugPanel();
    }

    init();
  </script>
</body>
</html>
"""
    return html_template.replace("__DATA_JSON__", data_json)


def export_html(
    root: Path,
    output_html: Path,
    max_events_per_business: int,
    max_violations_per_business: int,
) -> Dict[str, Any]:
    payload = load_latest_payload(root)
    run_id = clean_text(payload.get("run_id", ""))
    silver_event_csv, violation_csv = resolve_paths(root, payload)

    events_df = pd.read_csv(silver_event_csv, dtype=str)
    violations_df = pd.read_csv(violation_csv, dtype=str)
    events_df = prepare_events_df(events_df, root, payload)
    city_zip_lookup = build_zip_to_locality_lookup(events_df)
    violations_df = prepare_violations_df(violations_df, city_zip_lookup)

    summary_df = build_business_summary(events_df)
    summary_df = summary_df.sort_values(
        by=["inspection_date_dt", "inspection_count"],
        ascending=[False, False],
        na_position="last",
    )

    overview_payload = build_overview_payload(root, events_df, violations_df, payload)
    model_bundle = build_predict_payload(root, summary_df)
    homework_payload = build_homework_payload(root, events_df, violations_df, summary_df)
    map_boundary = load_king_county_boundary_line_coords(root)

    business_columns = [
        "business_id",
        "display_name",
        "address",
        "city",
        "zip",
        "latitude",
        "longitude",
        "latest_date",
        "latest_rating",
        "latest_score",
        "inspection_count",
        "latest_risk_level_num",
        "latest_risk_level",
        "latest_avg_red_points",
        "latest_rating_window_target",
        "latest_rating_routine_n",
        "latest_rating_source",
        "latest_recalculated_rating",
        "latest_official_rating",
        "city_scope",
        "city_cleaning_reason",
        "search_blob",
    ]

    businesses: List[List[Any]] = []
    for row in summary_df.itertuples():
        search_blob = " ".join(
            [
                clean_text(getattr(row, "display_name", "")),
                clean_text(getattr(row, "business_name_alt", "")),
                clean_text(getattr(row, "full_address_clean", "")),
                clean_text(getattr(row, "city_display", "")),
                clean_text(getattr(row, "city_canonical", "")),
                clean_text(getattr(row, "zip_code", "")),
                clean_text(getattr(row, "search_name_norm", "")),
            ]
        )
        businesses.append(
            [
                clean_text(getattr(row, "business_id", "")),
                clean_text(getattr(row, "display_name", "")),
                clean_text(getattr(row, "full_address_clean", "")),
                clean_text(getattr(row, "city_display", "")),
                clean_text(getattr(row, "zip_code", "")),
                "" if pd.isna(getattr(row, "latitude", None)) else round(float(getattr(row, "latitude", 0)), 6),
                "" if pd.isna(getattr(row, "longitude", None)) else round(float(getattr(row, "longitude", 0)), 6),
                clean_text(getattr(row, "inspection_date", "")),
                clean_text(getattr(row, "latest_rating", "")),
                "" if pd.isna(getattr(row, "inspection_score", None)) else round(float(getattr(row, "inspection_score", 0)), 1),
                int(getattr(row, "inspection_count", 0)),
                clean_text(getattr(row, "latest_risk_level", "")),
                clean_text(getattr(row, "latest_risk_level_label", "")),
                "" if pd.isna(getattr(row, "latest_rating_avg_red_points", None)) else round(float(getattr(row, "latest_rating_avg_red_points", 0)), 2),
                "" if pd.isna(getattr(row, "latest_rating_window_target", None)) else int(float(getattr(row, "latest_rating_window_target", 0))),
                "" if pd.isna(getattr(row, "latest_rating_routine_n", None)) else int(float(getattr(row, "latest_rating_routine_n", 0))),
                clean_text(getattr(row, "latest_rating_source_label", "")),
                clean_text(getattr(row, "latest_recalculated_rating", "")),
                clean_text(getattr(row, "latest_official_rating", "")),
                clean_text(getattr(row, "city_scope", "")),
                clean_text(getattr(row, "city_cleaning_reason", "")),
                search_blob,
            ]
        )

    event_columns = [
        "event_id",
        "date",
        "type",
        "result",
        "score",
        "rating",
        "red",
        "blue",
        "violations",
    ]
    events_by_business: Dict[str, List[List[Any]]] = {}
    event_subset = events_df.sort_values("inspection_date_dt", ascending=False, na_position="last")
    for business_id, group in event_subset.groupby("business_id", dropna=False):
        gid = clean_text(business_id)
        if not gid:
            continue
        if max_events_per_business > 0:
            group = group.head(max_events_per_business)
        rows: List[List[Any]] = []
        for r in group.itertuples():
            rows.append(
                [
                    clean_text(getattr(r, "inspection_event_id", "")),
                    clean_text(getattr(r, "inspection_date", "")),
                    clean_text(getattr(r, "inspection_type", "")),
                    clean_text(getattr(r, "inspection_result", "")),
                    "" if pd.isna(getattr(r, "inspection_score", None)) else round(float(getattr(r, "inspection_score", 0)), 1),
                    clean_text(getattr(r, "effective_rating_label", "")),
                    "" if pd.isna(getattr(r, "red_points_total", None)) else int(float(getattr(r, "red_points_total", 0))),
                    "" if pd.isna(getattr(r, "blue_points_total", None)) else int(float(getattr(r, "blue_points_total", 0))),
                    "" if pd.isna(getattr(r, "violation_count_total", None)) else int(float(getattr(r, "violation_count_total", 0))),
                ]
            )
        events_by_business[gid] = rows

    violation_columns = [
        "event_id",
        "date",
        "type",
        "code",
        "points",
        "violation",
        "priority",
        "category",
        "remediation_summary",
    ]
    violations_by_business: Dict[str, List[List[Any]]] = {}
    viol_subset = violations_df.sort_values("inspection_date_dt", ascending=False, na_position="last")
    viol_subset = viol_subset[
        (viol_subset["violation_code"].astype(str).str.strip() != "")
        | (viol_subset["violation_desc_clean"].astype(str).str.strip() != "")
    ]
    for business_id, group in viol_subset.groupby("business_id", dropna=False):
        gid = clean_text(business_id)
        if not gid:
            continue
        if max_violations_per_business > 0:
            group = group.head(max_violations_per_business)
        rows = []
        for r in group.itertuples():
            rows.append(
                [
                    clean_text(getattr(r, "inspection_event_id", "")),
                    clean_text(getattr(r, "inspection_date", "")),
                    clean_text(getattr(r, "violation_type", "")),
                    clean_text(getattr(r, "violation_code", "")),
                    clean_text(getattr(r, "violation_points", "")),
                    clean_text(getattr(r, "violation_desc_clean", "")),
                    clean_text(getattr(r, "action_priority", "")),
                    clean_text(getattr(r, "action_category", "")),
                    clean_text(getattr(r, "action_summary_en", "")),
                ]
            )
        violations_by_business[gid] = rows

    trend_df = build_monthly_trend(events_df)
    monthly_trend_rows = trend_df[
        ["month", "inspections", "high_risk_inspections", "high_risk_rate", "avg_score"]
    ].values.tolist()

    movement_by_month = build_movement_by_month(events_df)
    owner_payload = build_owner_view_payload(violations_df)
    regulator_payload = build_regulator_view_payload(events_df, min_inspections=30)
    quality_payload = load_quality_payload(root, run_id)
    consumer_events = events_df[events_df["inspection_date_dt"].notna()].copy()
    consumer_payload = {
        "total_inspections": int(len(consumer_events)),
        "total_restaurants": int(consumer_events["business_id"].nunique()),
        "high_risk_inspections": int(consumer_events["is_high_risk"].sum()),
        "high_risk_rate": float(
            consumer_events["is_high_risk"].mean() if len(consumer_events) else 0.0
        ),
        "top_high_risk_rows": build_consumer_top_high_risk_rows(events_df, top_n=20),
    }

    movement_columns = [
        "display_name",
        "city",
        "address",
        "prev_date",
        "current_date",
        "transition",
        "change_type",
    ]

    export_payload = {
        "metadata": {
            "run_id": run_id,
            "generated_at_utc": clean_text(payload.get("generated_at_utc", "")),
            "exported_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "limits": {
                "max_events_per_business": max_events_per_business,
                "max_violations_per_business": max_violations_per_business,
            },
        },
        "business_columns": business_columns,
        "businesses": businesses,
        "event_columns": event_columns,
        "events_by_business": events_by_business,
        "violation_columns": violation_columns,
        "violations_by_business": violations_by_business,
        "monthly_trend_columns": [
            "month",
            "inspections",
            "high_risk_inspections",
            "high_risk_rate",
            "avg_score",
        ],
        "monthly_trend_rows": monthly_trend_rows,
        "movement_columns": movement_columns,
        "movement_by_month": movement_by_month,
        "overview": overview_payload,
        "consumer": consumer_payload,
        "owner": owner_payload,
        "regulator": regulator_payload,
        "quality": quality_payload,
        "model": model_bundle,
        "homework": homework_payload,
        "map_boundary": map_boundary,
    }

    html = build_html(export_payload)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")

    return {
        "output_html": str(output_html),
        "business_count": len(businesses),
        "events_business_count": len(events_by_business),
        "violations_business_count": len(violations_by_business),
        "monthly_points": len(monthly_trend_rows),
        "movement_month_count": len(movement_by_month),
        "model_available": model_bundle.get("available", False),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a static HTML dashboard from latest pipeline outputs.")
    parser.add_argument("--root", type=str, default=".", help="Project root")
    parser.add_argument(
        "--output-html",
        type=str,
        default="outputs/dashboard/index.html",
        help="Output HTML path",
    )
    parser.add_argument(
        "--max-events-per-business",
        type=int,
        default=200,
        help="Max inspection events kept per business in HTML payload (-1 for no limit)",
    )
    parser.add_argument(
        "--max-violations-per-business",
        type=int,
        default=300,
        help="Max violations kept per business in HTML payload (-1 for no limit)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output_html = Path(args.output_html)
    if not output_html.is_absolute():
        output_html = (root / output_html).resolve()
    event_limit = args.max_events_per_business
    viol_limit = args.max_violations_per_business
    if event_limit < 0:
        event_limit = 0
    if viol_limit < 0:
        viol_limit = 0

    summary = export_html(
        root=root,
        output_html=output_html,
        max_events_per_business=event_limit,
        max_violations_per_business=viol_limit,
    )
    print(f"[ok] html={summary['output_html']}")
    print(
        f"[ok] businesses={summary['business_count']:,} "
        f"events_by_business={summary['events_business_count']:,} "
        f"violations_by_business={summary['violations_business_count']:,}"
    )
    print(
        f"[ok] monthly_points={summary['monthly_points']:,} "
        f"movement_months={summary['movement_month_count']:,} "
        f"model_available={summary['model_available']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
