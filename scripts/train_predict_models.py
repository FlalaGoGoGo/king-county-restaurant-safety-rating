#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


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

DATASET_ID = "f29f-zza5"
RANDOM_STATE = 42
TARGET_COLUMN = "target_next_high_risk"

HIGH_RISK_GRADES = {"4"}
HIGH_RISK_RED_POINTS_THRESHOLD = 25

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


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)


def load_latest_payload(root: Path) -> Dict[str, Any]:
    state_path = root / "Data" / "state" / f"{DATASET_ID}_latest_run.json"
    if not state_path.exists():
        raise FileNotFoundError(f"latest run state not found: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def resolve_event_csv(root: Path, payload: Dict[str, Any]) -> Path:
    run_id = clean_text(payload.get("run_id", ""))
    silver_event_csv = Path(clean_text(payload.get("silver_event_csv", "")))
    if not silver_event_csv.exists():
        silver_event_csv = root / "Data" / "silver" / DATASET_ID / run_id / "inspection_event.csv"
    if not silver_event_csv.exists():
        raise FileNotFoundError(f"silver event csv missing: {silver_event_csv}")
    return silver_event_csv


def prepare_events_df(events_df: pd.DataFrame) -> pd.DataFrame:
    events_df = events_df.fillna("").copy()
    for col in ["inspection_score", "red_points_total", "blue_points_total", "violation_count_total"]:
        events_df[col] = pd.to_numeric(events_df.get(col, ""), errors="coerce")
    events_df["inspection_date_dt"] = pd.to_datetime(events_df.get("inspection_date", ""), errors="coerce")
    events_df["grade_num"] = pd.to_numeric(events_df.get("grade", ""), errors="coerce")
    grade_high_risk = events_df.get("grade", "").astype(str).str.strip().isin(HIGH_RISK_GRADES)
    red_points_high_risk = events_df["red_points_total"].fillna(0) >= HIGH_RISK_RED_POINTS_THRESHOLD
    events_df["is_high_risk"] = (grade_high_risk | red_points_high_risk).astype(int)
    return events_df


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
    dataset[TARGET_COLUMN] = dataset.groupby("business_id")["is_high_risk"].shift(-1)
    dataset = dataset[dataset[TARGET_COLUMN].notna()].copy()
    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].astype(int)
    return dataset


def maybe_downsample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    sampled, _ = train_test_split(
        df,
        train_size=max_rows,
        random_state=seed,
        stratify=df[TARGET_COLUMN],
    )
    return sampled.reset_index(drop=True)


def build_preprocessor(sparse_output: bool) -> ColumnTransformer:
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)
    return ColumnTransformer(
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
                        ("onehot", cat_encoder),
                    ]
                ),
                MODEL_CATEGORICAL_FEATURES,
            ),
        ]
    )


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_proba)),
    }
    return out


def save_roc_plot(path: Path, fpr: np.ndarray, tpr: np.ndarray, auc_val: float, title: str) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={auc_val:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def extract_linear_coefficients(pipeline: Pipeline, top_n: int = 25) -> pd.DataFrame:
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = pre.get_feature_names_out()
    coef = model.coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coef})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    return coef_df.sort_values("abs_coefficient", ascending=False).head(top_n)


def extract_tree_importance(pipeline: Pipeline, top_n: int = 25) -> pd.DataFrame:
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = pre.get_feature_names_out()
    importance = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    return imp_df.sort_values("importance", ascending=False).head(top_n)


def train_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> Dict[str, Any]:
    pre = build_preprocessor(sparse_output=True)
    model = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    pipeline = Pipeline(steps=[("preprocessor", pre), ("model", model)])
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = evaluate_binary(y_test.values, y_pred, y_proba)
    fpr, tpr, _ = roc_curve(y_test.values, y_proba)

    model_path = output_dir / "models" / "logistic_regression.joblib"
    ensure_dir(model_path.parent)
    joblib.dump(pipeline, model_path)

    roc_png = output_dir / "plots" / "roc_logistic_regression.png"
    save_roc_plot(roc_png, fpr, tpr, metrics["ROC_AUC"], "ROC - Logistic Regression")

    coef_csv = output_dir / "artifacts" / "logistic_coefficients_top25.csv"
    ensure_dir(coef_csv.parent)
    extract_linear_coefficients(pipeline, top_n=25).to_csv(coef_csv, index=False)

    return {
        "name": "Logistic Regression",
        "kind": "sklearn_pipeline",
        "model_path": str(model_path),
        "best_params": {},
        "metrics": metrics,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_plot_path": str(roc_png),
        "extra": {
            "coefficients_csv": str(coef_csv),
        },
    }


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    cv_folds: int,
) -> Dict[str, Any]:
    pre = build_preprocessor(sparse_output=True)
    model = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    pipeline = Pipeline(steps=[("preprocessor", pre), ("model", model)])

    grid = {
        "model__max_depth": [3, 5, 7, 10],
        "model__min_samples_leaf": [5, 10, 20, 50],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = evaluate_binary(y_test.values, y_pred, y_proba)
    fpr, tpr, _ = roc_curve(y_test.values, y_proba)

    model_path = output_dir / "models" / "decision_tree.joblib"
    ensure_dir(model_path.parent)
    joblib.dump(best_pipeline, model_path)

    roc_png = output_dir / "plots" / "roc_decision_tree.png"
    save_roc_plot(roc_png, fpr, tpr, metrics["ROC_AUC"], "ROC - Decision Tree")

    tree_png = output_dir / "plots" / "decision_tree_visualization.png"
    try:
        preprocessor = best_pipeline.named_steps["preprocessor"]
        clf = best_pipeline.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        fig, ax = plt.subplots(figsize=(24, 10))
        plot_tree(
            clf,
            feature_names=feature_names,
            class_names=["Low Risk", "High Risk"],
            max_depth=3,
            filled=True,
            rounded=True,
            fontsize=5,
            ax=ax,
        )
        fig.tight_layout()
        fig.savefig(tree_png, dpi=180)
        plt.close(fig)
        tree_plot_path = str(tree_png)
    except Exception:
        tree_plot_path = ""

    return {
        "name": "Decision Tree",
        "kind": "sklearn_pipeline",
        "model_path": str(model_path),
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "metrics": metrics,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_plot_path": str(roc_png),
        "extra": {
            "tree_plot_path": tree_plot_path,
        },
    }


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    cv_folds: int,
) -> Dict[str, Any]:
    pre = build_preprocessor(sparse_output=True)
    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[("preprocessor", pre), ("model", model)])

    grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [3, 5, 8],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = evaluate_binary(y_test.values, y_pred, y_proba)
    fpr, tpr, _ = roc_curve(y_test.values, y_proba)

    model_path = output_dir / "models" / "random_forest.joblib"
    ensure_dir(model_path.parent)
    joblib.dump(best_pipeline, model_path)

    roc_png = output_dir / "plots" / "roc_random_forest.png"
    save_roc_plot(roc_png, fpr, tpr, metrics["ROC_AUC"], "ROC - Random Forest")

    imp_csv = output_dir / "artifacts" / "random_forest_feature_importance_top25.csv"
    ensure_dir(imp_csv.parent)
    extract_tree_importance(best_pipeline, top_n=25).to_csv(imp_csv, index=False)

    return {
        "name": "Random Forest",
        "kind": "sklearn_pipeline",
        "model_path": str(model_path),
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "metrics": metrics,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_plot_path": str(roc_png),
        "extra": {
            "feature_importance_csv": str(imp_csv),
        },
    }


def train_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    cv_folds: int,
) -> Dict[str, Any]:
    from xgboost import XGBClassifier

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = float(neg / max(1, pos))

    pre = build_preprocessor(sparse_output=True)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    pipeline = Pipeline(steps=[("preprocessor", pre), ("model", model)])

    grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [3, 4, 5],
        "model__learning_rate": [0.01, 0.05, 0.1],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = evaluate_binary(y_test.values, y_pred, y_proba)
    fpr, tpr, _ = roc_curve(y_test.values, y_proba)

    model_path = output_dir / "models" / "xgboost.joblib"
    ensure_dir(model_path.parent)
    joblib.dump(best_pipeline, model_path)

    roc_png = output_dir / "plots" / "roc_xgboost.png"
    save_roc_plot(roc_png, fpr, tpr, metrics["ROC_AUC"], "ROC - XGBoost")

    imp_csv = output_dir / "artifacts" / "xgboost_feature_importance_top25.csv"
    ensure_dir(imp_csv.parent)
    extract_tree_importance(best_pipeline, top_n=25).to_csv(imp_csv, index=False)

    return {
        "name": "XGBoost",
        "kind": "sklearn_pipeline",
        "model_path": str(model_path),
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "metrics": metrics,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_plot_path": str(roc_png),
        "extra": {
            "feature_importance_csv": str(imp_csv),
        },
    }


def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed. Run: pip install torch")

    pre = build_preprocessor(sparse_output=False)
    X_train_arr = np.asarray(pre.fit_transform(X_train), dtype=np.float32)
    X_test_arr = np.asarray(pre.transform(X_test), dtype=np.float32)
    y_train_arr = y_train.to_numpy(dtype=np.int64)
    y_test_arr = y_test.to_numpy(dtype=np.int64)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_arr,
        y_train_arr,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_arr,
    )

    device = torch.device("cpu")
    x_fit_t = torch.from_numpy(X_fit).to(device)
    y_fit_t = torch.from_numpy(y_fit.astype(np.float32).reshape(-1, 1)).to(device)
    x_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.float32).reshape(-1, 1)).to(device)
    x_test_t = torch.from_numpy(X_test_arr).to(device)

    def run_single_config(
        hidden_layers: Tuple[int, ...],
        learning_rate: float,
        dropout: float,
    ) -> Dict[str, Any]:
        model = MLPBinaryClassifier(
            input_dim=X_fit.shape[1],
            hidden_layers=hidden_layers,
            dropout=dropout,
        ).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_ds = TensorDataset(x_fit_t, y_fit_t)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

        max_epochs = 25
        patience = 4
        patience_count = 0
        best_val_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        history_rows: List[Dict[str, Any]] = []

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss_total = 0.0
            train_correct = 0
            train_count = 0

            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

                batch_n = int(yb.shape[0])
                train_loss_total += float(loss.item()) * batch_n
                train_pred_label = (pred >= 0.5).float()
                train_correct += int((train_pred_label == yb).sum().item())
                train_count += batch_n

            model.eval()
            with torch.no_grad():
                val_pred = model(x_val_t)
                val_loss = float(criterion(val_pred, y_val_t).item())
                val_pred_label = (val_pred >= 0.5).float()
                val_acc = float((val_pred_label == y_val_t).float().mean().item())

            train_loss = train_loss_total / max(1, train_count)
            train_acc = train_correct / max(1, train_count)

            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                }
            )

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            val_proba = model(x_val_t).cpu().numpy().reshape(-1)
        val_pred = (val_proba >= 0.5).astype(int)
        try:
            val_auc = float(roc_auc_score(y_val, val_proba))
        except Exception:
            val_auc = 0.0

        return {
            "model": model,
            "history_rows": history_rows,
            "epochs_trained": int(len(history_rows)),
            "hidden_layers": list(hidden_layers),
            "learning_rate": float(learning_rate),
            "dropout": float(dropout),
            "val_accuracy": float(accuracy_score(y_val, val_pred)),
            "val_precision": float(precision_score(y_val, val_pred, zero_division=0)),
            "val_recall": float(recall_score(y_val, val_pred, zero_division=0)),
            "val_f1": float(f1_score(y_val, val_pred, zero_division=0)),
            "val_roc_auc": val_auc,
        }

    hidden_layer_grid = [(64, 64), (128, 128), (256, 128)]
    learning_rate_grid = [0.001, 0.0005]
    dropout_grid = [0.0, 0.2]

    tuning_results: List[Dict[str, Any]] = []
    best_result: Dict[str, Any] | None = None

    print("[train][MLP] hyperparameter tuning (hidden_layers, lr, dropout)...")
    for hidden_layers in hidden_layer_grid:
        for learning_rate in learning_rate_grid:
            for dropout in dropout_grid:
                result = run_single_config(hidden_layers, learning_rate, dropout)
                tuning_results.append(result)
                print(
                    "[train][MLP][tune] "
                    f"h={result['hidden_layers']} lr={learning_rate} drop={dropout} "
                    f"val_f1={result['val_f1']:.4f} val_auc={result['val_roc_auc']:.4f}"
                )
                if best_result is None:
                    best_result = result
                else:
                    current_key = (result["val_f1"], result["val_roc_auc"], -result["dropout"])
                    best_key = (
                        best_result["val_f1"],
                        best_result["val_roc_auc"],
                        -best_result["dropout"],
                    )
                    if current_key > best_key:
                        best_result = result

    if best_result is None:
        raise RuntimeError("MLP tuning produced no result.")

    model = best_result["model"]
    history_rows = best_result["history_rows"]

    with torch.no_grad():
        y_proba = model(x_test_t).cpu().numpy().reshape(-1)
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = evaluate_binary(y_test_arr, y_pred, y_proba)
    fpr, tpr, _ = roc_curve(y_test_arr, y_proba)

    model_path = output_dir / "models" / "mlp_torch.pth"
    pre_path = output_dir / "models" / "mlp_preprocessor.joblib"
    ensure_dir(model_path.parent)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(X_fit.shape[1]),
            "hidden_layers": best_result["hidden_layers"],
            "dropout": best_result["dropout"],
        },
        model_path,
    )
    joblib.dump(pre, pre_path)

    roc_png = output_dir / "plots" / "roc_mlp.png"
    save_roc_plot(roc_png, fpr, tpr, metrics["ROC_AUC"], "ROC - MLP")

    history_df = pd.DataFrame(history_rows)
    history_csv = output_dir / "artifacts" / "mlp_history.csv"
    ensure_dir(history_csv.parent)
    history_df.to_csv(history_csv, index=False)

    tuning_rows = []
    for row in tuning_results:
        tuning_rows.append(
            {
                "hidden_layers": str(row["hidden_layers"]),
                "learning_rate": row["learning_rate"],
                "dropout": row["dropout"],
                "epochs_trained": row["epochs_trained"],
                "val_accuracy": row["val_accuracy"],
                "val_precision": row["val_precision"],
                "val_recall": row["val_recall"],
                "val_f1": row["val_f1"],
                "val_roc_auc": row["val_roc_auc"],
            }
        )
    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        by=["val_f1", "val_roc_auc"], ascending=False
    )
    tuning_csv = output_dir / "artifacts" / "mlp_tuning_results.csv"
    ensure_dir(tuning_csv.parent)
    tuning_df.to_csv(tuning_csv, index=False)

    tuning_png = output_dir / "plots" / "mlp_tuning_top_configs.png"
    top_tuning = tuning_df.head(8).copy()
    labels = [
        f"h={r.hidden_layers}, lr={r.learning_rate}, d={r.dropout}"
        for r in top_tuning.itertuples(index=False)
    ]
    fig_tune, ax_tune = plt.subplots(figsize=(10, 4.5))
    ax_tune.barh(labels[::-1], top_tuning["val_f1"].tolist()[::-1], color="#1f7a8c")
    ax_tune.set_xlabel("Validation F1")
    ax_tune.set_title("MLP Hyperparameter Tuning (Top Configurations)")
    fig_tune.tight_layout()
    fig_tune.savefig(tuning_png, dpi=160)
    plt.close(fig_tune)

    history_png = output_dir / "plots" / "mlp_training_history.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if "train_loss" in history_df:
        axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    if "val_loss" in history_df:
        axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[0].set_title("MLP Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    if "train_accuracy" in history_df:
        axes[1].plot(history_df["epoch"], history_df["train_accuracy"], label="train_accuracy")
    if "val_accuracy" in history_df:
        axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="val_accuracy")
    axes[1].set_title("MLP Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(history_png, dpi=160)
    plt.close(fig)

    return {
        "name": "MLP",
        "kind": "pytorch_mlp",
        "model_path": str(model_path),
        "preprocessor_path": str(pre_path),
        "best_params": {
            "hidden_layers": best_result["hidden_layers"],
            "activation": "relu",
            "optimizer": "adam",
            "epochs_trained": int(best_result["epochs_trained"]),
            "batch_size": 256,
            "framework": "pytorch",
            "learning_rate": best_result["learning_rate"],
            "dropout": best_result["dropout"],
            "tuning_grid": {
                "hidden_layers": [list(x) for x in hidden_layer_grid],
                "learning_rate": learning_rate_grid,
                "dropout": dropout_grid,
            },
            "selection_metric": "validation_f1",
        },
        "metrics": metrics,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_plot_path": str(roc_png),
        "extra": {
            "history_csv": str(history_csv),
            "history_plot_path": str(history_png),
            "tuning_results_csv": str(tuning_csv),
            "tuning_plot_path": str(tuning_png),
        },
    }


def normalize_shap_values(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            arr = np.asarray(shap_values[1])
        else:
            arr = np.asarray(shap_values[0])
    else:
        arr = np.asarray(shap_values)

    if arr.ndim == 3:
        if arr.shape[-1] > 1:
            arr = arr[:, :, 1]
        else:
            arr = arr[:, :, 0]
    return arr


def normalize_expected_value(expected_value: Any) -> float:
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        arr = np.asarray(expected_value).reshape(-1)
        if len(arr) > 1:
            return float(arr[1])
        return float(arr[0])
    return float(expected_value)


def generate_shap_artifacts(
    tree_model_info: Dict[str, Any],
    X_test: pd.DataFrame,
    output_dir: Path,
    shap_sample_size: int,
) -> Dict[str, Any]:
    model_path = Path(tree_model_info["model_path"])
    pipeline: Pipeline = joblib.load(model_path)
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    sample_n = min(max(100, shap_sample_size), len(X_test))
    sampled = X_test.sample(n=sample_n, random_state=RANDOM_STATE).copy()
    transformed = pre.transform(sampled)

    if hasattr(transformed, "toarray"):
        transformed_dense = np.asarray(transformed.toarray(), dtype=np.float32)
    else:
        transformed_dense = np.asarray(transformed, dtype=np.float32)

    feature_names = [str(x) for x in pre.get_feature_names_out()]
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(transformed_dense)
    shap_values = normalize_shap_values(shap_values_raw)
    expected_value = normalize_expected_value(explainer.expected_value)

    summary_png = output_dir / "plots" / "shap_summary.png"
    bar_png = output_dir / "plots" / "shap_bar.png"
    ensure_dir(summary_png.parent)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        transformed_dense,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(summary_png, dpi=170)
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        transformed_dense,
        feature_names=feature_names,
        max_display=20,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(bar_png, dpi=170)
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    imp_df = imp_df.sort_values("mean_abs_shap", ascending=False)
    imp_csv = output_dir / "artifacts" / "shap_mean_abs_top30.csv"
    ensure_dir(imp_csv.parent)
    imp_df.head(30).to_csv(imp_csv, index=False)

    return {
        "best_tree_model_name": tree_model_info["name"],
        "model_name": tree_model_info["name"],
        "summary_plot_path": str(summary_png),
        "bar_plot_path": str(bar_png),
        "mean_abs_shap_csv": str(imp_csv),
        "expected_value": expected_value,
        "feature_names": feature_names,
    }


def select_best_shap_tree_model(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    eligible_names = {"Random Forest", "XGBoost", "LightGBM"}
    shap_candidates = [row for row in results if row["name"] in eligible_names]
    if not shap_candidates:
        raise RuntimeError(
            "No eligible SHAP tree model found. Expected one of: Random Forest, XGBoost, LightGBM."
        )
    shap_candidates.sort(
        key=lambda row: (row["metrics"]["F1"], row["metrics"]["ROC_AUC"]),
        reverse=True,
    )
    return shap_candidates[0]


def build_interactive_defaults(dataset: pd.DataFrame) -> Dict[str, Any]:
    numeric_defaults = {}
    for col in MODEL_NUMERIC_FEATURES:
        numeric_defaults[col] = float(dataset[col].median()) if dataset[col].notna().any() else 0.0

    categorical_defaults = {}
    categorical_options = {}
    for col in MODEL_CATEGORICAL_FEATURES:
        values = [clean_text(v) for v in dataset[col].dropna().astype(str).tolist() if clean_text(v)]
        unique_values = sorted(set(values))
        categorical_options[col] = unique_values
        if unique_values:
            mode_val = pd.Series(values).mode(dropna=True)
            categorical_defaults[col] = clean_text(mode_val.iloc[0]) if not mode_val.empty else unique_values[0]
        else:
            categorical_defaults[col] = ""

    return {
        "numeric_defaults": numeric_defaults,
        "categorical_defaults": categorical_defaults,
        "categorical_options": categorical_options,
    }


def train_all_models(
    dataset: pd.DataFrame,
    output_dir: Path,
    test_size: float,
    cv_folds: int,
    skip_shap: bool,
    shap_sample_size: int,
) -> Dict[str, Any]:
    X = dataset[MODEL_ALL_FEATURES].copy()
    y = dataset[TARGET_COLUMN].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results: List[Dict[str, Any]] = []

    t0 = time.time()
    print("[train] Logistic Regression...")
    results.append(train_logistic(X_train, y_train, X_test, y_test, output_dir))

    print("[train] Decision Tree (GridSearchCV)...")
    results.append(train_decision_tree(X_train, y_train, X_test, y_test, output_dir, cv_folds=cv_folds))

    print("[train] Random Forest (GridSearchCV)...")
    results.append(train_random_forest(X_train, y_train, X_test, y_test, output_dir, cv_folds=cv_folds))

    print("[train] MLP (PyTorch)...")
    results.append(train_mlp(X_train, y_train, X_test, y_test, output_dir))

    print("[train] XGBoost (GridSearchCV)...")
    try:
        results.append(train_boosting(X_train, y_train, X_test, y_test, output_dir, cv_folds=cv_folds))
    except Exception as exc:
        print(f"[warn] XGBoost failed: {exc}")
        print("[train] Falling back to LightGBM (GridSearchCV)...")
        results.append(train_lightgbm(X_train, y_train, X_test, y_test, output_dir, cv_folds=cv_folds))

    elapsed = round(time.time() - t0, 2)
    print(f"[train] completed in {elapsed}s")

    comparison_rows = []
    for row in results:
        metrics = row["metrics"]
        comparison_rows.append(
            {
                "Model": row["name"],
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "ROC_AUC": metrics["ROC_AUC"],
            }
        )
    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["F1", "ROC_AUC"], ascending=False
    )
    best_model_name = str(comparison_df.iloc[0]["Model"])

    tree_candidates = [r for r in results if r["name"] in {"Decision Tree", "Random Forest", "XGBoost", "LightGBM"}]
    tree_candidates.sort(
        key=lambda r: (r["metrics"]["F1"], r["metrics"]["ROC_AUC"]),
        reverse=True,
    )
    best_tree = tree_candidates[0]
    best_shap_tree = select_best_shap_tree_model(results)

    if skip_shap:
        shap_bundle = {
            "best_tree_model_name": best_shap_tree["name"],
            "model_name": best_shap_tree["name"],
            "summary_plot_path": "",
            "bar_plot_path": "",
            "mean_abs_shap_csv": "",
            "expected_value": 0.0,
            "feature_names": [],
        }
    else:
        print("[train] SHAP artifacts...")
        shap_bundle = generate_shap_artifacts(
            tree_model_info=best_shap_tree,
            X_test=X_test,
            output_dir=output_dir,
            shap_sample_size=shap_sample_size,
        )

    comparison_csv = output_dir / "artifacts" / "model_comparison.csv"
    ensure_dir(comparison_csv.parent)
    comparison_df.to_csv(comparison_csv, index=False)

    roc_json = output_dir / "artifacts" / "roc_curves.json"
    roc_payload = {
        row["name"]: {
            "fpr": row["fpr"],
            "tpr": row["tpr"],
            "roc_auc": row["metrics"]["ROC_AUC"],
            "roc_plot_path": row["roc_plot_path"],
        }
        for row in results
    }
    save_json(roc_json, roc_payload)

    return {
        "results": results,
        "comparison_df": comparison_df,
        "comparison_csv": comparison_csv,
        "roc_json": roc_json,
        "best_model_name": best_model_name,
        "best_tree_model_name": best_tree["name"],
        "best_shap_tree_model_name": best_shap_tree["name"],
        "shap": shap_bundle,
        "elapsed_seconds": elapsed,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
    }


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    cv_folds: int,
) -> Dict[str, Any]:
    from lightgbm import LGBMClassifier

    pre = build_preprocessor(sparse_output=True)
    model = LGBMClassifier(
        random_state=RANDOM_STATE,
        objective="binary",
        class_weight="balanced",
        n_jobs=-1,
        verbosity=-1,
    )
    pipeline = Pipeline(steps=[("preprocessor", pre), ("model", model)])

    grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [3, 4, 5],
        "model__learning_rate": [0.01, 0.05, 0.1],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = evaluate_binary(y_test.values, y_pred, y_proba)
    fpr, tpr, _ = roc_curve(y_test.values, y_proba)

    model_path = output_dir / "models" / "lightgbm.joblib"
    ensure_dir(model_path.parent)
    joblib.dump(best_pipeline, model_path)

    roc_png = output_dir / "plots" / "roc_lightgbm.png"
    save_roc_plot(roc_png, fpr, tpr, metrics["ROC_AUC"], "ROC - LightGBM")

    imp_csv = output_dir / "artifacts" / "lightgbm_feature_importance_top25.csv"
    ensure_dir(imp_csv.parent)
    extract_tree_importance(best_pipeline, top_n=25).to_csv(imp_csv, index=False)

    return {
        "name": "LightGBM",
        "kind": "sklearn_pipeline",
        "model_path": str(model_path),
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "metrics": metrics,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_plot_path": str(roc_png),
        "extra": {
            "feature_importance_csv": str(imp_csv),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HW1 predictive models and save artifacts.")
    parser.add_argument("--root", type=str, default=".", help="Project root")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/hw1_predict",
        help="Output directory for trained models and artifacts",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test split ratio",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds for GridSearchCV",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=30000,
        help="Max rows for modeling dataset (0 means use all rows)",
    )
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=400,
        help="Sample size for SHAP summary generation",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP artifact generation (debug mode)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()

    set_global_seed(RANDOM_STATE)

    payload = load_latest_payload(root)
    run_id = clean_text(payload.get("run_id", ""))
    event_csv = resolve_event_csv(root, payload)

    print(f"[info] loading data: {event_csv}")
    events_df = pd.read_csv(event_csv, dtype=str)
    events_df = prepare_events_df(events_df)
    dataset = build_next_inspection_dataset(events_df)

    if dataset.empty:
        raise RuntimeError("Model dataset is empty.")

    source_rows = int(len(dataset))
    dataset = maybe_downsample(dataset, max_rows=int(args.max_rows), seed=RANDOM_STATE)
    used_rows = int(len(dataset))

    print(f"[info] dataset rows for modeling: {used_rows:,} (source {source_rows:,})")
    print(f"[info] positive rate: {dataset[TARGET_COLUMN].mean():.4f}")

    ensure_dir(output_dir)
    bundle = train_all_models(
        dataset=dataset,
        output_dir=output_dir,
        test_size=float(args.test_size),
        cv_folds=int(args.cv_folds),
        skip_shap=bool(args.skip_shap),
        shap_sample_size=int(args.shap_sample_size),
    )

    models_payload = {}
    for row in bundle["results"]:
        models_payload[row["name"]] = {
            "kind": row["kind"],
            "model_path": str(Path(row["model_path"]).resolve()),
            "best_params": row["best_params"],
            "metrics": row["metrics"],
            "roc_plot_path": str(Path(row["roc_plot_path"]).resolve()),
            "extra": {
                k: (str(Path(v).resolve()) if isinstance(v, str) and v else v)
                for k, v in row.get("extra", {}).items()
            },
        }
        if row.get("preprocessor_path"):
            models_payload[row["name"]]["preprocessor_path"] = str(Path(row["preprocessor_path"]).resolve())

    defaults = build_interactive_defaults(dataset)

    manifest = {
        "generated_at_utc": now_utc_iso(),
        "dataset_id": DATASET_ID,
        "run_id": run_id,
        "random_state": RANDOM_STATE,
        "target_column": TARGET_COLUMN,
        "feature_columns": MODEL_ALL_FEATURES,
        "numeric_features": MODEL_NUMERIC_FEATURES,
        "categorical_features": MODEL_CATEGORICAL_FEATURES,
        "source_rows": source_rows,
        "model_rows": used_rows,
        "test_size": float(args.test_size),
        "cv_folds": int(args.cv_folds),
        "train_rows": bundle["train_rows"],
        "test_rows": bundle["test_rows"],
        "positive_rate_train": bundle["positive_rate_train"],
        "positive_rate_test": bundle["positive_rate_test"],
        "elapsed_seconds": bundle["elapsed_seconds"],
        "best_model_name": bundle["best_model_name"],
        "best_tree_model_name": bundle["best_tree_model_name"],
        "best_shap_tree_model_name": bundle["best_shap_tree_model_name"],
        "models": models_payload,
        "comparison_csv_path": str(Path(bundle["comparison_csv"]).resolve()),
        "roc_json_path": str(Path(bundle["roc_json"]).resolve()),
        "shap": {
            "best_tree_model_name": bundle["shap"]["best_tree_model_name"],
            "model_name": bundle["shap"]["model_name"],
            "summary_plot_path": (
                str(Path(bundle["shap"]["summary_plot_path"]).resolve())
                if bundle["shap"]["summary_plot_path"]
                else ""
            ),
            "bar_plot_path": (
                str(Path(bundle["shap"]["bar_plot_path"]).resolve())
                if bundle["shap"]["bar_plot_path"]
                else ""
            ),
            "mean_abs_shap_csv": (
                str(Path(bundle["shap"]["mean_abs_shap_csv"]).resolve())
                if bundle["shap"]["mean_abs_shap_csv"]
                else ""
            ),
            "expected_value": bundle["shap"]["expected_value"],
            "feature_names": bundle["shap"]["feature_names"],
        },
        "interactive_defaults": defaults,
    }

    manifest_path = output_dir / "manifest.json"
    latest_manifest_path = output_dir / "latest_manifest.json"
    save_json(manifest_path, manifest)
    save_json(latest_manifest_path, {"manifest_path": str(manifest_path.resolve())})

    print(f"[ok] manifest: {manifest_path}")
    print(f"[ok] best model: {manifest['best_model_name']}")
    print(f"[ok] best tree model: {manifest['best_tree_model_name']}")
    print(f"[ok] best SHAP tree model: {manifest['best_shap_tree_model_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
