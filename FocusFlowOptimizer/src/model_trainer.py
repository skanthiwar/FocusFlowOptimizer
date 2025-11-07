"""
Focus Flow Optimizer — Model Trainer (Versioned, Full Reports, Enhanced)
------------------------------------------------------------------------
Trains a classifier to detect Flow (1) vs Distracted (0).
Adds engineered features, optional temporal split, optional grid search,
optional probability calibration, and best-threshold selection (macro-F1).

Outputs:
  - models/focus_model_vYYYY-MM-DD_HH-MM.pkl
  - models/app_encoder_vYYYY-MM-DD_HH-MM.pkl
  - models/focus_model_latest.pkl
  - models/app_encoder_latest.pkl
  - reports/classification_report.txt
  - reports/feature_importances.csv        (if available)
  - reports/metrics.json                   (includes best_threshold, macro metrics)
  - reports/app_label_mapping.csv
  - reports/test_predictions.csv           (optional)
  - reports/confusion_matrix.png           (optional)
  - reports/roc_curve.png                  (optional)
"""

import os
import sys
import json
import warnings
import joblib
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_curve, auc
)

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# Config (toggle here)
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.2

ADD_ENGINEERED_FEATURES = True      # add Total_Input, ratios, flags, app-group feature
USE_TEMPORAL_SPLIT = False          # set True if rows are chronological
DO_GRID_SEARCH = False              # quick hyperparameter search for RF
CALIBRATE_PROBAS = False            # wrap model with CalibratedClassifierCV
SAVE_FIGURES = True                 # save confusion matrix + ROC
SAVE_TEST_PREDICTIONS = True        # save reports/test_predictions.csv

REQUIRED_COLUMNS = ["Key_Rate", "Mouse_Rate", "Active_App", "Flow_State"]

def _project_root():
    # Works in normal script and in PyInstaller bundles
    return getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = _project_root()
DATA_PATH = os.path.join(ROOT, "data", "labeled_data.csv")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")

# Stable "latest" aliases for the optimizer
MODEL_LATEST = os.path.join(MODELS_DIR, "focus_model_latest.pkl")
ENCODER_LATEST = os.path.join(MODELS_DIR, "app_encoder_latest.pkl")

# Reports
CLASSIF_REPORT_PATH = os.path.join(REPORTS_DIR, "classification_report.txt")
FEATURE_IMP_PATH = os.path.join(REPORTS_DIR, "feature_importances.csv")
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")
CONF_MAT_PNG = os.path.join(REPORTS_DIR, "confusion_matrix.png")
ROC_CURVE_PNG = os.path.join(REPORTS_DIR, "roc_curve.png")
APP_LABEL_MAP_CSV = os.path.join(REPORTS_DIR, "app_label_mapping.csv")
TEST_PRED_PATH = os.path.join(REPORTS_DIR, "test_predictions.csv")

# =========================
# Utils
# =========================
def _ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

def _timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")

def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tolerant validator:
    - Accepts minor header variations (case/space)
    - Drops Flow_State == -1
    - Maps common text labels to 0/1
    - Cleans numerics like '1,234' -> 1234
    - Ensures both classes exist
    """
    want = {
        "key_rate": "Key_Rate",
        "mouse_rate": "Mouse_Rate",
        "active_app": "Active_App",
        "flow_state": "Flow_State",
    }
    lower_to_orig = {c.strip().lower(): c for c in df.columns}
    missing = [w for w in want if w not in lower_to_orig]
    if missing:
        raise ValueError(f"Missing required columns: {[want[m] for m in missing]}")

    df = df[[lower_to_orig["key_rate"],
             lower_to_orig["mouse_rate"],
             lower_to_orig["active_app"],
             lower_to_orig["flow_state"]]].copy()
    df.columns = ["Key_Rate", "Mouse_Rate", "Active_App", "Flow_State"]

    # Drop unlabeled (-1)
    df["Flow_State"] = df["Flow_State"].astype(str).str.strip()
    df = df[df["Flow_State"].str.lower() != "-1"]
    if df.empty:
        raise ValueError("No labeled rows found (Flow_State must be 0 or 1).")

    # Clean strings
    df["Active_App"] = df["Active_App"].astype(str).str.strip()
    df["Flow_State"] = df["Flow_State"].astype(str).str.strip().str.lower()

    # Map common labels to 0/1
    flow_map = {
        "1": 1, "0": 0, "flow": 1, "distracted": 0,
        "yes": 1, "no": 0, "true": 1, "false": 0
    }
    df["Flow_State"] = df["Flow_State"].map(flow_map).fillna(df["Flow_State"])

    # Clean numeric-like strings
    def clean_num(x):
        s = str(x).strip()
        s = s.replace(",", "")  # 1,234 -> 1234
        s = s.replace("+", "")
        return s

    df["Key_Rate"] = pd.to_numeric(df["Key_Rate"].apply(clean_num), errors="coerce")
    df["Mouse_Rate"] = pd.to_numeric(df["Mouse_Rate"].apply(clean_num), errors="coerce")
    df["Flow_State"] = pd.to_numeric(df["Flow_State"], errors="coerce")

    # Drop NaNs and enforce 0/1
    df = df.dropna(subset=["Key_Rate", "Mouse_Rate", "Flow_State"])
    df = df[df["Flow_State"].isin([0, 1])]

    if df.empty:
        raise ValueError("No valid numeric rows after cleaning.")
    if df["Flow_State"].nunique() < 2:
        raise ValueError("Need both classes present: Flow (1) and Distracted (0).")

    return df

def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # Basic signals
    df["Total_Input"]     = df["Key_Rate"] + df["Mouse_Rate"]
    df["Key_Mouse_Ratio"] = (df["Key_Rate"] / (df["Mouse_Rate"] + 1)).clip(0, 50)
    df["Is_ActiveKeys"]   = (df["Key_Rate"]  > 0).astype(int)
    df["Is_ActiveMouse"]  = (df["Mouse_Rate"]> 0).astype(int)

    # Simple app grouping to reduce sparsity
    dev = ["code.exe","pycharm64.exe","idea64.exe","notepad++.exe","excel.exe","notepad.exe","sublime_text.exe","devenv.exe"]
    comm= ["chrome.exe","msedge.exe","firefox.exe","slack.exe","teams.exe","outlook.exe","discord.exe","whatsapp.exe","telegram.exe"]
    df["App_Group"] = "other"
    df.loc[df["Active_App"].str.lower().isin([x.lower() for x in dev]),  "App_Group"] = "dev"
    df.loc[df["Active_App"].str.lower().isin([x.lower() for x in comm]), "App_Group"] = "comm"
    return df

def _encode_apps(df: pd.DataFrame):
    enc = LabelEncoder()
    df["Active_App_Encoded"] = enc.fit_transform(df["Active_App"].astype(str))
    return df, enc

def _encode_app_group(df: pd.DataFrame):
    if "App_Group" not in df.columns:
        return df, None
    enc = LabelEncoder()
    df["App_Group_Enc"] = enc.fit_transform(df["App_Group"].astype(str))
    return df, enc

def _maybe_grid_search(X_train, y_train):
    params = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", None],
    }
    base = RandomForestClassifier(
        random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
    )
    gs = GridSearchCV(base, params, scoring="f1", cv=3, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    print(f"Best params: {gs.best_params_}")
    return gs.best_estimator_

def _save_figures(y_test, y_proba):
    """Save confusion matrix and ROC curve using matplotlib (no seaborn)."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # headless-safe
        import matplotlib.pyplot as plt

        # Confusion Matrix at 0.5
        y_pred_05 = (y_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred_05)

        plt.figure(figsize=(4, 3))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix (thr=0.50)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks([0, 1], ["Distracted", "Flow"])
        plt.yticks([0, 1], ["Distracted", "Flow"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(CONF_MAT_PNG, dpi=150)
        plt.close()
        print(f"🖼️ Confusion matrix → {CONF_MAT_PNG}")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(4, 3))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(ROC_CURVE_PNG, dpi=150)
        plt.close()
        print(f"🖼️ ROC curve → {ROC_CURVE_PNG}")
    except Exception as e:
        print(f"⚠️ Could not save figures: {e}")

# =========================
# Train
# =========================
def train_focus_model():
    print("\n--- Starting Model Training ---")
    print(f"Data path: {DATA_PATH}")

    _ensure_dirs()

    if not os.path.exists(DATA_PATH):
        print("❌ Labeled data not found. Create and label 'data/labeled_data.csv'.")
        return

    df = pd.read_csv(DATA_PATH)
    try:
        df = _validate_input(df)
    except ValueError as e:
        print(f"❌ {e}")
        return

    # Class distribution
    print("\n--- Class Distribution ---")
    print(df["Flow_State"].value_counts())
    if df["Flow_State"].value_counts().min() < df["Flow_State"].value_counts().max() / 2:
        print("⚠️ Significant imbalance detected. Using class_weight='balanced'.")

    # Engineered features (optional)
    if ADD_ENGINEERED_FEATURES:
        df = _add_engineered_features(df)

    # Encode apps + save mapping
    df, app_encoder = _encode_apps(df)
    try:
        pd.DataFrame({
            "class_id": range(len(app_encoder.classes_)),
            "app_name": app_encoder.classes_
        }).to_csv(APP_LABEL_MAP_CSV, index=False)
        print(f"🗂️ App label mapping → {APP_LABEL_MAP_CSV}")
    except Exception:
        pass

    # Encode app group if present
    df, appgrp_encoder = _encode_app_group(df)

    # Features / target
    base_features = ["Key_Rate", "Mouse_Rate", "Active_App_Encoded"]
    eng_features = ["Total_Input", "Key_Mouse_Ratio", "Is_ActiveKeys", "Is_ActiveMouse", "App_Group_Enc"]
    features = base_features + (eng_features if ADD_ENGINEERED_FEATURES else [])
    features = [f for f in features if f in df.columns]  # guard

    X = df[features]
    y = df["Flow_State"].astype(int)

    # Split
    if USE_TEMPORAL_SPLIT:
        split_idx = int(len(df) * (1 - TEST_SIZE))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        print(f"\nTemporal split → Train: {len(X_train)}, Test: {len(X_test)}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        print(f"\nRandom split → Train: {len(X_train)}, Test: {len(X_test)}")

    # Train model
    if DO_GRID_SEARCH:
        print("\nRunning GridSearchCV...")
        model = _maybe_grid_search(X_train, y_train)
    else:
        print("\nTraining RandomForestClassifier...")
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

    # Optional probability calibration
    if CALIBRATE_PROBAS:
        from sklearn.calibration import CalibratedClassifierCV
        print("Calibrating probabilities (isotonic, cv=3)...")
        model = CalibratedClassifierCV(model, method="isotonic", cv=3).fit(X_train, y_train)

    print("✅ Training complete.")

    # ===== Balanced thresholding (macro-F1) =====
    y_proba = model.predict_proba(X_test)[:, 1]

    import numpy as np
    best_macro_f1, best_t = 0.0, 0.50
    for t in np.linspace(0.2, 0.8, 25):
        y_hat = (y_proba >= t).astype(int)
        f_macro = f1_score(y_test, y_hat, average="macro")
        if f_macro > best_macro_f1:
            best_macro_f1, best_t = f_macro, float(t)

    print(f"🔎 Best threshold (macro-F1) ~ {best_t:.2f} (macro-F1={best_macro_f1:.3f})")

    # Build report using best macro-F1 threshold
    y_pred_thr = (y_proba >= best_t).astype(int)
    report = classification_report(
        y_test, y_pred_thr, target_names=["Distracted (0)", "Flow (1)"], digits=4
    )
    acc = float(accuracy_score(y_test, y_pred_thr))
    f1_macro = float(f1_score(y_test, y_pred_thr, average="macro"))
    f1_weighted = float(f1_score(y_test, y_pred_thr, average="weighted"))
    prec_macro = float(precision_score(y_test, y_pred_thr, average="macro"))
    rec_macro = float(recall_score(y_test, y_pred_thr, average="macro"))
    cm = confusion_matrix(y_test, y_pred_thr).tolist()

    print("\n--- Evaluation (macro-F1 threshold) ---")
    print(report)

    # Versioned filenames
    vtag = _timestamp()
    model_versioned = os.path.join(MODELS_DIR, f"focus_model_v{vtag}.pkl")
    encoder_versioned = os.path.join(MODELS_DIR, f"app_encoder_v{vtag}.pkl")

    # Persist artifacts
    joblib.dump(model, model_versioned)
    joblib.dump(app_encoder, encoder_versioned)
    joblib.dump(model, MODEL_LATEST)
    joblib.dump(app_encoder, ENCODER_LATEST)

    print(f"\n💾 Saved model (versioned) → {model_versioned}")
    print(f"💾 Saved encoder (versioned) → {encoder_versioned}")
    print(f"🟢 Updated latest pointers → {MODEL_LATEST}, {ENCODER_LATEST}")

    # Reports / metrics
    with open(CLASSIF_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Trained: {vtag}\n")
        f.write(f"Best threshold (macro-F1): {best_t:.3f}\n\n")
        f.write(report)
    print(f"📝 Classification report → {CLASSIF_REPORT_PATH}")

    # Feature importances (only if available)
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            pd.DataFrame({"feature": features, "importance": importances})\
                .sort_values("importance", ascending=False)\
                .to_csv(FEATURE_IMP_PATH, index=False)
            print(f"📊 Feature importances → {FEATURE_IMP_PATH}")
    except Exception:
        pass

    # Save metrics (macro/weighted)
    metrics = {
        "version": vtag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": features,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "best_threshold": round(best_t, 3),
        "confusion_matrix": cm,
        "params": {k: v for k, v in (getattr(model, "get_params", lambda: {})() or {}).items()
                   if isinstance(v, (int, float, str, bool, type(None)))}
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"📦 Metrics JSON → {METRICS_PATH}")

    # Optional: save per-row test predictions for inspection
    if SAVE_TEST_PREDICTIONS:
        pd.DataFrame({
            "y_true": y_test.values,
            "proba":  y_proba,
            "y_pred": y_pred_thr
        }).to_csv(TEST_PRED_PATH, index=False)
        print(f"🧪 Test predictions → {TEST_PRED_PATH}")

    if SAVE_FIGURES:
        _save_figures(y_test.values, y_proba)

    print("\n🎉 Training finished. You can now run the flow optimizer (it can read best_threshold from metrics).")

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    train_focus_model()
