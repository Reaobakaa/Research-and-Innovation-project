# traffic_accident_models_v6.py
# ------------------------------------------------------------------
# • 80 / 20 train-test evaluation  → accuracy + full classification report
# • Re-fit each model on the FULL dataset → in-sample accuracy + report
# • --data <file.xlsx> can override the hard-coded path
# • --save persists the full-data pipelines as *_full.joblib
# ------------------------------------------------------------------
import argparse, sys, joblib
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # script runs without XGBoost if not installed

# Hard-coded dataset path (change here if you move the file)
DEFAULT_DATA_PATH = Path(r"C:\ADA ASSIDNMENT\RTA Dataset (clean) .xlsx")

TARGET = "Accident_severity"
BASE_FEATURES = [
    "Hour",
    "Age_band_of_driver", "Driving_experience", "Casualty_class",
    "Service_year_of_vehicle", "Type_of_vehicle", "Day_of_week",
    "Area_accident_occured_", "Lanes_or_Medians", "Types_of_Junction",
    "Cause_of_accident", "Number_of_vehicles_involved", "Number_of_casualties",
]
NUMERIC = ["Hour", "Number_of_vehicles_involved", "Number_of_casualties"]

# ───────────────────────────────────────────────────────── helpers
def load_df(xlsx: Path) -> pd.DataFrame:
    if not xlsx.exists():
        sys.exit(f"ERROR: dataset not found → {xlsx}")

    df = pd.read_excel(xlsx)
    df.columns = df.columns.str.strip()

    if "Area_accident_occured" in df.columns and "Area_accident_occured_" not in df.columns:
        df.rename(columns={"Area_accident_occured": "Area_accident_occured_"},
                  inplace=True)

    if "Time" in df.columns:
        df["Hour"] = pd.to_datetime(df["Time"], errors="coerce").dt.hour
        df.drop(columns=["Time"], inplace=True)

    df = df.dropna(subset=[TARGET])

    if "Hour" in df.columns and df["Hour"].notna().sum() == 0:
        df.drop(columns=["Hour"], inplace=True)

    return df


def make_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])


def build_models(prep, n_classes):
    models = {
        "log_reg": Pipeline([
            ("prep", prep),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced"))
        ]),
        "grad_boost": Pipeline([
            ("prep", prep),
            ("clf", GradientBoostingClassifier(random_state=42))
        ])
    }
    if XGBClassifier:
        models["xgboost"] = Pipeline([
            ("prep", prep),
            ("clf", XGBClassifier(
                objective="multi:softprob", num_class=n_classes,
                n_estimators=700, learning_rate=0.05,
                max_depth=6, subsample=0.9,
                colsample_bytree=0.8, random_state=42))
        ])
    return models

# ───────────────────────────────────────────────────────── workflow
def evaluate(label_encoder, name, model, X_test, y_test, encoded=False):
    if encoded:                       # for XGBoost
        preds_int = model.predict(X_test)
        preds = label_encoder.inverse_transform(preds_int)
    else:
        preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"{name:<12} accuracy = {acc:.3f}")
    print(classification_report(y_test, preds, digits=3))
    print("-"*60)

def main(data_path: Path, save_pipes: bool):
    df = load_df(data_path)

    feat_cols = [c for c in BASE_FEATURES if c in df.columns]
    num_cols  = [c for c in NUMERIC       if c in df.columns]
    cat_cols  = [c for c in feat_cols     if c not in num_cols]

    X, y = df[feat_cols], df[TARGET]
    le   = LabelEncoder().fit(y)
    y_enc = le.transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    y_tr_enc, y_te_enc = le.transform(y_tr), le.transform(y_te)

    prep   = make_preprocessor(num_cols, cat_cols)
    models = build_models(prep, len(le.classes_))

    # ─── Hold-out evaluation
    print("\n=== Hold-out (20 %) metrics ===")
    for name, mdl in models.items():
        if name == "xgboost":
            mdl.fit(X_tr, y_tr_enc)
            evaluate(le, name, mdl, X_te, y_te, encoded=True)
        else:
            mdl.fit(X_tr, y_tr)
            evaluate(le, name, mdl, X_te, y_te)

    # ─── Full-data fit & in-sample report
    print("\n=== Metrics after re-fitting on FULL dataset ===")
    for name, mdl in models.items():
        if name == "xgboost":
            mdl.fit(X, y_enc)
            evaluate(le, name, mdl, X, y, encoded=True)
        else:
            mdl.fit(X, y)
            evaluate(le, name, mdl, X, y)

        if save_pipes:
            joblib.dump(mdl, f"{name}_full.joblib")

    if save_pipes:
        print("Saved full-data pipelines → *_full.joblib")

# ───────────────────────────────────────────────────────── entry-point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to Excel dataset",
                        default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--save", action="store_true",
                        help="Save the full-data pipelines")
    args = parser.parse_args()

    main(Path(args.data).expanduser(), save_pipes=args.save)
