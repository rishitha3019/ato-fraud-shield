import argparse, json, os, warnings
warnings.filterwarnings("ignore")
import mlflow, mlflow.xgboost
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, average_precision_score
import xgboost as xgb

FEATURE_COLS = [
    "hours_since_last_login","km_from_last_login","velocity_kmh","velocity_kmh_log",
    "impossible_velocity_flag","is_new_device","failed_attempts_1h","failed_attempts_6h",
    "failed_attempts_24h","failed_attempts_ratio","session_duration_sec","actions_per_minute",
    "time_to_first_action_sec","account_age_days","hour_deviation_from_mean","hour_of_day",
    "day_of_week","is_weekend","is_night",
]
TARGET_COL = "is_fraud"
ARTIFACTS_DIR = "model/artifacts"

def load_and_prepare(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path).dropna(subset=FEATURE_COLS + [TARGET_COL])
    print(f"  Shape: {df.shape}  |  Fraud rate: {df[TARGET_COL].mean():.2%}")
    return df[FEATURE_COLS].copy(), df[TARGET_COL].copy()

def train(data_path, threshold, test_size, n_estimators, max_depth, learning_rate, smote):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    X, y = load_and_prepare(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    if smote:
        print("Applying SMOTE...")
        X_train, y_train = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_train, y_train)
        print(f"  Resampled: {pd.Series(y_train).value_counts().to_dict()}")
    model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=1 if smote else (y_train==0).sum()/(y_train==1).sum(),
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    mlflow.set_experiment("ato-fraud-detection")
    with mlflow.start_run(run_name="xgboost-smote") as run:
        print(f"MLflow run: {run.info.run_id}")
        mlflow.log_params({"n_estimators":n_estimators,"max_depth":max_depth,
            "learning_rate":learning_rate,"smote":smote,"threshold":threshold})
        print("Training XGBoost...")
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_prob = model.predict_proba(X_test)[:,1]
        y_pred = (y_prob >= threshold).astype(int)
        roc_auc = roc_auc_score(y_test, y_prob)
        avg_prec = average_precision_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        p, r, f1 = report["1"]["precision"], report["1"]["recall"], report["1"]["f1-score"]
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n{'='*50}")
        print(f"  ROC-AUC:       {roc_auc:.4f}")
        print(f"  Avg Precision: {avg_prec:.4f}")
        print(f"  Precision:     {p:.4f}")
        print(f"  Recall:        {r:.4f}")
        print(f"  F1:            {f1:.4f}")
        print(f"{'='*50}")
        print(f"Confusion Matrix:\n  TN={cm[0,0]:,}  FP={cm[0,1]:,}\n  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
        mlflow.log_metrics({"roc_auc":roc_auc,"avg_precision":avg_prec,"precision":p,"recall":r,"f1":f1})
        prec_c, rec_c, thr_c = precision_recall_curve(y_test, y_prob)
        f1_c = 2*prec_c*rec_c/(prec_c+rec_c+1e-9)
        best_idx = np.argmax(f1_c)
        best_thr = float(thr_c[best_idx]) if best_idx < len(thr_c) else threshold
        print(f"\nOptimal F1 threshold: {best_thr:.3f}")
        importance = dict(zip(FEATURE_COLS, model.feature_importances_))
        top10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 features:")
        for feat, imp in top10:
            print(f"  {feat:<35} {imp:.4f}")
        model.save_model(os.path.join(ARTIFACTS_DIR, "xgb_model.json"))
        meta = {"feature_cols":FEATURE_COLS,"threshold":threshold,"optimal_f1_threshold":best_thr,"roc_auc":roc_auc,"avg_precision":avg_prec}
        with open(os.path.join(ARTIFACTS_DIR, "feature_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        pd.DataFrame({"threshold":list(thr_c)+[1.0],"precision":list(prec_c),"recall":list(rec_c),"f1":list(f1_c)}).to_csv(os.path.join(ARTIFACTS_DIR,"pr_curve.csv"),index=False)
        mlflow.xgboost.log_model(model, "model")
        print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default="data/raw/login_events.csv")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--no-smote", action="store_true")
    args = p.parse_args()
    train(args.data_path, args.threshold, args.test_size, args.n_estimators, args.max_depth, args.learning_rate, not args.no_smote)

if __name__ == "__main__":
    main()
