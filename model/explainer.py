import argparse, json, os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

ARTIFACTS_DIR = "model/artifacts"

def load_model_and_meta():
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(ARTIFACTS_DIR, "xgb_model.json"))
    with open(os.path.join(ARTIFACTS_DIR, "feature_meta.json")) as f:
        meta = json.load(f)
    return model, meta

def top_shap_features(shap_vals, feature_names, n=5):
    pairs = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:n]
    return [{"feature":feat,"shap_value":round(float(val),4),"direction":"increases fraud risk" if val>0 else "decreases fraud risk"} for feat,val in pairs]

def run_explainer(data_path, n_samples, top_n):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print("Loading model...")
    model, meta = load_model_and_meta()
    feature_cols = meta["feature_cols"]
    print(f"Loading data ({n_samples:,} samples)...")
    df = pd.read_csv(data_path).dropna(subset=feature_cols)
    fraud_idx = df[df["is_fraud"]==1].index
    legit_idx = df[df["is_fraud"]==0].index
    n_fraud = min(len(fraud_idx), n_samples//5)
    n_legit = n_samples - n_fraud
    sample_idx = np.concatenate([
        np.random.choice(fraud_idx, size=n_fraud, replace=False),
        np.random.choice(legit_idx, size=n_legit, replace=False),
    ])
    X = df[feature_cols].loc[sample_idx].reset_index(drop=True)
    y = df["is_fraud"].loc[sample_idx].reset_index(drop=True)
    print(f"  {len(X):,} events ({y.sum()} fraud)")
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    y_prob = model.predict_proba(X)[:,1]
    print("Generating plots...")
    plt.figure(figsize=(10,7))
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.title("SHAP Feature Importance — ATO Fraud Model", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR,"shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_summary.png")
    fraud_mask = y==1
    if fraud_mask.any():
        idx = int(np.where(fraud_mask)[0][0])
        explanation = shap.Explanation(values=shap_values[idx], base_values=explainer.expected_value, data=X.iloc[idx].values, feature_names=list(X.columns))
        plt.figure(figsize=(10,6))
        shap.waterfall_plot(explanation, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR,"shap_waterfall.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: shap_waterfall.png")
    mean_abs = np.abs(shap_values).mean(axis=0)
    pairs = sorted(zip(feature_cols, mean_abs), key=lambda x: x[1])
    feats, vals = zip(*pairs[-15:])
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(feats, vals, color="#1D9E75", alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Impact on ATO Fraud Prediction")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR,"shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_bar.png")
    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in feature_cols])
    shap_df["fraud_probability"] = y_prob
    shap_df["is_fraud"] = y.values
    shap_df["predicted_fraud"] = (y_prob>=0.5).astype(int)
    shap_cols = [f"shap_{c}" for c in feature_cols]
    top_idx = np.abs(shap_df[shap_cols].values).argmax(axis=1)
    shap_df["top_feature"] = [feature_cols[i] for i in top_idx]
    shap_df.to_csv(os.path.join(ARTIFACTS_DIR,"shap_values.csv"), index=False)
    print("  Saved: shap_values.csv")
    if fraud_mask.any():
        print(f"\nTop {top_n} SHAP features for fraud event #{idx}:")
        for i,item in enumerate(top_shap_features(shap_values[idx], feature_cols, top_n), 1):
            sign = "▲" if item["shap_value"]>0 else "▼"
            print(f"  {i}. {sign} {item['feature']:<35} {item['shap_value']:+.4f}")
    print("\nExplainer complete.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default="data/raw/login_events.csv")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--top-n", type=int, default=5)
    args = p.parse_args()
    run_explainer(args.data_path, args.n_samples, args.top_n)

if __name__ == "__main__":
    main()
