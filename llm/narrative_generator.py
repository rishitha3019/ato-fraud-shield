import argparse, json, os, sys, time
import anthropic
import numpy as np
import pandas as pd

ARTIFACTS_DIR = "model/artifacts"
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 400
RISK_THRESHOLDS = {"HIGH": 0.75, "MEDIUM": 0.40, "LOW": 0.0}

def risk_level(p):
    if p >= 0.75: return "HIGH"
    if p >= 0.40: return "MEDIUM"
    return "LOW"

def format_feature(feature, shap_value, raw_value):
    descriptions = {
        "is_new_device": f"Device fingerprint: {'UNKNOWN — never seen before for this account' if raw_value==1 else 'recognized device'}",
        "km_from_last_login": f"Distance from last login: {raw_value:.0f} km ({'extremely far' if raw_value>500 else 'far' if raw_value>100 else 'nearby'})",
        "velocity_kmh": f"Travel velocity implied: {raw_value:.0f} km/h ({'physically impossible' if raw_value>900 else 'suspicious' if raw_value>300 else 'plausible'})",
        "velocity_kmh_log": None,
        "impossible_velocity_flag": f"Impossible travel flag: YES — velocity exceeds commercial flight speed" if raw_value==1 else None,
        "failed_attempts_1h": f"Failed login attempts (last 1h): {int(raw_value)} ({'high — credential stuffing pattern' if raw_value>=5 else 'normal'})",
        "failed_attempts_6h": f"Failed login attempts (last 6h): {int(raw_value)}",
        "failed_attempts_24h": f"Failed login attempts (last 24h): {int(raw_value)}",
        "failed_attempts_ratio": f"Recent failure concentration: {raw_value:.2f} ({'concentrated burst' if raw_value>0.5 else 'spread out'})",
        "actions_per_minute": f"Session action rate: {raw_value:.1f} actions/min ({'very high — automated scraping pattern' if raw_value>20 else 'high' if raw_value>10 else 'normal'})",
        "time_to_first_action_sec": f"Time to first action after login: {raw_value:.1f}s ({'instant — bot-like' if raw_value<2 else 'fast' if raw_value<5 else 'normal'})",
        "session_duration_sec": f"Session duration: {raw_value:.0f}s ({'very short — smash-and-grab pattern' if raw_value<30 else 'short' if raw_value<60 else 'normal'})",
        "hour_of_day": f"Login hour: {int(raw_value):02d}:00 ({'unusual overnight hours' if raw_value<6 else 'business hours'})",
        "hour_deviation_from_mean": f"Deviation from user typical login hour: {raw_value:.1f} hours ({'significant' if raw_value>4 else 'minor'})",
        "is_night": f"Night-time login: {'yes' if raw_value==1 else 'no'}",
        "is_weekend": f"Weekend login: {'yes' if raw_value==1 else 'no'}",
        "hours_since_last_login": f"Time since last login: {raw_value:.1f} hours",
        "account_age_days": f"Account age: {int(raw_value)} days ({'new account' if raw_value<90 else 'established'})",
        "day_of_week": None,
    }
    return descriptions.get(feature, f"{feature.replace('_',' ').title()}: {raw_value:.3f}")

def build_prompt(fraud_probability, top_features, raw_event):
    level = risk_level(fraud_probability)
    signal_lines = []
    for item in top_features:
        feat = item["feature"]
        raw_val = raw_event.get(feat, 0.0)
        desc = format_feature(feat, item["shap_value"], float(raw_val))
        if desc:
            sign = "+" if item["shap_value"] > 0 else "-"
            signal_lines.append(f"  [{sign}] {desc}")
    signals_text = "\n".join(signal_lines) if signal_lines else "  No strong signals."
    return f"""You are a fraud analyst AI at a fintech company. A machine learning model has flagged a login event.

Risk level: {level}
Fraud probability: {fraud_probability:.1%}

Top signals driving this decision:
{signals_text}

Write a concise fraud risk narrative (3-5 sentences) for the fraud operations team. The narrative should:
1. State the risk level and probability clearly upfront
2. Explain the 2-3 most important signals in plain English
3. Identify the most likely attack pattern (credential stuffing, impossible travel, ATO session abuse, or slow-burn ATO)
4. End with a recommended action (block, step-up authentication, or monitor)

Be direct and specific. Write for a fraud analyst who needs to act quickly.
Do not use bullet points. Write in flowing prose."""

def generate_narrative(fraud_probability, top_features, raw_event, client=None, retries=2):
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY='your-key'")
        client = anthropic.Anthropic(api_key=api_key)
    prompt = build_prompt(fraud_probability, top_features, raw_event)
    for attempt in range(retries + 1):
        try:
            message = client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            return {
                "narrative": message.content[0].text.strip(),
                "risk_level": risk_level(fraud_probability),
                "fraud_probability": round(fraud_probability, 4),
                "model_used": MODEL,
                "top_features": top_features,
            }
        except Exception as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return {
                "narrative": f"[API error: {str(e)}] Risk: {risk_level(fraud_probability)}, Probability: {fraud_probability:.1%}",
                "risk_level": risk_level(fraud_probability),
                "fraud_probability": round(fraud_probability, 4),
                "model_used": MODEL,
                "top_features": top_features,
                "error": str(e),
            }

def load_sample_event(event_idx=0, top_n=5):
    shap_path = os.path.join(ARTIFACTS_DIR, "shap_values.csv")
    data_path = "data/raw/login_events.csv"
    if not os.path.exists(shap_path):
        raise FileNotFoundError(f"Run model/explainer.py first.")
    shap_df = pd.read_csv(shap_path)
    data_df = pd.read_csv(data_path).reset_index(drop=True)
    fraud_mask = shap_df["is_fraud"] == 1
    fraud_shap = shap_df[fraud_mask].reset_index(drop=True)
    fraud_data = fraud_shap.copy()  # use shap_df directly since it is already sampled
    idx = min(event_idx, len(fraud_shap) - 1)
    row_shap = fraud_shap.iloc[idx]
    row_data = fraud_data.iloc[idx] if idx < len(fraud_data) else fraud_shap.iloc[idx]
    fraud_prob = float(row_shap["fraud_probability"])
    shap_cols = [c for c in shap_df.columns if c.startswith("shap_")]
    feature_cols = [c.replace("shap_", "") for c in shap_cols]
    shap_pairs = [(f, float(row_shap[f"shap_{f}"])) for f in feature_cols]
    shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = [{"feature": f, "shap_value": v, "direction": "increases fraud risk" if v>0 else "decreases fraud risk"} for f, v in shap_pairs[:top_n]]
    raw_event = {feat: float(row_data.get(feat, 0.0)) for feat in feature_cols}
    return fraud_prob, top_features, raw_event

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-idx", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not set.")
        print("Run: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)
    print(f"Loading fraud event #{args.event_idx}...")
    fraud_prob, top_features, raw_event = load_sample_event(args.event_idx, args.top_n)
    print(f"Fraud probability: {fraud_prob:.1%}  |  Risk: {risk_level(fraud_prob)}")
    print(f"\nTop {args.top_n} SHAP features:")
    for item in top_features:
        sign = "▲" if item["shap_value"]>0 else "▼"
        print(f"  {sign} {item['feature']:<35} {item['shap_value']:+.4f}")
    print("\nGenerating narrative via Claude API...")
    client = anthropic.Anthropic(api_key=api_key)
    result = generate_narrative(fraud_prob, top_features, raw_event, client)
    print(f"\n{'='*60}")
    print(f"RISK LEVEL: {result['risk_level']}")
    print(f"FRAUD PROBABILITY: {result['fraud_probability']:.1%}")
    print(f"\nNARRATIVE:")
    print(result["narrative"])
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

