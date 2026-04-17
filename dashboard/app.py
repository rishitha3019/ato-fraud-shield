import json, os
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

ARTIFACTS_DIR = "model/artifacts"
import streamlit as st
try:
    API_BASE = st.secrets["API_BASE_URL"]
except:
    API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Fraud Shield", layout="wide")

st.markdown("""
<style>
hr { border: none; border-top: 0.5px solid rgba(128,128,128,0.2); margin: 1.25rem 0; }
.narrative { font-size: 0.875rem; line-height: 1.7;
             border-left: 2px solid rgba(128,128,128,0.3);
             padding-left: 0.875rem; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_meta():
    p = os.path.join(ARTIFACTS_DIR, "feature_meta.json")
    if not os.path.exists(p): return {}
    with open(p) as f: return json.load(f)

@st.cache_data(ttl=60)
def load_shap():
    p = os.path.join(ARTIFACTS_DIR, "shap_values.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data(ttl=300)
def load_pr():
    p = os.path.join(ARTIFACTS_DIR, "pr_curve.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

def api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except: return None

def api_score(payload, explain=False):
    try:
        r = requests.post(
            f"{API_BASE}{'/predict/explain' if explain else '/predict'}",
            json=payload, timeout=15)
        return (r.json(), None) if r.status_code == 200 else (None, r.json().get("detail","error"))
    except Exception as e: return None, str(e)

meta = load_meta()
health = api_health()
pr = load_pr()

with st.sidebar:
    st.write("**fraud shield**")
    st.write("account takeover detection")
    st.write("---")

    if health:
        st.write(f"api · online · {health.get('uptime_seconds',0):.0f}s")
    else:
        st.write("api · offline")

    st.write("---")

    if meta:
        st.write(f"roc-auc · {meta.get('roc_auc',0):.4f}")
        st.write(f"avg precision · {meta.get('avg_precision',0):.4f}")
        st.write(f"features · {len(meta.get('feature_cols',[]))}")

    st.write("---")

    threshold = st.slider(
        "threshold",
        min_value=0.1, max_value=0.9,
        value=float(meta.get("optimal_f1_threshold", 0.5)),
        step=0.01
    )

    if pr is not None:
        row = pr.iloc[(pr["threshold"] - threshold).abs().idxmin()]
        st.write(f"precision · {row['precision']:.0%}")
        st.write(f"recall · {row['recall']:.0%}")
        st.write(f"f1 · {row['f1']:.0%}")

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.write("#### score a login")
    st.write("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        km = st.number_input("distance (km)", value=0.0, min_value=0.0, step=50.0, format="%.0f")
        velocity = st.number_input("velocity (km/h)", value=0.0, min_value=0.0, step=100.0, format="%.0f")
    with c2:
        is_new = st.selectbox("device", ["known", "new"])
        failed_1h = st.number_input("failed / 1h", value=0, min_value=0)
    with c3:
        hour = st.slider("login hour", 0, 23, 14)
        actions = st.number_input("actions / min", value=3.0, min_value=0.0, step=1.0)

    with st.expander("more signals"):
        d1, d2, d3 = st.columns(3)
        with d1:
            hours_since = st.number_input("hours since last login", value=24.0, min_value=0.0)
            failed_6h = st.number_input("failed / 6h", value=0, min_value=0)
        with d2:
            failed_24h = st.number_input("failed / 24h", value=0, min_value=0)
            time_first = st.number_input("time to first action (s)", value=10.0, min_value=0.0)
        with d3:
            session_dur = st.number_input("session duration (s)", value=300.0, min_value=0.0)
            hour_dev = st.number_input("hour deviation", value=0.0, min_value=0.0)
            account_age = st.number_input("account age (days)", value=365, min_value=0)
            dow = st.selectbox("day", ["mon","tue","wed","thu","fri","sat","sun"])

    use_llm = st.checkbox("include narrative", value=True)
    scored = st.button("score", type="primary", use_container_width=True)

with right:
    st.write("#### result")
    st.write("---")

    if scored:
        if not health:
            st.error("api offline — run: uvicorn api.main:app --port 8000")
        else:
            payload = {
                "event_id": "evt_001", "user_id": 1,
                "hours_since_last_login": float(hours_since),
                "km_from_last_login": float(km),
                "velocity_kmh": float(velocity),
                "is_new_device": 1 if is_new == "new" else 0,
                "failed_attempts_1h": int(failed_1h),
                "failed_attempts_6h": int(failed_6h),
                "failed_attempts_24h": int(failed_24h),
                "session_duration_sec": float(session_dur),
                "actions_per_minute": float(actions),
                "time_to_first_action_sec": float(time_first),
                "account_age_days": int(account_age),
                "hour_deviation_from_mean": float(hour_dev),
                "hour_of_day": int(hour),
                "day_of_week": ["mon","tue","wed","thu","fri","sat","sun"].index(dow),
            }
            with st.spinner(""):
                result, err = api_score(payload, explain=use_llm)

            if err:
                st.error(err)
            elif result:
                level = result["risk_level"]
                prob = result["fraud_probability"]
                action = {"HIGH":"block","MEDIUM":"step-up auth","LOW":"allow"}.get(level,"allow")

                c1, c2, c3 = st.columns(3)
                c1.metric("risk", level)
                c2.metric("probability", f"{prob:.1%}")
                c3.metric("action", action)

                st.write("---")

                feats = result["top_features"]
                names = [f["feature"].replace("_"," ") for f in feats]
                vals = [f["shap_value"] for f in feats]
                colors = ["#c04848" if v > 0 else "#1d9e75" for v in vals]

                fig, ax = plt.subplots(figsize=(5, 2.8))
                fig.patch.set_facecolor("none")
                ax.set_facecolor("none")
                bars = ax.barh(names, vals, color=colors, height=0.5)
                ax.axvline(0, color="#999", linewidth=0.7)
                ax.set_xlabel("shap value", fontsize=9, color="#999")
                ax.tick_params(labelsize=9, colors="#aaa")
                for s in ax.spines.values(): s.set_visible(False)
                ax.tick_params(left=False, bottom=False)
                ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8, color="#aaa")
                plt.tight_layout(pad=0.4)
                st.pyplot(fig, transparent=True)
                plt.close()

                if "narrative" in result and result["narrative"]:
                    st.write("---")
                    st.markdown(
                        f"<div class='narrative'>{result['narrative']}</div>",
                        unsafe_allow_html=True
                    )
    else:
        st.caption("fill in the form and hit score")

st.write("---")
st.write("#### recent events")

shap_df = load_shap()
if shap_df is None:
    st.caption("no data — run model/explainer.py first")
else:
    col_f, col_n = st.columns([1, 3])
    with col_f:
        fraud_only = st.checkbox("fraud only", value=False)
    with col_n:
        n = st.slider("rows", 10, 100, 20, label_visibility="collapsed")

    df = shap_df[shap_df["is_fraud"]==1].copy() if fraud_only else shap_df.copy()
    df = df.head(n).copy()
    df["flagged"] = (df["fraud_probability"] >= threshold).astype(int)
    df["risk"] = df["fraud_probability"].apply(
        lambda p: "HIGH" if p>=0.75 else ("MED" if p>=0.40 else "LOW"))
    show = [c for c in ["fraud_probability","risk","is_fraud","flagged","top_feature"] if c in df.columns]
    st.dataframe(
        df[show].style
          .background_gradient(subset=["fraud_probability"], cmap="RdYlGn_r", vmin=0, vmax=1)
          .format({"fraud_probability": "{:.1%}"}),
        use_container_width=True, height=300
    )
