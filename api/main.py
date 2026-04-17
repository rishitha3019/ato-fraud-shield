import json, os, sys, time
from contextlib import asynccontextmanager
import anthropic
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.narrative_generator import generate_narrative, risk_level
from api.schemas import LoginEvent, PredictionResponse, ExplainResponse, HealthResponse, ModelInfoResponse

MODEL_STATE = {"model": None, "meta": None, "anthropic_client": None, "loaded_at": None}
ARTIFACTS_DIR = "model/artifacts"

def load_model():
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(ARTIFACTS_DIR, "xgb_model.json"))
    with open(os.path.join(ARTIFACTS_DIR, "feature_meta.json")) as f:
        meta = json.load(f)
    return model, meta

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ATO fraud model...")
    MODEL_STATE["model"], MODEL_STATE["meta"] = load_model()
    MODEL_STATE["loaded_at"] = time.time()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        MODEL_STATE["anthropic_client"] = anthropic.Anthropic(api_key=api_key)
        print("Anthropic client initialized.")
    else:
        print("WARNING: ANTHROPIC_API_KEY not set.")
    print(f"Model ready. Features: {len(MODEL_STATE['meta']['feature_cols'])}")
    yield
    print("Shutting down.")

app = FastAPI(title="ATO Fraud Shield API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def engineer_features(event: LoginEvent) -> pd.DataFrame:
    d = event.model_dump()
    d["velocity_kmh_log"] = float(np.log1p(d.get("velocity_kmh", 0)))
    d["failed_attempts_ratio"] = d.get("failed_attempts_1h", 0) / (d.get("failed_attempts_24h", 0) + 1)
    d["impossible_velocity_flag"] = int(d.get("velocity_kmh", 0) > 900)
    d["is_weekend"] = int(d.get("day_of_week", 0) >= 5)
    d["is_night"] = int(d.get("hour_of_day", 12) < 6)
    feature_cols = MODEL_STATE["meta"]["feature_cols"]
    return pd.DataFrame([{col: d.get(col, 0.0) for col in feature_cols}])[feature_cols]

def get_top_shap_features(X, n=5):
    import shap
    explainer = shap.TreeExplainer(MODEL_STATE["model"])
    shap_values = explainer.shap_values(X)
    feature_cols = MODEL_STATE["meta"]["feature_cols"]
    pairs = sorted(zip(feature_cols, shap_values[0]), key=lambda x: abs(x[1]), reverse=True)[:n]
    return [{"feature": f, "shap_value": round(float(v), 4), "direction": "increases fraud risk" if v > 0 else "decreases fraud risk"} for f, v in pairs]

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    model_loaded = MODEL_STATE["model"] is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        llm_available=MODEL_STATE["anthropic_client"] is not None,
        uptime_seconds=round(time.time() - MODEL_STATE["loaded_at"], 1) if MODEL_STATE["loaded_at"] else 0,
    )

@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    if not MODEL_STATE["meta"]:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    meta = MODEL_STATE["meta"]
    return ModelInfoResponse(
        feature_count=len(meta["feature_cols"]), features=meta["feature_cols"],
        threshold=meta["threshold"], optimal_f1_threshold=meta["optimal_f1_threshold"],
        roc_auc=meta["roc_auc"], avg_precision=meta["avg_precision"],
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Fraud Detection"])
async def predict(event: LoginEvent):
    if MODEL_STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        X = engineer_features(event)
        fraud_prob = float(MODEL_STATE["model"].predict_proba(X)[0, 1])
        threshold = MODEL_STATE["meta"].get("optimal_f1_threshold", 0.5)
        top_features = get_top_shap_features(X, n=5)
        return PredictionResponse(
            event_id=event.event_id, user_id=event.user_id,
            fraud_probability=round(fraud_prob, 4), risk_level=risk_level(fraud_prob),
            predicted_fraud=fraud_prob >= threshold, threshold_used=threshold,
            top_features=top_features,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/explain", response_model=ExplainResponse, tags=["Fraud Detection"])
async def predict_explain(event: LoginEvent):
    if MODEL_STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if MODEL_STATE["anthropic_client"] is None:
        raise HTTPException(status_code=503, detail="LLM unavailable — ANTHROPIC_API_KEY not set.")
    try:
        X = engineer_features(event)
        fraud_prob = float(MODEL_STATE["model"].predict_proba(X)[0, 1])
        threshold = MODEL_STATE["meta"].get("optimal_f1_threshold", 0.5)
        top_features = get_top_shap_features(X, n=6)
        narrative_result = generate_narrative(fraud_prob, top_features, event.model_dump(), client=MODEL_STATE["anthropic_client"])
        return ExplainResponse(
            event_id=event.event_id, user_id=event.user_id,
            fraud_probability=round(fraud_prob, 4), risk_level=risk_level(fraud_prob),
            predicted_fraud=fraud_prob >= threshold, threshold_used=threshold,
            top_features=top_features, narrative=narrative_result["narrative"],
            model_used=narrative_result["model_used"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explain error: {str(e)}")
