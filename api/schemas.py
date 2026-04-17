from typing import Optional
from pydantic import BaseModel, Field

class LoginEvent(BaseModel):
    event_id: str = Field(..., description="Unique event identifier")
    user_id: int = Field(..., description="User account ID")
    hours_since_last_login: float = Field(0.0, ge=0)
    km_from_last_login: float = Field(0.0, ge=0)
    velocity_kmh: float = Field(0.0, ge=0)
    is_new_device: int = Field(0, ge=0, le=1)
    failed_attempts_1h: int = Field(0, ge=0)
    failed_attempts_6h: int = Field(0, ge=0)
    failed_attempts_24h: int = Field(0, ge=0)
    session_duration_sec: float = Field(300.0, ge=0)
    actions_per_minute: float = Field(3.0, ge=0)
    time_to_first_action_sec: float = Field(10.0, ge=0)
    account_age_days: int = Field(365, ge=0)
    hour_deviation_from_mean: float = Field(0.0, ge=0)
    hour_of_day: int = Field(12, ge=0, le=23)
    day_of_week: int = Field(0, ge=0, le=6)

class ShapFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str

class PredictionResponse(BaseModel):
    event_id: str
    user_id: int
    fraud_probability: float
    risk_level: str
    predicted_fraud: bool
    threshold_used: float
    top_features: list[ShapFeature]

class ExplainResponse(PredictionResponse):
    narrative: str
    model_used: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    llm_available: bool
    uptime_seconds: float

class ModelInfoResponse(BaseModel):
    feature_count: int
    features: list[str]
    threshold: float
    optimal_f1_threshold: float
    roc_auc: float
    avg_precision: float
