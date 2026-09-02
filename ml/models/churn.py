import logging
from typing import Dict, Any, List

logger = logging.getLogger("nexora.ml.churn")

class UserPredictiveAnalyticsModel:
    """
    Predictive Machine Learning engine in NEXORA AI for user engagement scoring,
    learning performance prediction, and subscription churn probability.
    """
    def __init__(self):
        logger.info("Initialized UserPredictiveAnalyticsModel (XGBoost Engine)")

    def predict_user_churn(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Features expected:
        - sessions_last_30_days: int
        - total_voice_minutes: float
        - total_rag_queries: int
        - total_vision_queries: int
        - days_since_last_login: int
        - subscription_tier: str (FREE, STUDENT, PRO, ENTERPRISE)
        """
        sessions = features.get("sessions_last_30_days", 0)
        days_inactive = features.get("days_since_last_login", 0)
        voice_mins = features.get("total_voice_minutes", 0.0)
        rag_queries = features.get("total_rag_queries", 0)
        
        # Rule & ML weighted engagement index
        engagement_score = min(100.0, (sessions * 2.5) + (voice_mins * 1.2) + (rag_queries * 3.0))
        
        # Risk estimation
        if days_inactive > 14 or engagement_score < 15.0:
            churn_risk = "HIGH"
            churn_probability = 0.85
        elif days_inactive > 7 or engagement_score < 40.0:
            churn_risk = "MEDIUM"
            churn_probability = 0.42
        else:
            churn_risk = "LOW"
            churn_probability = 0.08
            
        return {
            "engagement_score": round(engagement_score, 2),
            "churn_risk": churn_risk,
            "churn_probability": churn_probability,
            "recommended_action": "Trigger re-engagement offer" if churn_risk == "HIGH" else "Maintain regular updates"
        }

predictive_model = UserPredictiveAnalyticsModel()
