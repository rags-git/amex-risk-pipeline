import os
import joblib
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from backend.schemas import CustomerProfilePayload, RiskAssessmentResponse

model = None
MODEL_PATH = "models/amex_production_lgb.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Warning: Model load failure, falling back to runtime calculation: {e}")
    yield
    del model

app = FastAPI(
    title="Enterprise Credit Default & Risk Prediction Engine",
    description="Production Inference Microservice optimized for Enterprise Portfolio Underwriting.",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/api/v1/predict-risk", response_model=RiskAssessmentResponse, status_code=status.HTTP_200_OK)
async def predict_risk(payload: CustomerProfilePayload):
    try:
        prob_default = None
        
        if model is not None and hasattr(model, "feature_name"):
            try:
                feature_names = model.feature_name()
                input_data = {col: [0.0] for col in feature_names}
                input_df = pd.DataFrame(input_data)
                
                if "P_2_mean" in input_df.columns:
                    input_df["P_2_mean"] = [payload.P_2_mean]
                if "D_39_max" in input_df.columns:
                    input_df["D_39_max"] = [payload.D_39_max]
                if "B_1_last" in input_df.columns:
                    input_df["B_1_last"] = [payload.B_1_last]
                
                cat_cols = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]
                for col in input_df.columns:
                    base_name = col.split("_")[0]
                    if col.endswith("_last") and base_name in cat_cols:
                        input_df[col] = input_df[col].astype("category")
                
                prob_default = float(model.predict(input_df)[0])
            except Exception as inner_e:
                print(f"Model prediction shape error, falling back to algebraic calculation: {inner_e}")
                prob_default = None

        if prob_default is None:
            score = 2.0 * payload.D_39_max + 1.5 * payload.B_1_last - 2.5 * payload.P_2_mean
            prob_default = 1.0 / (1.0 + np.exp(-score))
            
        prob_default = max(0.0001, min(0.9999, float(prob_default)))
        
        if prob_default < 0.15:
            risk_tier = "Low Risk"
            decision_status = "APPROVED"
            credit_limit = 25000.00
        elif prob_default < 0.45:
            risk_tier = "Medium Risk"
            decision_status = "CONDITIONAL_APPROVAL"
            credit_limit = 10000.00
        else:
            risk_tier = "High Risk"
            decision_status = "REJECTED"
            credit_limit = 0.00
            
        return RiskAssessmentResponse(
            customer_ID=payload.customer_ID,
            default_probability=round(prob_default, 4),
            risk_tier=risk_tier,
            recommended_credit_limit=credit_limit,
            decision_status=decision_status
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Inference fatal execution failure: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
