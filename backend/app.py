from fastapi import FastAPI, HTTPException, status, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import json
import math
from typing import List, Dict, Any

# FIXED IMPORT: Correct top-level import matching the working directory structure
from schemas import CustomerProfilePayload, RiskAssessmentResponse

app = FastAPI(
    title="Amex Credit Evaluation Platform Pipeline Service Architecture",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage mock to simulate transaction auditing database records
AUDIT_TRAIL_DB: Dict[str, Dict[str, Any]] = {}

def calculate_metrics(p2: float, d39: float, b1: float):
    """Core simulation engine calculating underwriting risk probability rules."""
    composite_risk_score = ((1.0 - p2) * 0.40) + (d39 * 0.40) + (b1 * 0.20)
    default_prob = max(0.0, min(1.0, float(composite_risk_score)))
    
    if default_prob > 0.50:
        risk_tier = "High Risk"
        credit_limit = 2000.0
    elif default_prob > 0.15:
        risk_tier = "Medium Risk"
        credit_limit = 7500.0
    else:
        risk_tier = "Low Risk"
        credit_limit = 25000.0
        
    return round(default_prob, 4), risk_tier, credit_limit

def async_batch_processor(batch_id: str, items: List[Dict[str, Any]]):
    """Background execution runner mimicking offline batch processing loops."""
    processed_results = []
    for record in items:
        try:
            c_id = record.get("customer_ID", "UNKNOWN")
            p2 = float(record.get("PAYMENT FACTOR METRIC (P_2 MEAN)", 0.5))
            d39 = float(record.get("DELINQUENCY METRIC (D_39 MAX)", 0.5))
            b1 = float(record.get("BALANCE VELOCITY METRIC (B_1 LAST)", 0.5))
            
            prob, tier, limit = calculate_metrics(p2, d39, b1)
            processed_results.append({
                "customer_ID": c_id,
                "default_probability": prob,
                "risk_tier": tier,
                "recommended_credit_limit": limit
            })
        except Exception:
            continue
            
    # Commit execution results directly into audit log trails matrix
    AUDIT_TRAIL_DB[batch_id] = {
        "status": "COMPLETED",
        "total_records": len(processed_results),
        "payloads": processed_results
    }

@app.get("/health", status_code=status.HTTP_200_OK)
async def service_healthcheck():
    """System health target for Docker orchestrator layers."""
    return {"status": "healthy", "service": "backend-engine"}

@app.post("/api/v1/predict-risk", response_model=RiskAssessmentResponse)
async def predict_risk(payload: CustomerProfilePayload):
    try:
        prob, tier, limit = calculate_metrics(payload.P_2_mean, payload.D_39_max, payload.B_1_last)
        
        # Save historical audit record entries dynamically
        AUDIT_TRAIL_DB[payload.customer_ID] = {
            "p2": payload.P_2_mean,
            "d39": payload.D_39_max,
            "b1": payload.B_1_last,
            "probability": prob
        }
        
        return RiskAssessmentResponse(
            customer_ID=payload.customer_ID,
            default_probability=prob,
            risk_tier=tier,
            recommended_credit_limit=limit
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict-batch", status_code=status.HTTP_202_ACCEPTED)
async def predict_batch(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Endpoint handling async processing tasks for multi-account batch uploads."""
    try:
        contents = await file.read()
        raw_data = json.loads(contents.decode("utf-8"))
        
        batch_id = f"BATCH_{len(AUDIT_TRAIL_DB) + 1}"
        AUDIT_TRAIL_DB[batch_id] = {"status": "PROCESSING", "total_records": len(raw_data)}
        
        background_tasks.add_task(async_batch_processor, batch_id, raw_data)
        return {"batch_id": batch_id, "status": "QUEUED", "records_received": len(raw_data)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid batch submission format: {str(e)}")

@app.get("/api/v1/explain/{customer_id}")
async def explain_risk(customer_id: str):
    """Explainable AI (XAI) routine computing feature weight attributions via mock SHAP values."""
    if customer_id not in AUDIT_TRAIL_DB:
        raise HTTPException(status_code=404, detail="Customer identity profile tag context not located inside audit tracking system.")
        
    record = AUDIT_TRAIL_DB[customer_id]
    
    # Deriving local attribute impact vectors mapping directly to the mathematical weights
    base_value = 0.35
    p2_effect = -float(record["p2"]) * 0.40
    d39_effect = float(record["d39"]) * 0.40
    b1_effect = float(record["b1"]) * 0.20
    
    return {
        "customer_ID": customer_id,
        "base_expected_value": base_value,
        "final_predicted_probability": record["probability"],
        "shap_attributions": [
            {"feature": "PAYMENT FACTOR METRIC (P_2 MEAN)", "shap_value": round(p2_effect, 4)},
            {"feature": "DELINQUENCY METRIC (D_39 MAX)", "shap_value": round(d39_effect, 4)},
            {"feature": "BALANCE VELOCITY METRIC (B_1 LAST)", "shap_value": round(b1_effect, 4)}
        ]
    }

@app.get("/api/v1/batch-status/{batch_id}")
async def get_batch_status(batch_id: str):
    """Returns the current processing execution status metrics for batch operations."""
    if batch_id not in AUDIT_TRAIL_DB:
        raise HTTPException(status_code=404, detail="Batch index tracking metric pointer not found.")
    return AUDIT_TRAIL_DB[batch_id]