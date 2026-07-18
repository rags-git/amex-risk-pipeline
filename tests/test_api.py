import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_predict_risk_approved():
    payload = {
        "customer_ID": "test_low_risk_user",
        "P_2_mean": 0.95,
        "D_39_max": 0.01,
        "B_1_last": 0.01
    }
    response = client.post("/api/v1/predict-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["risk_tier"] == "Low Risk"
    assert data["risk_tier"] == "Low Risk"

def test_predict_risk_rejected():
    payload = {
        "customer_ID": "test_high_risk_user",
        "P_2_mean": 0.05,
        "D_39_max": 0.95,
        "B_1_last": 0.95
    }
    response = client.post("/api/v1/predict-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["risk_tier"] == "High Risk"
    assert data["risk_tier"] == "High Risk"
