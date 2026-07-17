from pydantic import BaseModel, Field

class CustomerProfilePayload(BaseModel):
    customer_ID: str = Field(..., example="0000099db563072c3d5e7")
    P_2_mean: float = Field(..., description="Payment component metric mean value", example=0.62)
    D_39_max: float = Field(..., description="Delinquency threshold maximum flag value", example=0.02)
    B_1_last: float = Field(..., description="Balance metric status balance velocity", example=0.01)

class RiskAssessmentResponse(BaseModel):
    customer_ID: str
    default_probability: float
    risk_tier: str
    recommended_credit_limit: float
    decision_status: str
