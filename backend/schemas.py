from pydantic import BaseModel, Field, ConfigDict

class CustomerProfilePayload(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True
    )
    
    customer_ID: str = Field(..., alias="customer_ID")
    P_2_mean: float = Field(..., validation_alias="PAYMENT FACTOR METRIC (P_2 MEAN)")
    D_39_max: float = Field(..., validation_alias="DELINQUENCY METRIC (D_39 MAX)")
    B_1_last: float = Field(..., validation_alias="BALANCE VELOCITY METRIC (B_1 LAST)")

class RiskAssessmentResponse(BaseModel):
    customer_ID: str
    default_probability: float
    risk_tier: str
    recommended_credit_limit: float