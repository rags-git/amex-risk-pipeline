from pydantic import BaseModel, Field

class RiskRequest(BaseModel):
    customer_ID: str = Field(..., json_schema_extra={"example": "0000099db563072c3d5e7"})
    P_2_mean: float = Field(..., description="Payment component metric mean value", json_schema_extra={"example": 0.62})
    D_39_max: float = Field(..., description="Delinquency threshold maximum flag value", json_schema_extra={"example": 0.02})
    B_1_last: float = Field(..., description="Balance metric status balance velocity", json_schema_extra={"example": 0.01})
