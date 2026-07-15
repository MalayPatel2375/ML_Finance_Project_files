from pydantic import BaseModel

class PredictionRequest(BaseModel):

    Daily_Return: float
    Return_Lag1: float
    Return_Lag2: float
    Return_Lag3: float
    Volatility_Ratio: float
    Volatility_Trend: float
    Return_Volume: float
    VolRation_Volume: float

class PredictionResponse(BaseModel):

    Prediction: int
    Probability: float
    Confidence: float
    Model: str
    Version: str