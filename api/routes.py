from fastapi import APIRouter

from api.schemas import PredictionRequest
from api.schemas import PredictionResponse
from api.services import make_prediction

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    return make_prediction(
        request.model_dump()
    )

@router.get("/health")
def health():

    return{
        "status":"healthy",
        "model":"loaded",
        "scaler":"loaded"
    }
