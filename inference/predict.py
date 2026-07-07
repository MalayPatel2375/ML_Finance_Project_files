import pandas as pd
import numpy as np

def predict(model, scaler, data):
    probabilities = model.predict_proba(data)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    results = pd.DataFrame({

        "Prediction": predictions,

        "Probability": probabilities,

        "Confidence": np.maximum(
            probabilities,
            1 - probabilities
        )

    })

    return results
