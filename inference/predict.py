import numpy as np
import pandas as pd


def predict(
    model,
    scaler,
    X,
):

    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(X_scaled)[:, 1]

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
