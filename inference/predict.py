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

    results = X.copy()

    results["Prediction"] = predictions

    results["Probability"] = probabilities

    results["Confidence"] = np.maximum(probabilities, 1 - probabilities)

    return results

def predict_single(model, scaler, sample):

    if isinstance(sample, pd.Series):
        sample = sample.to_frame().T

    sample_scaled = scaler.transform(sample)

    probability = model.predict_proba(sample_scaled)[0][1]

    prediction = int(probability >= 0.5)

    confidence = max(probability, 1 - probability)

    return {
        "Prediction": prediction,
        "Probability": float(probability),
        "Confidence": float(confidence)
    }
