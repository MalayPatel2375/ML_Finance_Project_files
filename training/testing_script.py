import pandas as pd

from training.cloud_training import train_model

df = pd.read_csv(
    "data/model_input/model_interaction_data.csv"
)

scaler, model, metrics = train_model(df)

print(metrics["accuracy"])