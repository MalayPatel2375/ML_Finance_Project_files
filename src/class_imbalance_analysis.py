import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

features = [

    "Daily_Return",

    "Return_Lag1",

    "Return_Lag2",

    "Return_Lag3",

    "Volatility_Ratio",

    "Volatility_Trend",

    "Return_Volume",

    "VolRation_Volume"
]

df = pd.read_csv("data/model_input/model_interaction_data.csv")

target = "Target"

X = df[features]
y = df[target]

split_index = int(len(df)*0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier(
    n_estimator = 50,
    random_state = 42,
    max_depth = 2,
    learning_rate = 0.01
)

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

model_accuracy = accuracy_score(y_test, predictions)

print("\nActual Distribution")
print("--------------------")

actual_distribution = (y_test.value_counts(normalize=True)*100)

print(actual_distribution)

print("\nPrediction Distribution")
print("-------------------")

prediction_distribution = (pd.Series(predictions).value_counts(normalize=True)*100)

print(prediction_distribution)


## BULLISH BASELINE ##
bullish_predictions = [1] * len(y_test)

bullish_accuracy = accuracy_score(y_test, bullish_predictions)

print("\nBULLISH BASELINE")
print("-----------------------")

print(round(bullish_accuracy, 4))

print(round(model_accuracy, 4))

## MODEL EDGE ##
edge = model_accuracy - bullish_accuracy

print("\nMODEL EDGE")
print("-----------------------")

print(round(edge, 4))

## CONFUSION MATRIX ##
cm = confusion_matrix(y_test, predictions)

print("\nCONFUSION MATRIX")
print("--------------------------")

print(cm)

## EXTRA METRIC ##
bullish_prediction_rate = ((predictions.sum() / len(predictions)) *100)

print("\nBULLISH PREDICTION RATE")
print("-----------------")

print(round(bullish_prediction_rate, 2),"%")

