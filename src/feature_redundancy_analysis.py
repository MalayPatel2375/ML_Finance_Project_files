import pandas as pd
import numpy as np

df = pd.read_csv("data/model_input/model_interaction_data.csv")

elite_features = [

    "Daily_Return",

    "RSI_14",

    "Return_Lag1",

    "Return_Lag2",

    "Return_Lag3",

    "Price_VS_SMA20",

    "Distance_From_Low",

    "Volatility_Ratio",

    "Volatility_Trend",

    "Return_Volume",

    "VolRation_Volume"
]

corr_matrix = (df[elite_features].corr().round(3))

print("\nCORRELATION MATRIX")
print("---------------------")
print(corr_matrix)

high_corr = []

for i in range(len(corr_matrix.columns)):

    for j in range(i):

        corr = abs(

            corr_matrix.iloc[i, j]
        )

        if corr > 0.80:

            high_corr.append({

                "Feature_1":
                corr_matrix.columns[i],

                "Feature_2":
                corr_matrix.columns[j],

                "Correlation":
                corr
            })

high_corr_df = pd.DataFrame(
    high_corr
)

high_corr_df.to_csv("results/elite_feature_corrrr.csv", index=False)

print("\nHIGH CORRELATIONS")
print("----------------------")

print(high_corr_df)

uniqueness_scores = []

for feature in elite_features:

    correlations = (

        corr_matrix[feature]

        .drop(feature)

        .abs()
    )

    avg_corr = correlations.mean()

    uniqueness = 1 - avg_corr

    uniqueness_scores.append({

        "Feature": feature,

        "Uniqueness_Score":
        round(uniqueness, 4)
    })

uniqueness_df = pd.DataFrame(
    uniqueness_scores
)

uniqueness_df = (

    uniqueness_df

    .sort_values(
        by="Uniqueness_Score",
        ascending=False
    )
)

print("\nFEATURE UNIQUENESS")
print("----------------------")

print(uniqueness_df)

uniqueness_df.to_csv("results/elite_feature_uniqueness.csv", index=False)