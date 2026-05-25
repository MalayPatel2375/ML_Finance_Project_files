import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/model_input/model_ready_data.csv")

df = df.drop(columns=["Unnamed: 0", "Date"])

correlation_matrix = df.corr()
correlation_pairs = correlation_matrix.unstack()

correlation_pairs = correlation_pairs[correlation_pairs != 1]

sorted_pairs = correlation_pairs.abs().sort_values(ascending=False)

sorted_pairs = sorted_pairs.drop_duplicates()

print("\nTOP CORRELATIONS")
print("------------------")
print(sorted_pairs.head(20))

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, cmap="coolwarm", center=0)
plt.title("Feature Correlation HeatMap")

plt.savefig("results/feature_corr_heatmap.png", bbox_inches="tight")

#Detecting problematic features
high_correlation_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        correlation_value = correlation_matrix.iloc[i, j]
        if abs(correlation_value) > 0.80:
            feature_1 = correlation_matrix.columns[i]
            feature_2 = correlation_matrix.columns[j]
            high_correlation_pairs.append((feature_1, feature_2, correlation_value))

print("\nHIGH CORRELATION FEATURES")
print("------------------------")
for pair in high_correlation_pairs:
    print(pair)

correlation_matrix.to_csv("results/feature_corr_matrix.csv")
