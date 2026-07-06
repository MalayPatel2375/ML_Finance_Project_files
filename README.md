# Financial ML Project

## Objective

Predict next-day stock movement using machine learning.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- yfinance
- AWS SageMaker (planned)

## Week 1 - Data Handling

## Current Progress

- Multi-ticker ingestion pipeline
- Data validation pipeline
- Structured raw/processed workflow

## Next Steps

- Feature engineering
- Predictive modeling
- Backtesting
- AWS deployment

## Week 2 - Feature Engineering

Implemented machine learning feature engineering pipeline for financial time-series data.

### Features Added

- Daily percentage returns
- Binary target variable for next-day prediction
- Simple Moving Averages (5, 10, 20)
- Rolling volatility indicators
- Price vs trend distance features
- Momentum indicators
- RSI (Relative Strength Index)
- Lagged returns and lagged closing prices

### Key Concepts Learned

- Time-series feature engineering
- Rolling window calculations
- Momentum and volatility modeling
- Supervised learning target construction
- Preventing ticker cross-contamination using groupby()
- Handling rolling-window NaN values
- Sequential financial data processing

## Current Data Pipeline

raw_data.csv

        ↓

validated_data.csv

        ↓

basic_features.csv

        ↓

technical_features.csv

        ↓

enhanced_features.csv

        ↓

model_ready_data.csv

        ↓

Logistic Regression

        ↓

Random Forest

## Week 2 — Financial ML Validation & Baseline Modeling

This phase focused on transforming engineered financial indicators into a validated, model-ready machine learning pipeline.

The primary objective was to build a realistic financial prediction workflow while avoiding common machine learning mistakes such as:

- data leakage
- improper train/test splitting
- overfitting
- misleading accuracy metrics

The workflow evolved from feature validation into supervised machine learning and model comparison.

---

## Dataset Validation & Integrity Checks

Before training any machine learning model, the dataset was thoroughly validated to ensure reliability and chronological correctness.

Validation procedures included:

- Missing value inspection
- Duplicate row detection
- Constant feature detection
- Correlation analysis
- Leakage detection
- Target distribution analysis

---

## Handling Structural NaNs

Several engineered indicators naturally produced missing values during initialization periods.

Examples:

- RSI_14
- Momentum_5
- Return_Lag1/2/3
- Close_Lag1/2/3

These NaNs were expected because rolling indicators and lagged features require historical observations before calculations become valid.

## Week 2 — Day 7  

## Probability-Based Backtesting, Threshold Tuning & Strategy Evaluation

This phase marked an important transition in the project:

From:

- simple classification predictions

To:

- probability-driven financial decision making

Instead of blindly predicting whether the market would move up or down, the system was extended to evaluate:

- model confidence
- probability thresholds
- trade filtering
- basic financial strategy performance

This stage introduced several foundational quantitative finance concepts including:

- probability calibration
- confidence threshold tuning
- strategy signal generation
- backtesting logic
- financial edge evaluation

---

## Objective_

The primary goal was to determine whether machine learning prediction confidence could improve financial decision quality.

The workflow focused on:

- generating probability predictions
- filtering low-confidence trades
- comparing strategy returns against average market returns
- evaluating whether higher confidence produced stronger financial performance

---

## Probability Predictions

Instead of generating only binary predictions:

```python
model.predict()
```

the pipeline was upgraded to use:

```python
model.predict_proba()
```

This allowed the model to output probability estimates such as:

| Probability Down | Probability Up |
|       ---        |       ---      |
| 0.42 | 0.58 |
| 0.30 | 0.70 |
| 0.49 | 0.51 |

These values represented: 
- model confidence
- estimated probability of upward market movement

---

## Extracting Bullish Confidence

The probability of upward movement was extracted using:

```python
up_probabilities = probabilities[:, 1]
```

This became the foundation for probability-based trading decisions.

---

## Confidence Threshold Tuning

A major concept introduced during this phase was:

## Threshold Tuning

Instead of trading every prediction, the system only executed trades when prediction confidence exceeded a predefined threshold.

Trading signals were generated using:

```python
signals = np.where(up_probabilities > threshold, 1, 0)
```

Where:
- 1 → enter trade
- 0 → stay out of market

---

## Why Threshold Tuning Matters

In real financial systems:

- not all predictions are equally valuable
- weak confidence predictions often introduce noise
- selective execution can improve robustness

Threshold tuning is widely used in:

- algorithmic trading
- fraud detection
- risk systems
- probability-based ML pipelines

---

## Strategy Return Logic

Market returns were linked to strategy signals using:

```python
test_df["Strategy_Return"] = (
    test_df["Strategy_Signal"]
    * test_df["Market_Return"]
)
```

This created a simplified trading simulation where:

- trades participated in market movement only when confidence exceeded the threshold
- low-confidence predictions remained inactive

---

## Important Backtesting Correction

An early implementation mistake involved compounding returns across multiple tickers simultaneously using cumulative multiplication.

This produced unrealistic exponential growth because:

- multiple assets were incorrectly treated as one continuous timeline
- ticker separation was ignored

The backtesting logic was corrected by replacing cumulative portfolio growth with:

- average strategy return analysis

This produced more realistic evaluation metrics for the current project architecture.

---

## Threshold Experiments

Multiple probability thresholds were tested systematically.

---

## Threshold Results

| Threshold | Avg Strategy Return | Trade Count |
|---|---|---|
| 0.50 | -0.001487 | 11382 |
| 0.51 | -0.002498 | 10600 |
| 0.52 | -0.004883 | 8416 |
| 0.55 | -0.035829 | 352 |
| 0.60 | -0.074095 | 13 |

Average market return:

```text
0.000602
```

---

## Major Findings

## 1. Higher Confidence Did NOT Improve Performance

One of the most important discoveries was:

> Higher prediction confidence did not produce better financial returns.

As thresholds increased:

- trade frequency dropped significantly
- strategy returns deteriorated sharply

This indicated that the model's probability estimates were poorly calibrated.

---

## 2. Confidence Calibration Problems

The model frequently assigned:

- high confidence
- poor-quality predictions

This introduced the concept of:

## Probability Calibration

A calibrated model should assign:

- higher probabilities to genuinely stronger predictions

However, in this financial dataset:

- confidence scores did not align with financial edge quality

This is a common challenge in financial machine learning due to:

- market noise
- unstable relationships
- weak predictive signals

---

## 3. Prediction Accuracy ≠ Profitability

Although earlier models achieved:

- ~52–53% directional accuracy

the strategy still produced:

- negative average returns

This demonstrated a critical quantitative finance principle:

> A model can predict direction slightly better than random while still failing financially.

Reasons include:

- incorrect magnitude prediction
- volatility effects
- asymmetric losses
- poor confidence calibration
- weak signal strength

---

## 4. Trade Frequency Tradeoff

Lower thresholds:

- generated more trades
- increased market participation

Higher thresholds:

- reduced trade count dramatically
- attempted to isolate “high-confidence” predictions

However:

- excessive filtering destroyed opportunity volume
- the remaining trades were often lower quality

This introduced the concept of:

## Precision vs Opportunity Tradeoff

A common optimization challenge in:

- quantitative trading
- machine learning systems
- probabilistic classification pipelines

---

## Key Concepts Learned

This stage introduced several advanced ML and quantitative finance concepts:

- Probability-based prediction
- Confidence scoring
- Threshold tuning
- Strategy signal generation
- Financial backtesting
- Trade filtering
- Probability calibration
- Precision vs opportunity tradeoff
- Strategy evaluation
- Financial edge analysis

---

## Important Code Components

## Probability Prediction

```python
probabilities = model.predict_proba(X_test_scaled)
```

---

## Confidence Extraction

```python
up_probabilities = probabilities[:, 1]
```

---

## Threshold-Based Trading Signals

```python
signals = np.where(up_probabilities > threshold, 1, 0)
```

---

## Strategy Return Generation

```python
test_df["Strategy_Return"] = (
    test_df["Strategy_Signal"]
    * test_df["Market_Return"]
)
```

---

## Average Strategy Evaluation

```python
avg_strategy_return = (
    test_df.loc[
        test_df["Strategy_Signal"] == 1,
        "Market_Return"
    ].mean()
)
```

---

## Overall Takeaway

This phase demonstrated that:

- building a financially useful ML system is significantly harder than achieving acceptable classification accuracy
- probability confidence alone is not sufficient for profitable decision making
- financial machine learning requires robust calibration, filtering, and risk-aware strategy design

The project successfully evolved from:

- simple predictive modeling

into:

- probability-aware quantitative strategy experimentation

This stage provided one of the most realistic lessons of the project so far:

> Financial prediction quality and financial profitability are not the same problem.

## Week 2 — Day 8  

## Strategy Visualization & Financial Diagnostics

This phase focused on visualizing strategy behavior and evaluating financial performance beyond traditional ML accuracy metrics.

The project moved from:

- simple prediction evaluation

to:

- financial strategy diagnostics
- risk analysis
- probability behavior analysis

---

## Objectives

Implemented:

- Equity curve visualization
- Market vs strategy comparison
- Probability distribution analysis
- Trading signal visualization
- Drawdown analysis
- Sharpe Ratio evaluation

---

## Equity Curve Analysis

Cumulative returns were generated using:

```
df["Cumulative_Market"] = (
    1 + df["Market_Return"]
).cumprod()

df["Cumulative_Strategy"] = (
    1 + df["Strategy_Return"]
).cumprod()
```

This allowed visualization of:

- strategy growth
- market comparison
- long-term performance behavior

---

## Probability Distribution Insights

Prediction probabilities from:

```
model.predict_proba()
```

were visualized using histograms.

Key finding:

- most probabilities clustered near 0.50
- confidence separation was weak
- threshold filtering provided limited advantage

This introduced the concept of:

## Probability Calibration

Higher confidence did not necessarily produce better financial predictions.

---

## Drawdown Analysis

Drawdown measures:

- decline from previous portfolio peak

Calculated using:

```
rolling_max = (
    df["Cumulative_Strategy"]
    .cummax()
)

drawdown = (
    df["Cumulative_Strategy"]
    - rolling_max
) / rolling_max
```

This helped evaluate:

- strategy stability
- crash severity
- financial risk exposure

---

## Sharpe Ratio

Implemented a basic Sharpe Ratio to evaluate:

- return relative to risk

```
sharpe = (
    df["Strategy_Return"].mean()
    /
    df["Strategy_Return"].std()
) * np.sqrt(252)
```

---

## Key Concepts Learned

- Equity curve analysis
- Financial visualization
- Strategy diagnostics
- Drawdown analysis
- Sharpe Ratio
- Probability calibration
- Risk-adjusted evaluation

---

## Major Takeaways

- Accuracy alone is not enough in financial ML
- Visualization reveals problems hidden by metrics
- Higher confidence predictions were not necessarily better
- Risk analysis is essential for evaluating trading systems

This phase pushed the project closer to a real quantitative research workflow instead of a basic ML classification project.

# WEEK 3 — DAYS 1 TO 3  
# Time-Series Validation, XGBoost & Hyperparameter Tuning

This phase focused on transforming the project from a basic machine learning workflow into a more realistic quantitative finance research pipeline.

The major focus areas included:
- eliminating data leakage
- implementing time-aware validation
- experimenting with advanced ensemble learning
- detecting overfitting
- understanding model generalization

---

# DAY 1 — TIME-SERIES VALIDATION

## Objective

Replace random train/test splitting with:
- chronological validation
- future-aware evaluation
- realistic forecasting structure

---

# Problem Discovered

The original ML workflow used:
```python
train_test_split()
```

This caused:
- random shuffling of financial data
- future information leaking into training
- unrealistic model evaluation

This is extremely dangerous in:
- finance
- forecasting systems
- time-series modeling

---

# Major Dataset Issue

During implementation:
```text
Date column was missing from model_ready_data.csv
```

Without timestamps:
- chronological splitting becomes impossible
- realistic backtesting breaks
- temporal validation cannot be performed

---

# Root Cause

The `Date` column was accidentally dropped during preprocessing while selecting model features.

Likely caused by:
```python
drop(columns=["Date"])
```

or numeric-only filtering.

---

# Solution

Rebuilt the dataset pipeline so that:
- Date remains in dataframe
- Ticker remains in dataframe
- only ML features are removed during training

Final structure:
```python
[
    "Date",
    "Ticker",
    feature columns,
    "Target"
]
```

---

# Additional Dataset Error

Unexpected column:
```text
Unnamed: 0
```

appeared in dataset.

---

# Root Cause

CSV index was accidentally saved:
```python
df.to_csv("file.csv")
```

instead of:
```python
df.to_csv("file.csv", index=False)
```

---

# Solution

Removed accidental index columns using:
```python
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
```

and ensured:
```python
index=False
```

for all future exports.

---

# Time-Series Validation Pipeline

Implemented:
- chronological sorting
- position-based slicing
- past-to-future validation

Key implementation:
```python
df = df.sort_values("Date")
```

and:
```python
split_index = int(len(df) * 0.8)
```

Training:
- first 80%

Testing:
- last 20%

---

# Major Concepts Learned

## Data Leakage

Future information accidentally enters training data and inflates performance.

---

## Temporal Validation

Validation must respect:
```text
time sequence
```

especially in:
- finance
- forecasting
- economic systems

---

## Production Thinking

Real systems:
- train on history
- predict unseen future

not:
```text
random shuffled reality
```

---

# Day 1 Results

## Accuracy
```text
52.99%
```

---

# Major Observation

The model heavily favored:
```text
bullish predictions
```

Meaning:
- strong UP prediction behavior
- weak bearish understanding

---

# Important Insight

Even after removing leakage:
- accuracy remained near ~53%

This suggested:
```text
market signal is weak and noisy
```

which is realistic in financial prediction systems.

---

# DAY 2 — XGBOOST MODEL

## Objective

Move from:
- linear learning
- simple ensembles

to:
```text
boosted nonlinear learning
```

using:
```text
XGBoost
```

---

# Why XGBoost Matters

XGBoost is heavily used in:
- finance
- Kaggle competitions
- fraud detection
- production ML systems

because it captures:
- nonlinear relationships
- sequential learning behavior
- boosting-based optimization

---

# XGBoost Architecture

Unlike Random Forest:
- trees are not independent

Instead:
- each tree learns from previous errors

This creates:
```text
sequential boosted learning
```

---

# XGBoost Results

## Accuracy
```text
51.73%
```

Lower than:
- Logistic Regression
- Random Forest

---

# Major Insight

More advanced models:
```text
do NOT guarantee better financial prediction
```

This is one of the biggest lessons in quantitative finance.

---

# Important Behavioral Difference

XGBoost:
- improved bearish prediction capability
- reduced bullish prediction dominance

Example:
```text
Correct bearish predictions increased significantly
```

Meaning:
```text
XGBoost learned more balanced directional behavior
```

---

# Feature Importance Analysis

Top features included:
- Daily_Return
- Return_Lag2
- RSI_14
- Momentum_5
- SMA features

---

# Major Insight

The model relied mostly on:
```text
short-term momentum behavior
```

instead of:
- fundamentals
- macroeconomics
- valuation metrics

---

# Another Important Observation

Volume importance decreased compared to Random Forest.

This suggested:
```text
price-action features mattered more than volume behavior
```

for this specific dataset.

---

# Major Concepts Learned

## Boosting

Sequential learning where new trees attempt to correct prior mistakes.

---

## Feature Importance

Understanding:
```text
what actually drives model predictions
```

---

## Nonlinear Learning

XGBoost can capture:
- complex interactions
- nonlinear relationships
- feature combinations

better than linear models.

---

# DAY 3 — HYPERPARAMETER TUNING

## Objective

Systematically optimize:
- model complexity
- learning behavior
- generalization ability

while controlling:
```text
overfitting
```

---

# Hyperparameters Tested

| Trees | Depth | Learning Rate |
|---|---|---|
| 50 | 2 | 0.01 |
| 100 | 4 | 0.05 |
| 200 | 6 | 0.10 |
| 300 | 8 | 0.20 |

---

# Major Discovery

As model complexity increased:
- training accuracy exploded
- testing accuracy declined

This demonstrated:
```text
classic financial overfitting
```

---

# Final Results Summary

| Complexity | Train Accuracy | Test Accuracy |
|---|---|---|
| Simple | 52.5% | 53.5% |
| Medium | 56.2% | 51.7% |
| Large | 73.6% | 50.9% |
| Extreme | 98.1% | 50.8% |

---

# Biggest Lesson

The simplest model:
```text
generalized best
```

while complex models:
```text
memorized noise
```

---

# Extreme Overfitting Example

Model:
```text
300 trees
depth 8
learning rate 0.20
```

achieved:
```text
98% training accuracy
```

but failed on future data.

---

# Why This Happened

Financial markets contain:
- randomness
- unstable patterns
- regime shifts
- noisy behavior

Complex models learned:
```text
historical noise
```

instead of:
```text
true predictive structure
```

---

# Major Concepts Learned

## Overfitting

Model memorizes training noise and fails on unseen data.

---

## Generalization

Goal is NOT:
```text
high training accuracy
```

Goal is:
```text
stable future performance
```

---

## Bias-Variance Tradeoff

Increasing complexity:
- reduces bias
- increases variance
- raises overfitting risk

---

## Financial ML Reality

Finance ML is largely:
```text
an overfitting management problem
```

not:
```text
an accuracy maximization problem
```

---

# Biggest Week 3 Lessons So Far

## 1. Time-aware validation is mandatory

Random splitting creates:
```text
fake financial performance
```

---

## 2. More complexity is not always better

Advanced models can:
```text
overfit extremely quickly
```

in financial systems.

---

## 3. Simpler models often generalize better

Especially when:
```text
true market signal is weak
```

---

## 4. Accuracy alone is misleading

Important factors include:
- directional balance
- robustness
- stability
- overfitting control
- future generalization

---

# Current Project Evolution

The project has now evolved from:
```text
basic ML experimentation
```

toward:
```text
quantitative research workflow thinking
```

The system now includes:
- time-series validation
- boosted ensemble learning
- feature importance analysis
- overfitting diagnostics
- realistic financial evaluation structure

This represents a major step toward:
- production-aware ML
- quantitative finance engineering
- real-world forecasting system design

# Week 3 — Day 4 to Day 7  
# Advanced Model Evaluation, Calibration, Explainability & Regime Analysis

---

# Day 4 — Time-Series Cross Validation

## Objective
Evaluate model stability across multiple time periods instead of relying on a single train-test split.

---

## Concepts Learned
- TimeSeriesSplit
- Sequential validation
- Temporal generalization
- Market regime dependency
- Model stability analysis

---

## Key Implementation

### Time-Series Cross Validation
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
```

### Fold-Based Training
```python
for train_index, test_index in tscv.split(X):
```

### Accuracy Tracking
```python
scores.append(accuracy)
```

### Final Evaluation
```python
average_score = np.mean(scores)
std_score = np.std(scores)
```

---

## Key Results

| Fold | Accuracy |
|---|---|
| 1 | 0.4987 |
| 2 | 0.5328 |
| 3 | 0.5423 |
| 4 | 0.4845 |
| 5 | 0.5413 |

### Final Metrics
- Average Accuracy: ~52%
- Standard Deviation: ~0.024

---

## Major Lessons
- Financial model performance changes across time periods.
- Different market conditions create different prediction behavior.
- Some market regimes completely weaken predictive power.
- Cross-validation exposed instability hidden in single train-test splits.

---

## Important Insight
The project shifted from:

> "How accurate is the model?"

to:

> "When does the model fail?"

---

# Day 5 — Probability Calibration & Confidence Analysis

## Objective
Analyze prediction confidence instead of only binary predictions.

---

## Concepts Learned
- Probability outputs
- Confidence thresholds
- Calibration analysis
- Conviction strength
- Prediction uncertainty

---

## Key Implementation

### Probability Predictions
```python
probabilities = model.predict_proba(X_test_scaled)
```

### Extract Bullish Probability
```python
up_probability = probabilities[:, 1]
```

### Manual Threshold Predictions
```python
predictions = (up_probability >= 0.50).astype(int)
```

### Confidence Calculation
```python
results_df["Confidence"] = np.where(
    results_df["Up_Probability"] >= 0.5,
    results_df["Up_Probability"],
    1 - results_df["Up_Probability"]
)
```

---

## Threshold Testing

| Threshold | Result |
|---|---|
| 0.50 | Trades found |
| 0.60 | No trades |
| 0.70 | No trades |
| 0.80 | No trades |

---

## Major Discovery
The model probabilities were concentrated near:

> 0.50

This revealed:
- weak confidence
- weak signal separation
- limited conviction
- compressed probability outputs

---

## Important Lesson
The absence of high-confidence trades was itself a valuable insight.

The model was essentially saying:

> "I am slightly confident, but not strongly confident."

---

# Day 6 — SHAP Explainability Analysis

## Objective
Understand WHY the model makes predictions.

---

## Concepts Learned
- SHAP values
- Explainable AI (XAI)
- Global feature importance
- Local prediction explanations
- Feature contribution analysis

---

## Key Implementation

### Create SHAP Explainer
```python
explainer = shap.Explainer(model)
```

### Generate SHAP Values
```python
shap_values = explainer(X_test_scaled)
```

### Global Feature Importance
```python
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean_shap_value": np.abs(shap_values.values).mean(axis=0)
})
```

### Local Prediction Analysis
```python
local_df = pd.DataFrame({
    "Feature": X.columns,
    "SHAP_contribution": local_values
})
```

---

## Key Findings

### Most Important Features
- Daily_Return
- Return_Lag1/2/3
- RSI_14
- Momentum-based indicators

### Weak / Ignored Features
- Open
- High
- Low
- Volume

---

## Major Insight
The model learned:

> short-term momentum behavior

instead of:

> raw price-level behavior

---

## Important Discovery
Engineered features were significantly more useful than raw OHLC data.

This confirmed:
- momentum dominance
- weak raw-price signal
- importance of feature engineering

---

# Day 7 — Market Regime Analysis

## Objective
Evaluate model behavior across different market conditions.

---

## Concepts Learned
- Market regimes
- Volatility segmentation
- Bullish vs bearish conditions
- Environment-aware ML
- Conditional model evaluation

---

# Volatility Regime Analysis

## Regime Logic
```python
df["Volatility_Regime"] = np.where(
    df["Volatility_5"] >= volatility_threshold,
    "High_volatility",
    "Low_volatility"
)
```

---

## Results

| Regime | Accuracy |
|---|---|
| Low Volatility | 53.32% |
| High Volatility | 53.79% |

---

## Insight
The model performed similarly across volatility conditions.

This suggested:
- moderate robustness
- limited volatility sensitivity
- stable but weak predictive edge

---

# Trend Regime Analysis

## Initial Problem
Original logic:
```python
>= 1
```

incorrectly classified the entire dataset as bearish.

---

## Root Cause
`Price_VS_SMA20` was centered near:

> 0

instead of:

> 1

because the feature represented:

> difference

rather than:

> ratio

---

## Corrected Logic
```python
df["Trend_Regime"] = np.where(
    df["Price_VS_SMA20"] >= 0,
    "Bullish",
    "Bearish"
)
```

---

## Final Results

| Regime | Accuracy |
|---|---|
| Bullish | 53.29% |
| Bearish | 54.02% |

---

## Major Insight
The model performed slightly better during bearish markets.

Possible reasons:
- sharper downside momentum
- stronger bearish volatility
- more detectable downward movement patterns

---

# Biggest Week 3 Lessons

## 1. Financial Signals Are Weak
Most predictions remain close to:

> 50% confidence

---

## 2. Model Stability Matters More Than Raw Accuracy
Higher complexity caused:
- severe overfitting
- unstable generalization
- weak robustness

---

## 3. Feature Engineering Is Critical
Engineered momentum features significantly outperformed raw price features.

---

## 4. Explainability Changed Model Understanding
SHAP analysis revealed:
- what the model actually learned
- which features were useful
- which features added noise

---

## 5. Market Context Matters
Different market regimes create different prediction behavior.

---

## 6. Assumption Validation Is Essential
Small threshold-definition mistakes completely changed analytical conclusions.

---

# Final Week 3 Conclusion

The project evolved from:

> basic predictive modeling

into:

> research-driven quantitative analysis

This week introduced:
- probabilistic forecasting
- explainable AI
- regime-aware modeling
- calibration analysis
- temporal validation
- model reliability evaluation

The system currently demonstrates:
- weak but persistent predictive edge
- momentum-driven behavior
- moderate regime stability
- low-confidence probability separation
- realistic financial ML characteristics

# WEEK 4 — DAYS 1 TO 3
# Feature Engineering, Noise Reduction & Ensemble Learning

---

# DAY 1 — FEATURE CORRELATION ANALYSIS

## Objective
Analyze relationships between features to identify:
- multicollinearity
- redundancy
- duplicated information
- noisy inputs

---

## Key Concepts Learned

### Correlation
Correlation measures how strongly two features move together.

- `+1` → almost identical movement
- `0` → unrelated
- `-1` → opposite movement

---

## Major Findings

Extremely high correlations were discovered across the dataset.

Examples:
- Open ↔ High → `0.9999`
- SMA_5 ↔ SMA_10 → `0.9997`
- Close_Lag1 ↔ Close_Lag2 → `0.9997`

This showed that many features were effectively:
- duplicates
- derived from the same price series
- carrying nearly identical information

---

## Important Insight

The model appeared to have many features, but most features contained overlapping information.

This explained:
- earlier overfitting behavior
- weak confidence predictions
- low SHAP importance for OHLC variables

---

## Financial ML Lesson

More features do NOT automatically create:
- better intelligence
- stronger prediction quality

In finance:
- cleaner signals
- lower redundancy
- unique information

often matter more than feature quantity.

---

# DAY 2 — FEATURE SELECTION & NOISE REDUCTION

## Objective
Reduce redundancy and build a cleaner feature set using:
- SHAP insights
- correlation analysis
- feature importance findings

---

## Selected Features

```python
selected_features = [

    "Daily_Return",
    "RSI_14",
    "Momentum_5",
    "Volatility_5",
    "Return_Lag1",
    "Return_Lag2",
    "Return_Lag3",
    "Price_VS_SMA5",
    "Price_VS_SMA20"
]
```

---

## Features Removed

Removed highly redundant variables:
- Open
- High
- Low
- Close
- SMA overlap features
- lagged close duplication

---

## Results

| Metric | Result |
|---|---|
| Original Features | 19 |
| Reduced Features | 9 |
| Accuracy | ~0.5355 |

Despite removing 10 features:
- model accuracy barely changed

---

## Important Discovery

This strongly suggested:
- many removed features added little real predictive value
- redundancy was extremely high
- simpler datasets can perform similarly

---

## Classification Behavior

The model became:
- highly bullish-biased
- weak at detecting bearish moves

Results:
- bullish recall → `0.99`
- bearish recall → `0.01`

This showed:
- cleaner models are not automatically smarter models
- removing features can reduce prediction diversity

---

## Major Lesson

Feature selection is NOT:
`remove everything correlated`

Instead:
`remove redundancy without destroying useful signal diversity`

---

# DAY 3 — ENSEMBLE LEARNING

## Objective
Combine multiple ML models into a single prediction system using:
- Logistic Regression
- Random Forest
- XGBoost

through:
- majority voting

---

## Ensemble Logic

Each model predicted:
- `0` → bearish
- `1` → bullish

Voting system:
- if at least 2/3 models predicted bullish → final prediction = 1
- otherwise → 0

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~0.53 |
| Random Forest | ~0.53 |
| XGBoost | ~0.52–0.53 |
| Ensemble | ~0.5355 |

---

## Agreement Analysis

Agreement Rate:
`84.37%`

Meaning:
- all three models made the same prediction most of the time

---

## Important Discovery

The ensemble produced almost no improvement because:
- all models learned similar market behavior
- all models relied on similar features
- all models captured similar weak signals

This meant:
`model diversity was very low`

---

## Classification Behavior

The ensemble remained:
- heavily bullish-biased
- weak at bearish detection

Results:
- bullish recall → `0.95`
- bearish recall → `0.06`

---

## Major Ensemble Lesson

Adding more models does NOT automatically create:
- stronger intelligence
- better predictions

Ensembles become powerful only when:
- models think differently
- models capture different patterns
- models make different mistakes

---

# OVERALL WEEK 4 (DAYS 1–3) INSIGHTS

## Core Discoveries

### 1. Extreme Feature Redundancy Exists
Most financial features were heavily correlated because they originated from the same underlying price series.

---

### 2. Simpler Models Can Perform Similarly
Reducing features from 19 → 9 barely changed accuracy.

This showed:
- many features were unnecessary
- redundancy was very high

---

### 3. Financial Signals Are Weak
Even multiple sophisticated models converged toward similar predictions and similar accuracy levels.

---

### 4. Bullish Bias Persisted Across All Models
Every model consistently favored bullish predictions.

This suggested:
- dataset-level bias
- weak bearish signal detection
- compressed probability separation

---

### 5. Ensemble Quality Depends on Diversity
Three similar models do not create a significantly smarter system.

Model diversity matters more than model quantity.

---

# Important Technical Concepts Learned

- Multicollinearity
- Feature Redundancy
- Noise Reduction
- Feature Selection
- Dimensionality Reduction
- Ensemble Learning
- Majority Voting
- Model Agreement Analysis
- Signal Quality
- Prediction Stability

---

# Biggest Takeaway

This week shifted the project from:
`building ML models`

toward:
`understanding how financial prediction systems actually behave`

The project is now evolving into:
- quantitative research
- signal analysis
- robustness evaluation
- decision-system engineering

rather than simple accuracy optimization.

# WEEK 4 — DAYS 4 TO 7
# Model Reliability, Robustness & Market Regime Analysis

---

# DAY 4 — PROBABILITY CALIBRATION

## Objective
Evaluate whether the model's probability estimates can be trusted.

---

## Results

| Metric | Value |
|----------|----------|
| Accuracy | 0.5355 |
| Brier Score | 0.249 |
| Average Probability | 0.5222 |
| Maximum Probability | 0.5751 |
| Minimum Probability | 0.4800 |
| High Confidence Predictions (>70%) | 0 |

---

## Key Findings

- Model accuracy remained around 53.5%.
- Probabilities stayed very close to 50%.
- The model never produced highly confident predictions.
- No prediction exceeded 70% confidence.
- Brier Score indicated weak probability quality.

---

## Important Lesson

The model is not overconfident.

Instead, it is:
- uncertain most of the time
- unable to identify strong trading opportunities
- producing compressed probability estimates

---

## Takeaway

The model has a small predictive edge, but confidence scores are not strong enough to support aggressive trading decisions.

---

# DAY 5 — FEATURE SENSITIVITY ANALYSIS

## Objective
Determine how dependent the model is on individual features.

---

## Method

Removed one feature at a time and retrained the model.

---

## Results

| Removed Feature | Accuracy |
|----------|----------|
| Daily_Return | 0.5371 |
| RSI_14 | 0.5363 |
| Return_Lag2 | 0.5363 |
| Volatility_5 | 0.5358 |
| Return_Lag1 | 0.5358 |
| Return_Lag3 | 0.5356 |
| Momentum_5 | 0.5355 |
| Price_VS_SMA5 | 0.5355 |
| Price_VS_SMA20 | 0.5355 |

---

## Key Findings

- Removing any feature did not reduce accuracy.
- Several feature removals slightly improved performance.
- No feature was essential to the model.
- Many features appear redundant.

---

## Important Lesson

The model is not relying on any single feature.

Instead:
- multiple weak signals contribute collectively
- feature overlap is extremely high
- redundancy exists throughout the dataset

---

## Takeaway

The primary limitation is signal quality rather than missing features.

---

# DAY 6 — REGIME-SPECIFIC FEATURE IMPORTANCE

## Objective
Determine whether feature importance changes across market conditions.

---

## Bullish Regime

Top Features:

| Feature | Importance |
|----------|----------|
| Return_Lag2 | 0.202 |
| Return_Lag1 | 0.174 |
| RSI_14 | 0.170 |
| Daily_Return | 0.158 |
| Price_VS_SMA20 | 0.153 |

---

## Bearish Regime

Top Features:

| Feature | Importance |
|----------|----------|
| Return_Lag2 | 0.243 |
| Return_Lag3 | 0.204 |
| Daily_Return | 0.159 |
| Volatility_5 | 0.158 |
| Return_Lag1 | 0.127 |

---

## Key Findings

Bullish Markets:
- RSI becomes more important.
- Trend-related features contribute more.

Bearish Markets:
- Volatility becomes more important.
- Longer return memory becomes more important.

Across both regimes:
- Return_Lag2 remained the strongest feature.

---

## Important Lesson

Feature importance is not fixed.

A feature may:
- perform well in one regime
- perform poorly in another

---

## Takeaway

Market conditions influence which signals the model uses.

---

# DAY 7 — REGIME-SPECIFIC MODELS

## Objective
Train separate models for bullish and bearish market environments.

---

## Results

| Regime | Samples | Accuracy |
|----------|----------|----------|
| Bullish | 32,301 | 0.5450 |
| Bearish | 28,629 | 0.5260 |

---

## Key Findings

Bullish Model:
- Outperformed the overall baseline.
- Produced the strongest accuracy observed this week.

Bearish Model:
- Underperformed the overall baseline.
- Market behavior remained harder to predict.

---

## Important Lesson

Not all market environments are equally predictable.

Bullish markets:
- exhibit stronger trends
- contain more persistent signals

Bearish markets:
- contain more volatility
- contain less stable patterns

---

## Takeaway

The model predicts upward market conditions more effectively than downward market conditions.

---

# WEEK 4 OVERALL INSIGHTS

## Major Discoveries

### 1. Weak Confidence
The model rarely produces strong confidence scores and remains close to 50/50 probabilities.

### 2. High Feature Redundancy
Removing individual features has little impact on performance.

### 3. Dynamic Feature Importance
Different market regimes rely on different signals.

### 4. Bull Markets Are Easier To Predict
The bullish model achieved the strongest performance.

### 5. Signal Quality Remains The Main Challenge
Model complexity is no longer the bottleneck.

The next improvements will likely come from:
- better features
- stronger signals
- alternative data sources
- improved market representations

rather than simply testing more algorithms.

---

# Technical Concepts Learned

- Probability Calibration
- Brier Score
- Confidence Reliability
- Feature Sensitivity Analysis
- Model Robustness
- Market Regimes
- Regime-Specific Feature Importance
- Adaptive Modeling
- Regime-Based Machine Learning
- Quantitative Research Methodology

---

# Biggest Takeaway

Week 4 shifted the project from:

"Which model is best?"

to:

"Under what market conditions does the model work best?"

This represents a transition from traditional machine learning toward quantitative research and market behavior analysis.

# Week 5: Advanced Feature Engineering & Feature Selection

## Overview

Week 5 focused on moving beyond traditional technical indicators and exploring advanced feature engineering techniques. The objective was to discover whether market structure, volume, volatility, and feature interactions could provide stronger predictive signals than conventional indicators.

Throughout the week, dozens of engineered features were created, tested, validated, and either retained or removed based on objective model performance.

---

# Day 1: Market Structure Features

## Objective

Create features that capture market breakout and breakdown behavior.

### New Features Created

- Breakout_5
- Breakout_10
- Breakdown_5
- Breakdown_10
- Distance_From_High
- Distance_From_Low

---

## Feature Statistics

### Missing Values

| Feature | Missing Values |
|----------|----------|
| Breakout_5 | 0 |
| Breakout_10 | 0 |
| Breakdown_5 | 0 |
| Breakdown_10 | 0 |
| Distance_From_High | 19 |
| Distance_From_Low | 19 |

---

### Feature Counts

| Feature | Positive Signals |
|----------|----------|
| Breakout_5 | 10,286 |
| Breakout_10 | 7,827 |
| Breakdown_5 | 6,705 |
| Breakdown_10 | 4,528 |

---

## Model Results

### Accuracy

```text
0.5256
```

---

## Top Features

| Feature | Importance |
|----------|----------|
| Daily_Return | 0.163 |
| Price_VS_SMA20 | 0.151 |
| Return_Lag3 | 0.133 |
| Return_Lag1 | 0.131 |
| Return_Lag2 | 0.125 |
| Distance_From_Low | 0.107 |

---

## Key Findings

### Successful Feature

```text
Distance_From_Low
```

provided useful information and entered the top feature rankings.

### Failed Features

```text
Breakout_5
Breakout_10
Breakdown_5
Breakdown_10
Distance_From_High
```

all received zero importance.

### Lesson Learned

Simple breakout and breakdown indicators were not informative.

The model preferred continuous distance-based information rather than binary breakout signals.

---

# Day 2: Volume-Based Features

## Objective

Determine whether volume behavior improves predictive performance.

### New Features Created

- RVOL
- Volume_Change
- Volume_Momentum
- Volume_VS_Avg

---

## Model Results

### Accuracy

```text
0.5257
```

---

## Top Features

| Feature | Importance |
|----------|----------|
| Volume_Change | 0.160 |
| Daily_Return | 0.153 |
| Price_VS_SMA20 | 0.138 |
| Return_Lag3 | 0.124 |
| Return_Lag1 | 0.122 |

---

## Key Findings

### Successful Feature

```text
Volume_Change
```

became the strongest newly created feature.

### Failed Features

```text
RVOL
Volume_Momentum
Volume_VS_Avg
```

received zero importance.

### Lesson Learned

Changes in volume matter more than raw volume levels.

---

# Day 3: Volatility Engineering

## Objective

Determine whether volatility behavior contains predictive information.

### New Features Created

- Volatility_Ratio
- Volatility_Expansion
- Volatility_Contraction
- Volatility_Trend

---

## Model Results

### Accuracy

```text
0.5258
```

---

## Top Features

| Feature | Importance |
|----------|----------|
| Volatility_Ratio | 0.105 |
| Volatility_Trend | 0.070 |

---

## Key Findings

### Successful Features

```text
Volatility_Ratio
Volatility_Trend
```

### Failed Features

```text
Volatility_Expansion
Volatility_Contraction
```

### Lesson Learned

Relative volatility changes are more useful than simple volatility state flags.

Markets react to changing volatility rather than volatility itself.

---

# Day 4: Feature Interaction Engineering

## Objective

Test whether relationships between features contain predictive information.

### Interaction Features Created

- RSI_Volume
- Return_Volume
- VolRation_Volume
- RSI_VolRation
- LowDist_Volume
- Return_VolRation

---

## Model Results

### Accuracy

```text
0.5258
```

---

## Top Features

| Feature | Importance |
|----------|----------|
| VolRation_Volume | 0.103 |
| Return_Volume | 0.088 |

---

## Key Findings

### Major Discovery

```text
VolRation_Volume
```

became the second most important feature in the model.

### Observation

Volume_Change lost all importance after interaction features were introduced.

This suggests volume becomes useful only when combined with other market conditions.

### Lesson Learned

Relationships between variables are more informative than many standalone indicators.

---

# Day 5: Elite Feature Selection

## Objective

Create a compact feature set containing only the strongest signals.

---

## Elite Model Results

### Accuracy

```text
0.5260
```

---

## Selected Features

```text
Daily_Return
RSI_14
Return_Lag1
Return_Lag2
Return_Lag3
Price_VS_SMA20
Distance_From_Low
Volatility_Ratio
Volatility_Trend
Return_Volume
VolRation_Volume
```

---

## Feature Stability Test

### Removed Features

| Feature Removed | Accuracy |
|----------|----------|
| RSI_14 | 0.5260 |
| Distance_From_Low | 0.5260 |
| Volatility_Trend | 0.5260 |
| VolRation_Volume | 0.5260 |
| Return_Volume | 0.5257 |

---

## Key Findings

Only Return_Volume caused any measurable decline.

Most elite features appeared partially redundant.

---

# Day 6: Feature Redundancy Analysis

## Objective

Determine whether surviving features overlap.

---

## Correlation Analysis

### High Correlations

```text
None Found
```

No feature pair exceeded:

```text
0.80
```

correlation.

---

## Feature Uniqueness Scores

| Feature | Uniqueness |
|----------|----------|
| VolRation_Volume | 0.980 |
| Return_Lag3 | 0.911 |
| Return_Volume | 0.906 |
| Return_Lag1 | 0.905 |
| Return_Lag2 | 0.903 |

---

## Lowest Uniqueness

| Feature | Score |
|----------|----------|
| Price_VS_SMA20 | 0.684 |

---

## Key Findings

The final feature set was highly diverse.

No major redundancy existed among surviving features.

---

# Day 7: Final Feature Pruning

## Objective

Identify the smallest feature set that preserves performance.

---

## Baseline Accuracy

```text
0.5260
```

---

## Pruning Results

| Removed Feature | Accuracy | Accuracy Change |
|----------|----------|----------|
| RSI_14 | 0.5260 | 0.0000 |
| Distance_From_Low | 0.5260 | 0.0000 |
| Price_VS_SMA20 | 0.5252 | -0.0008 |

---

## Final Decisions

### Removed

```text
RSI_14
Distance_From_Low
Price_VS_SMA20
```

### Retained

```text
Daily_Return
Return_Lag1
Return_Lag2
Return_Lag3
Volatility_Ratio
Volatility_Trend
Return_Volume
VolRation_Volume
```

---

# Final Week 5 Feature Set

```text
Daily_Return
Return_Lag1
Return_Lag2
Return_Lag3
Volatility_Ratio
Volatility_Trend
Return_Volume
VolRation_Volume
```

---

# Week 5 Features That Failed

```text
Breakout_5
Breakout_10
Breakdown_5
Breakdown_10
Distance_From_High

RVOL
Volume_Momentum
Volume_VS_Avg

Volatility_Expansion
Volatility_Contraction

RSI_Volume
RSI_VolRation

RSI_14
Distance_From_Low
Price_VS_SMA20
```

---

# Week 5 Features That Survived

```text
Daily_Return
Return_Lag1
Return_Lag2
Return_Lag3

Volatility_Ratio
Volatility_Trend

Return_Volume
VolRation_Volume
```

---

# Major Lessons Learned

## 1. Traditional Indicators Were Not the Strongest Signals

Many common indicators eventually became redundant or unimportant.

---

## 2. Relative Information Beats Absolute Information

Examples:

```text
Volatility_Ratio > Volatility_5

Volume_Change > RVOL

Distance_From_Low > Breakout Signals
```

---

## 3. Interaction Features Are Powerful

The strongest discoveries came from combining features:

```text
Return_Volume

VolRation_Volume
```

---

## 4. Simpler Models Can Perform Equally Well

Feature pruning reduced complexity while maintaining accuracy.

---

## 5. Market Context Matters

Returns, volatility, and volume become more informative when evaluated together rather than independently.

---

# Week 5 Conclusion

Week 5 transformed the project from a traditional indicator-based model into a feature-engineered machine learning system.

The strongest predictive signals were not standard technical indicators, but engineered features capturing relationships between:

- Returns
- Volatility
- Volume

The final result was a compact, interpretable feature set that maintained model performance while eliminating redundant information.

This feature set will serve as the foundation for Week 6, where the focus shifts from feature engineering to model optimization, probability calibration, threshold tuning, and trading-system evaluation.

# Week 6 – Probability Calibration, Threshold Optimization & Error Analysis

## Overview

Week 6 marked a shift from improving model accuracy to understanding **how the model behaves**. Instead of focusing solely on predictive performance, this week investigated prediction confidence, calibration, sample weighting, threshold optimization, and error analysis.

The primary objective was to determine whether the current model could be improved through tuning or whether the limitation originated from the available feature set.

---

# Day 1 – XGBoost Baseline Evaluation

## Objective

Train a baseline XGBoost classifier and evaluate its prediction behaviour before applying any calibration techniques.

## Work Completed

* Trained an XGBoost classifier
* Generated prediction probabilities
* Evaluated prediction distribution
* Computed confusion matrix
* Compared predicted and actual class distributions

## Results

| Metric                  |      Value |
| ----------------------- | ---------: |
| Accuracy                | **52.57%** |
| Bullish Prediction Rate | **98.33%** |
| Model Edge              |  **0.25%** |

## Key Findings

* The model predicted almost every observation as **Bullish**.
* Although the accuracy appeared acceptable, the prediction distribution was highly imbalanced.
* This suggested that accuracy alone was not an adequate evaluation metric.

---

# Day 2 – Probability Calibration

## Objective

Evaluate the quality of the model's predicted probabilities and determine whether confidence values were meaningful.

## Work Completed

* Generated prediction probabilities
* Calculated confidence values
* Computed Brier Score
* Tested multiple confidence thresholds

## Results

### Probability Statistics

| Metric              |  Value |
| ------------------- | -----: |
| Minimum Probability | 0.3955 |
| Maximum Probability | 0.5963 |
| Average Probability | 0.5260 |

### Brier Score

**0.2492**

### Threshold Performance

| Threshold | Signals | Accuracy |
| --------- | ------- | -------- |
| 0.50      | 12,172  | 52.57%   |
| 0.55      | 102     | 65.69%   |
| 0.60      | 1       | 100%     |

## Key Findings

* Prediction probabilities were tightly clustered around **0.52**.
* Very few predictions exceeded 0.55 confidence.
* Higher confidence predictions were more accurate but extremely rare.
* The model showed weak probability separation.

---

# Day 3 – High Confidence Prediction Analysis

## Objective

Determine whether the model's highest-confidence predictions were significantly more reliable.

## Work Completed

* Extracted high-confidence predictions
* Compared predicted and actual class distributions
* Investigated strongest predictions

## Results

| Metric                  |      Value |
| ----------------------- | ---------: |
| High Confidence Signals |    **102** |
| Percentage of Test Set  |  **0.84%** |
| Accuracy                | **65.69%** |
| Average Confidence      |   **0.56** |

## Key Findings

* High-confidence predictions were substantially more accurate.
* However, they represented less than **1%** of the entire dataset.
* The model rarely generated predictions with strong conviction.

---

# Day 4 – Sample Weight Optimization

## Objective

Reduce bullish prediction bias by adjusting bearish sample weights during training.

## Work Completed

* Tested multiple bearish sample weights
* Compared recall values
* Evaluated prediction balance
* Selected the optimal weight

## Best Configuration

**Bearish Sample Weight = 1.09**

## Results

| Metric                  |      Value |
| ----------------------- | ---------: |
| Accuracy                | **52.50%** |
| Bullish Recall          | **84.25%** |
| Bearish Recall          | **17.66%** |
| Bullish Prediction Rate | **83.34%** |

## Key Findings

* Sample weighting reduced bullish prediction bias.
* Prediction balance improved significantly.
* Overall model accuracy remained largely unchanged.
* Bearish market detection was still relatively weak.

---

# Day 5 – Threshold Optimization

## Objective

Determine whether modifying the classification threshold could improve prediction quality.

## Work Completed

* Evaluated thresholds between **0.45** and **0.60**
* Compared accuracy
* Compared recall
* Compared prediction distributions

## Results

**Best Threshold: 0.50**

Observations:

* Lower thresholds classified nearly every sample as bullish.
* Higher thresholds classified nearly every sample as bearish.
* No threshold produced balanced performance.

## Key Findings

* Threshold optimization alone could not improve model performance.
* Small threshold adjustments caused dramatic shifts in prediction behaviour.
* Model probabilities remained concentrated around 0.50.

---

# Day 6 – Error Analysis

## Objective

Identify where the model makes mistakes and determine whether incorrect predictions share common characteristics.

## Work Completed

* Compared correct and incorrect predictions
* Calculated false positives and false negatives
* Compared average feature values
* Compared prediction confidence
* Reviewed highest-confidence incorrect predictions

## Results

| Metric                |      Value |
| --------------------- | ---------: |
| Total Test Samples    | **12,172** |
| Correct Predictions   |  **6,390** |
| Incorrect Predictions |  **5,782** |
| False Positives       |  **4,779** |
| False Negatives       |  **1,003** |

### Confidence Comparison

| Prediction Type | Average Confidence |
| --------------- | -----------------: |
| Correct         |         **0.5068** |
| Incorrect       |         **0.5066** |

## Key Findings

* The model produced nearly **five times more false positives** than false negatives.
* Correct and incorrect predictions exhibited nearly identical feature values.
* Confidence values were virtually identical for correct and incorrect predictions.
* Even the model's most confident incorrect predictions had confidence values close to **55%**, indicating very limited certainty.

---

# Day 7 – Final Benchmark

## Objective

Create a final benchmark summarizing the model's performance before moving to cloud deployment and advanced feature engineering.

## Final Model Configuration

### Model

* XGBoost Classifier

### Final Feature Set

* Daily_Return
* Return_Lag1
* Return_Lag2
* Return_Lag3
* Volatility_Ratio
* Volatility_Trend
* Return_Volume
* VolRation_Volume

### Sample Weight

**1.09**

### Decision Threshold

**0.50**

---

## Final Benchmark

| Metric             | Result                  |
| ------------------ | ----------------------- |
| Accuracy           | ~52.5%                  |
| Bullish Recall     | ~84%                    |
| Bearish Recall     | ~18%                    |
| Average Confidence | ~50.7%                  |
| Prediction Bias    | Bullish                 |
| Primary Limitation | Weak Feature Separation |

---

# Major Learnings

## Accuracy Alone Is Misleading

Classification accuracy should always be evaluated alongside confidence, recall, prediction balance, and calibration.

---

## Probability Calibration Provides Deeper Insight

Prediction probabilities revealed substantially more information than accuracy alone.

---

## Sample Weighting Improved Prediction Balance

Sample weighting reduced bullish bias but produced only modest improvements in overall model behaviour.

---

## Threshold Optimization Has Practical Limits

Adjusting decision thresholds significantly changed prediction distributions but did not meaningfully improve predictive performance.

---

## Error Analysis Identified the True Bottleneck

Correct and incorrect predictions occurred under almost identical feature conditions.

This strongly suggests that the current feature set lacks sufficient predictive information for the model to confidently distinguish bullish and bearish market movements.

---

# Overall Conclusion

Week 6 demonstrated that the primary limitation of the current financial machine learning model is **not the XGBoost algorithm or its hyperparameters**, but the **quality and predictive strength of the engineered features**.

Across probability calibration, sample weighting, threshold optimization, and detailed error analysis, model performance consistently remained around **52–53% accuracy**. These results indicate that further parameter tuning is unlikely to provide substantial improvements.

Future progress should therefore focus on expanding the feature set with richer technical indicators, broader market context, and production-ready deployment.

---

# Next Steps

## Week 7 – AWS Cloud Integration

* AWS Cloud Practitioner (alongside the project)
* Amazon S3 integration
* EC2 deployment
* Automated daily data pipeline
* Automated prediction pipeline
* Cloud-based prediction storage

---

## Week 8 – Advanced Feature Engineering

* EMA indicators
* MACD
* ATR
* Bollinger Bands
* OBV
* Money Flow Index
* VIX
* S&P 500 features
* Sector ETF features
* Cross-asset indicators
* Time-based features
* Advanced price action indicators

---

# Week 6 Summary

Week 6 transformed the project from simply evaluating prediction accuracy into understanding **model reliability, confidence, and limitations**.

The experiments consistently demonstrated that the current performance ceiling is not caused by the learning algorithm itself, but by the information contained within the available features. This establishes a strong foundation for the next phase of the project, where cloud deployment and richer feature engineering will become the primary focus.

# Week 7: Cloud Integration & AWS Foundations

## Weekly Goal

After completing six weeks focused on machine learning, feature engineering, model evaluation, and probability analysis, the project transitioned into cloud engineering. The objective of Week 7 is to migrate the Financial Machine Learning pipeline toward a production-ready cloud architecture using AWS services.

---

# Day 1 – AWS Environment Setup

## Objective

Establish the cloud infrastructure required for future model training and deployment.

## Topics Learned

- Introduction to AWS Cloud
- AWS Console
- Identity and Access Management (IAM)
- IAM Users and Permissions
- AWS CLI
- Programmatic Authentication

## Tasks Completed

- Created AWS Account
- Created Administrator IAM User
- Created restricted IAM user (`ml_user_predict`)
- Configured AWS CLI
- Verified authentication using AWS STS
- Installed boto3
- Connected local project with AWS

## Engineering Decisions

Instead of using the root AWS account, all project operations use a dedicated IAM user following the Principle of Least Privilege.

---

# Day 2 – Amazon S3 & Intelligent Synchronization

## Objective

Replace manual dataset management with an automated cloud synchronization layer.

## AWS Services Used

- Amazon S3
- IAM
- boto3 SDK

## S3 Architecture

Created an organized S3 data lake.

```
Bucket

data/
    raw/
    processed/
    features/
    model_inputs/
    predictions/

models/
    xgboost/
    checkpoints/
    archived/

reports/

logs/

config/
```

## Cloud Module

Implemented a dedicated cloud package responsible for AWS operations.

```
cloud/

config.py

s3_client.py

uploader.py

downloader.py

manifest.py

utils.py

sync.py
```

## Synchronization Engine

Designed and implemented an intelligent synchronization engine.

Workflow:

Local Dataset

↓

Generate MD5 Hash

↓

Compare Manifest

↓

File Changed?

↓

Upload to Amazon S3

↓

Update Manifest

## Validation

Three synchronization scenarios were successfully tested.

### Test 1

Initial synchronization

Result:

- Uploaded: 11
- Skipped: 0

---

### Test 2

Run synchronization again

Result:

- Uploaded: 0
- Skipped: 11

---

### Test 3

Modify a single dataset

Result:

- Uploaded: 1
- Skipped: 10

The synchronization engine correctly uploaded only the modified dataset.

## Challenges

### AWS Credentials

Initially encountered:

NoCredentialsError

Resolved by configuring AWS CLI correctly.

### IAM Permissions

Encountered:

AccessDenied

Resolved by attaching AmazonS3FullAccess to the project IAM user.

## Engineering Decisions

- Created a dedicated cloud layer instead of embedding AWS code into ML scripts.
- Used MD5 hashing instead of timestamps to detect file changes.
- Maintained upload state through a manifest file.
- Organized S3 using a production-style data lake structure.

## Weekly Deliverablesss

✔ AWS Account

✔ IAM Configuration

✔ AWS CLI

✔ Amazon S3 Bucket

✔ Cloud Module

✔ Intelligent Synchronization Engine

✔ Automated Dataset Uploads

✔ Manifest Tracking

✔ Incremental Synchronization

## Key Learning

Week 7 marks the evolution of the project from a locally executed machine learning workflow into a cloud-enabled machine learning platform. Future work will leverage Amazon SageMaker to train and deploy models directly from datasets stored in Amazon S3.

                    Financial ML Platform

                  Yahoo Finance API
                          │
                          ▼
                Local Data Collection
                          │
                          ▼
                Feature Engineering
                          │
                          ▼
                Model Input Datasets
                          │
                          ▼
             Intelligent Cloud Sync Engine
                          │
                          ▼
                  Amazon S3 Data Lake
                  ┌─────────┴─────────┐
                  ▼                   ▼
         Local Development    SageMaker Studio
                  │                   │
                  └─────────┬─────────┘
                            ▼
                     XGBoost Training
                            │
                            ▼
                    Trained ML Models
                            │
                            ▼
                      Amazon S3 Storage

# Week 7 - Day 3 & Day 4
## Amazon SageMaker Integration & Cloud Model Training

---

# Overview

This milestone focused on transitioning the Financial Machine Learning project from a fully local development environment to a cloud-based machine learning workflow using Amazon Web Services (AWS). The objective was to establish a reproducible cloud training environment while maintaining consistency with the existing local ML pipeline.

---

# Day 3 - Amazon SageMaker Environment Setup

## Objectives

- Configure Amazon SageMaker Studio
- Validate AWS IAM permissions
- Connect SageMaker to the project S3 bucket
- Verify cloud access to project datasets
- Establish a reproducible cloud development environment

---

## Infrastructure Configuration

Successfully configured:

- Amazon SageMaker Studio Domain
- SageMaker User Profile
- SageMaker Execution Role
- AWS IAM Authentication
- Amazon S3 connectivity
- JupyterLab cloud environment

---

## S3 Integration

Verified successful connection to the project data lake.

Project Bucket:

```
malay-ml-sagemaker
```

Validated project directories:

```
data/
├── raw/
├── processed/
├── features/
├── model_input/
├── predictions/

models/
config/
logs/
reports/
```

---

## Dataset Validation

Successfully loaded the primary modeling dataset directly from Amazon S3.

Dataset Statistics

| Metric | Value |
|---------|------:|
| Samples | 60,859 |
| Features | 39 |
| Missing Values | 0 |
| Bullish Samples | 52.52% |
| Bearish Samples | 47.48% |

This confirmed that the cloud dataset exactly matched the local development dataset.

---

## Key Challenges Solved

### SageMaker SDK Compatibility

While configuring the notebook environment, differences between SageMaker SDK versions required adjustments to the development workflow.

Lessons learned:

- SDK version compatibility
- Python environment management
- Cloud package dependencies

---

### Project Structure Consistency

Identified a naming inconsistency between:

```
model_input
```

and

```
model_inputs
```

Standardized the project to use:

```
data/model_input/
```

across:

- Local project
- Amazon S3
- SageMaker notebooks
- Synchronization engine

---

## Day 3 Outcome

Successfully established a fully functional cloud development environment capable of reading project datasets directly from Amazon S3.

---

# Day 4 - Cloud Model Training

## Objectives

- Integrate GitHub with SageMaker
- Reuse existing training pipeline
- Train XGBoost inside SageMaker
- Validate cloud reproducibility
- Store model artifacts

---

## GitHub Integration

Migrated the project workflow from manual file management to Git-based development.

Current workflow:

```
VS Code
      │
      ▼
GitHub Repository
      │
      ▼
Amazon SageMaker Studio
```

This provides a single source of truth for future development.

---

## Reusable Training Pipeline

Created a reusable training module:

```
training/

├── __init__.py
├── cloud_training.py
├── evaluation.py
└── utils.py
```

The training pipeline now performs:

- Feature selection
- Time-based train/test split
- StandardScaler preprocessing
- Sample weighting
- XGBoost training
- Performance evaluation
- Probability prediction

---

## Model Configuration

Model:

```
XGBoost Classifier
```

Hyperparameters:

| Parameter | Value |
|-----------|-------|
| Estimators | 50 |
| Max Depth | 2 |
| Learning Rate | 0.01 |
| Sample Weight | 1.09 |

These settings reproduce the optimized configuration developed during Weeks 3–6.

---

## Cloud Training Results

Successfully trained the Financial ML model inside Amazon SageMaker.

Cloud evaluation matched the locally trained model, confirming that the cloud environment reproduces the existing machine learning pipeline without performance degradation.

Validated metrics:

- Accuracy
- F1 Score
- Classification Report
- Confusion Matrix
- Prediction Probabilities

---

## Model Artifact Management

Saved trained model artifacts:

```
models/

└── xgboost/

    xgboost_financial_model_v1.json

    training_metrics.json

    feature_names.json
```

Uploaded all artifacts to Amazon S3 for centralized storage.

---

# Architecture

```
                Local Development
                       │
                 Visual Studio Code
                       │
                       ▼
                  GitHub Repository
                       │
                       ▼
                Amazon SageMaker
                       │
              Cloud Model Training
                       │
                       ▼
                Amazon S3 Storage
                       │
                       ▼
              Versioned ML Artifacts
```

---

# Skills Developed

- Amazon SageMaker Studio
- Amazon S3 Data Management
- AWS IAM Roles & Permissions
- Cloud-based Machine Learning
- GitHub Integration
- Modular ML Pipeline Design
- Model Artifact Management
- Cloud Reproducibility
- MLOps Fundamentals

---

# Key Takeaways

- Successfully transitioned from local-only machine learning to cloud-based training.
- Established GitHub as the single source of truth for project development.
- Verified reproducible model training across local and cloud environments.
- Implemented reusable training modules to improve maintainability.
- Stored trained models and metadata in Amazon S3 for future deployment and versioning.

---

# Next Steps

- Implement SageMaker Training Jobs
- Automate cloud model training
- Build model inference pipeline
- Deploy trained models as scalable endpoints
- Introduce automated MLOps workflows