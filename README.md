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