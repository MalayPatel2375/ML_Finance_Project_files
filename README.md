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
