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
