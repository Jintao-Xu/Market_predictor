# Open Source Gold Price Prediction — Reference List

A curated list of open source repositories and research papers for gold price analysis and prediction.

---

## GitHub Repositories

### Multi-Model / Comprehensive

| Repo | Description |
|------|-------------|
| [MohammadvHossein/predict-gold-price](https://github.com/MohammadvHossein/predict-gold-price) | Python project with Linear Regression, Random Forest, XGBoost, LSTM, and CNN; uses historical data + economic indicators. Most similar in scope to this project. |
| [ayoub-mg/Gold-Price-Forecasting](https://github.com/ayoub-mg/Gold-Price-Forecasting) | Multiple ML models including LSTM for next-day closing price prediction. |

### Random Forest / Classical ML

| Repo | Description |
|------|-------------|
| [MYoussef885/Gold_Price_Prediction](https://github.com/MYoussef885/Gold_Price_Prediction) | NumPy, Pandas, sklearn, Random Forest; clean and well-documented. MIT licensed. |
| [jigyasaG18/Gold-Price-Prediction-Project-using-Machine-Learning](https://github.com/jigyasaG18/Gold-Price-Prediction-Project-using-Machine-Learning) | Uses SPX, USO, SLV, EUR/USD as features — macro-indicator approach similar to this project. |
| [R-Mahesh45/Gold-Price-Prediction-Using-Machine-Learning](https://github.com/R-Mahesh45/Gold-Price-Prediction-Using-Machine-Learning) | Random Forest + ARIMA, deployed with Streamlit for real-time forecasting. |
| [jigyasaG18/Gold-Price-Prediction-Project-using-Machine-Learning](https://github.com/jigyasaG18/Gold-Price-Prediction-Project-using-Machine-Learning) | Predicts GLD using SPX, USO, SLV, EUR/USD indicators with Random Forest Regressor. |

### LSTM / Deep Learning

| Repo | Description |
|------|-------------|
| [kittinan/predict-gold-price](https://github.com/kittinan/predict-gold-price) | Simple LSTM baseline; useful for comparing against local LSTM/BiLSTM/GRU results. |

### Kaggle Notebooks

| Notebook | Description |
|----------|-------------|
| [Gold Price Prediction by using LSTM](https://www.kaggle.com/code/eisgandar/gold-price-prediction-by-using-lstm) | LSTM-based notebook with feature engineering ideas. |

---

## Research Papers

| Paper | Key Takeaway |
|-------|--------------|
| [Analysis and forecasting of daily global gold price: an SARIMA-LSTM approach with Random Forest technique (2025)](https://www.tandfonline.com/doi/full/10.1080/23322039.2025.2568969) | Hybrid SARIMA + LSTM + Random Forest captures both linear and nonlinear dependencies. Relevant for understanding why pure LSTM underperforms. |
| [Gold price prediction by a CNN-Bi-LSTM model along with automatic parameter tuning (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10919698/) | CNN-Bi-LSTM with auto tuning; explains how to improve BiLSTM performance — relevant since BiLSTM is a weak model in Exp 9. |
| [Forecasting gold price using machine learning methodologies (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0960077923009803) | Benchmark study across multiple ML approaches on gold price data. |
| [Deep learning systems for forecasting crude oil and precious metals prices](https://jfin-swufe.springeropen.com/articles/10.1186/s40854-024-00637-z) | XGBoost and LightGBM leveraged for gold and bitcoin price forecasting. |

---

## Tutorials / Guides

| Resource | Description |
|----------|-------------|
| [Gold Price Prediction: Step-by-Step Guide Using Python ML (QuantInsti)](https://blog.quantinsti.com/gold-price-prediction-using-machine-learning-python/) | Practical walkthrough with feature engineering and model evaluation. |
| [PyCaret: Predicting Gold Prices Using Machine Learning](https://github.com/pycaret/pycaret-docs/blob/main/learn-pycaret/official-blog/predicting-gold-prices-using-machine-learning.md) | Low-code ML pipeline using PyCaret for gold price prediction. |
| [GitHub Topic: gold-price-prediction](https://github.com/topics/gold-price-prediction) | Aggregated list of all public repos tagged with this topic. |

---

## Notes for This Project

- Local LSTM/BiLSTM/GRU models perform poorly (Sharpe ~0 or negative in Exp 9). The SARIMA-LSTM hybrid paper and CNN-Bi-LSTM paper may explain why pure deep learning underperforms on this task and how hybridization helps.
- Classical models (Ridge, Lasso, ElasticNet) currently lead on Sharpe ratio. The `jigyasaG18` repo uses similar macro indicators (SPX, USO, SLV, EUR/USD) and could be a useful feature engineering reference.
- The `MohammadvHossein` repo is the most comparable end-to-end project covering the same model range as this codebase.
