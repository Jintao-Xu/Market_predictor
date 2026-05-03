# Model Architecture Analysis Report
## APS1052H Option 6 (CAD/USD) & Gold Prediction Pipelines

---

## 1. Current State Summary

| | CAD/USD Weekly | Gold Daily |
|---|---|---|
| **Target** | Frac-diff log(CADUSD), d≈0.4 | Frac-diff log(Gold), d≈0.3 |
| **Samples** | ~1,002 | ~Daily since COT coverage |
| **Features** | ~29–41 (COT, TA-Lib, VIX, 10Y, lags) | ~35 (COT gold, DXY, VIX, 10Y/30Y, TA, lags) |
| **Best CV model** | XGBoost k=5, MSE 0.000184 | Same pipeline, same lineup |
| **Neural net results** | LSTM: overfits (MSE 0.0034±0.005); SimpleRNN: MSE 0.028 | Same architecture, same issues expected |

### Current Neural Net Architecture (both notebooks)

```
LSTM:      Sequential → LSTM(32, dropout=0.2) → Dense(1)   epochs=30, batch=16
SimpleRNN: Sequential → SimpleRNN(32, dropout=0.2) → Dense(1)   epochs=30, batch=16
MLP:       MLPRegressor(hidden_layer_sizes=(64,32), alpha=0.01)
```

---

## 2. Core Problem Diagnosis

### Why LSTM Overfits on CAD/USD
- **~1,002 weekly samples** is borderline-small for LSTM. After TimeSeriesSplit into 5 folds, each training window may be as small as **150–400 rows** — far below LSTM's data appetite.
- A single-layer LSTM(32) has ~(32×(32+features+1)) ≈ **3,000–6,000 trainable parameters** against a few hundred samples. This is a **3–20× overparameterization ratio** relative to training set size.
- Dropout=0.2 is too light; current regularization does not compensate.
- **LSTM was designed for sequences of 50–500 timesteps per sample.** With TIMESTEPS=4, you're feeding it very short windows — the gating mechanism brings no benefit over simpler models.

### Why SimpleRNN Fails (MSE 0.028)
- SimpleRNN has **no gating**, so gradients vanish over even 4-step windows.
- It has nearly the same parameter count as LSTM without the memory cell advantage.
- Literature consensus (2021–2025): SimpleRNN is explicitly inferior to GRU/LSTM on financial return data. Your result matches.

### Why XGBoost Wins
- Gradient boosting is **naturally suited to tabular, feature-rich, small-to-medium datasets**.
- Feature selection (k=5) eliminates noise — LSTM sees all 29–41 features, XGBoost only 5 curated ones.
- XGBoost has no sequential inductive bias, which is actually a strength here: **fractionally differenced returns are nearly IID** (that's the point of frac-diff — to achieve stationarity), so sequential memory adds noise, not signal.

---

## 3. Architecture Recommendations

### 3.1 For Both Pipelines — Neural Network Improvements

#### Option A: Replace SimpleRNN → GRU (Drop-in, High Impact)

```python
# GRU: ~30% fewer parameters than LSTM, same API, better short-sequence performance
from tensorflow.keras.layers import GRU

def build_gru(units=32, dropout=0.3, recurrent_dropout=0.2, n_features=...):
    model = Sequential([
        GRU(units, input_shape=(TIMESTEPS, n_features),
            dropout=dropout,              # input dropout
            recurrent_dropout=recurrent_dropout),  # recurrent dropout (key addition)
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model
```

**Why GRU over SimpleRNN:** GRU keeps the reset/update gate mechanism (capturing short-term dependencies) while being lighter than LSTM. Literature shows GRU ≈ LSTM accuracy with 20–30% fewer params on short sequences. **Expected: significantly better than SimpleRNN, more stable than LSTM.**

#### Option B: Improved LSTM with Recurrent Dropout

```python
def build_lstm_v2(units=32, dropout=0.35, recurrent_dropout=0.2, n_features=...):
    model = Sequential([
        LSTM(units, input_shape=(TIMESTEPS, n_features),
             dropout=dropout,
             recurrent_dropout=recurrent_dropout,  # regularizes hidden-to-hidden weights
             kernel_regularizer=l2(0.001)),         # L2 on input weights
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001, decay=1e-4), loss='mse')
    return model
```

**Key changes vs current:** `dropout 0.2 → 0.35`, add `recurrent_dropout=0.2`, add `kernel_regularizer=l2(0.001)`, increase `epochs 30 → 50` with EarlyStopping(patience=10).

#### Option C: TCN (Temporal Convolutional Network) — Best Alternative

```python
# pip install keras-tcn
from tcn import TCN

def build_tcn(filters=32, kernel_size=3, n_features=...):
    model = Sequential([
        TCN(nb_filters=filters, kernel_size=kernel_size, dilations=[1,2,4],
            use_skip_connections=True, dropout_rate=0.3,
            input_shape=(TIMESTEPS, n_features)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
```

**Why TCN:** Dilated causal convolutions capture temporal patterns **without** vanishing gradients. Parallelizable (trains faster than RNN). Research shows 78% MAE reduction vs ARIMA, outperforms vanilla LSTM on structured financial data with <1000 samples. **This is the strongest architectural upgrade available.**

---

### 3.2 Task-Specific Recommendations

#### CAD/USD Weekly (option6_v2_timed.ipynb) — Priority Order

| Rank | Model | Expected Impact | Notes |
|---|---|---|---|
| 1 | **XGBoost + k=5 SelectKBest** | Already best, keep | The IID nature of frac-diff returns favors this |
| 2 | **Random Forest** (current) | Good baseline | Bagging reduces variance, solid runner-up |
| 3 | **GRU** (replace SimpleRNN) | MSE likely 0.005–0.015 | Better than SimpleRNN, still won't beat XGB |
| 4 | **LSTM v2** (recurrent dropout + L2) | Reduce variance | May reduce CV std from ±0.005 |
| 5 | **TCN** (dilated, 2–3 blocks) | Potential MSE ~0.003 | Architectural upgrade if neural net is required |
| ✗ | SimpleRNN | Confirmed worst | Retire it |
| ✗ | Transformer | Overkill | Too many params for 1002 weekly samples |

**Key insight for CAD/USD:** The fractionally differenced series has minimal temporal structure left by design — this limits all sequential models. XGBoost should remain dominant. Neural architectures are mainly relevant for satisfying the assignment's requirement of Keras models, not for prediction quality.

---

#### Gold Daily (gold_price_prediction.ipynb) — Priority Order

| Rank | Model | Expected Impact | Notes |
|---|---|---|---|
| 1 | **XGBoost + SelectKBest** | Strong baseline | Same logic applies |
| 2 | **Bidirectional LSTM (BiLSTM)** | Best neural for gold | Daily frequency gives more samples; BiLSTM validated in gold prediction literature specifically |
| 3 | **CNN-LSTM** | Good for TA features | CNN extracts local patterns from RSI/ROC/MACD/BB windows, LSTM processes COT sequence |
| 4 | **GRU** | Replace SimpleRNN | Same improvement as CAD/USD |
| 5 | **Attention-LSTM** | Weights COT vs macro features | DXY inverse relationship with gold makes attention useful here |

**BiLSTM config for gold:**

```python
from tensorflow.keras.layers import Bidirectional

def build_bilstm(units=32, dropout=0.4, n_features=...):
    model = Sequential([
        Bidirectional(LSTM(units, return_sequences=False,
                           recurrent_dropout=0.2),
                      input_shape=(TIMESTEPS, n_features)),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model
```

Gold specifically benefits from BiLSTM because macro regime shifts (DXY direction changes, yield curve dynamics) have both forward and backward temporal patterns in the training data.

---

## 4. Hyperparameter Tuning Table

| Parameter | Current | Recommended (Weekly) | Recommended (Daily) | Rationale |
|---|---|---|---|---|
| LSTM/GRU units | 32 | 32–48 | 32–48 | Minimize params; don't scale up |
| Dropout | 0.2 | **0.35–0.4** | **0.4–0.5** | Current too low for sample size |
| Recurrent dropout | 0 | **0.15–0.2** | **0.2** | Regularizes hidden-state weights (key missing ingredient) |
| L2 kernel | none | **0.001** | **0.0005–0.001** | Prevents weight explosion |
| Batch size | 16 | 16–24 | 20–32 | Small batches, noisy gradients help generalization |
| Epochs | 30 | **50** + EarlyStopping | **60** + EarlyStopping | EarlyStopping(patience=10, restore_best_weights=True) |
| Learning rate | Adam default | **0.001** with decay | **0.001** with decay | Adam(lr=0.001, decay=1e-4) |
| TIMESTEPS | 4 | **4–8** | **5–10** | More context for daily data |

---

## 5. Trading Performance Fix (Negative Sharpe Issue)

Both pipelines show neural nets generating **negative CAGR/Sharpe** despite low MSE. This is caused by:

1. **Magnitude underestimation:** Neural nets predict near-zero values (minimizing MSE), so signal direction may be right but magnitude too small to clear transaction costs.
2. **No volatility scaling:** Fixed position sizing penalizes low-confidence signals the same as high-confidence ones.

**Architecture-level fix:**

```python
# Add volatility scaling to prediction output
predicted_signal = model.predict(X_test)
rolling_vol = pd.Series(y_train).rolling(20).std().iloc[-1]
position = np.sign(predicted_signal) * (target_vol / rolling_vol)
```

**Model-level fix:** Train with a direction-aware loss instead of pure MSE:

```python
def directional_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    direction_penalty = tf.reduce_mean(tf.nn.relu(-y_true * y_pred))  # penalize wrong sign
    return mse + 0.3 * direction_penalty
```

---

## 6. Summary: What to Actually Do

### Immediate (drop-in swaps, minimal effort)
1. **Replace SimpleRNN → GRU** in both notebooks — better architecture, same code structure
2. **Increase LSTM dropout: 0.2 → 0.35**, add `recurrent_dropout=0.2`, add `kernel_regularizer=l2(0.001)`
3. **Increase epochs 30 → 50** and ensure `EarlyStopping(patience=10, restore_best_weights=True)` is used

### If extending gold_price_prediction.ipynb
4. **Add BiLSTM** as a model variant — gold daily data has enough samples to benefit
5. **Extend TIMESTEPS from 4 to 7** for daily gold (7 days ≈ 1 week of context)

### Don't bother
- **Transformer/BERT-style attention** — massively overparameterized for <1500 rows
- **More LSTM units (>64)** — more parameters = more overfitting
- **Deeper stacks (2+ LSTM layers)** — same issue; requires 5,000+ rows to regularize effectively

### Keep as-is
- **XGBoost + SelectKBest** is the correct winner for both tasks and empirically validated by 2024 literature on structured financial data. The neural net improvements above are about reducing their failure modes, not expecting them to beat XGBoost.

---

## 7. Implementation Changelog (Gold Notebook)

### Changes Made to `gold_price_prediction.ipynb`

| Change | Old | New | Reason |
|---|---|---|---|
| **LSTM dropout** | 0.2 | 0.35 | Better regularization for small sample |
| **LSTM recurrent_dropout** | none | 0.2 | Regularizes h→h weights |
| **LSTM kernel_regularizer** | none | l2(0.001) | Weight decay |
| **LSTM epochs** | 30 | 50 + EarlyStopping(10) | Let model converge properly |
| **SimpleRNN → GRU** | SimpleRNN(32) | GRU(32, recurrent_dropout=0.2) | Better gating, less vanishing gradient |
| **GRU dropout** | 0.2 | 0.3 | Slightly higher since GRU trains faster |
| **GRU epochs** | 30 | 50 + EarlyStopping(10) | Same as LSTM |
| **Added BiLSTM** | n/a | Bidirectional(LSTM(32)) + Dense(16) | Gold-specific upgrade, literature-backed |
| **TIMESTEPS** | 4 | 7 | 7 trading days = 1 week of context |

---

*Research sources: arxiv.org/abs/2405.08045 (FOREX neural architectures), arxiv.org/abs/2204.02623 (Attention CNN-LSTM XGBoost hybrid), ResearchGate BiLSTM gold study (2024), MDPI Fractal 2024 (TCN financial survey), comparative XGBoost vs LSTM benchmarks (2024).*
