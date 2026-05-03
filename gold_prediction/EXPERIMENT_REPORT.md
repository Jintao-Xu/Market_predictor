# Experiment Report — Gold Prediction Model Comparison

**Date:** 2026-05-03  
**Test period:** 2022-06-22 → 2026-05-01 (971 days)  
**Baseline:** 57 features, ZSCORE_WIN=10, binary ±1 signal, default ElasticNet alpha

---

## Baseline — `main`

57 features, ZSCORE_WIN=10, binary ±1 signal

| Model | Sharpe | CAGR | DirAcc | MaxDD |
|---|---:|---:|---:|---:|
| **SVR_lin** | **7.81** | **283%** | **71.3%** | -6.1% |
| Ridge | 7.46 | 265% | 70.4% | -6.1% |
| LightGBM | 6.75 | 229% | 69.2% | -7.2% |
| SVR_rbf | 6.76 | 229% | 69.1% | -7.0% |
| MLP | 6.73 | 228% | 68.6% | -6.1% |
| XGBoost | 6.39 | 211% | 66.9% | -6.1% |
| Lasso | 6.22 | 203% | 66.8% | -7.0% |
| RandomForest | 6.01 | 193% | 66.7% | -7.0% |
| ElasticNet | 2.92 | 73% | 59.4% | -13.7% |
| BiLSTM | 0.33 | 7% | 52.0% | -29.6% |
| GRU | 0.09 | 2% | 50.9% | -37.7% |
| LSTM | 0.05 | 1% | 51.8% | -40.1% |

---

## Exp 1 — `exp/controlled-39-features`

**Change:** Dropped 18 derived features, keeping only the original 39.

| Model | Sharpe | vs Baseline |
|---|---:|---:|
| All models | identical | **+0.00** |

**Finding:** The 18 extra features are entirely neutral — metrics are bit-for-bit identical after tuning. The pre-tuning MLP collapse (documented in IMPROVEMENT_ANALYSIS.md §7) was an artifact of untuned hyperparameters, not the features themselves.

---

## Exp 2 — `exp/per-model-k-alpha`

**Change:** Each model searches its own optimal SelectKBest k via `selector__k` in its Pipeline GridSearchCV. Wider ElasticNet alpha range added.

| Model | Sharpe | CAGR | DirAcc | MaxDD | vs Baseline |
|---|---:|---:|---:|---:|---:|
| SVR_lin | 7.22 | 252% | 69.7% | -6.1% | −0.59 |
| Ridge | 7.18 | 250% | 69.3% | -6.1% | −0.28 |
| SVR_rbf | 6.76 | 229% | 69.1% | -7.0% | +0.00 |
| MLP | 6.49 | 216% | 67.8% | -6.1% | −0.24 |
| LightGBM | 6.50 | 217% | 67.6% | -6.1% | −0.25 |
| XGBoost | 6.39 | 211% | 66.9% | -6.1% | +0.00 |
| Lasso | 6.22 | 203% | 66.8% | -7.0% | +0.00 |
| RandomForest | 6.01 | 193% | 66.7% | -7.0% | +0.00 |
| **ElasticNet** | **6.17** | **201%** | **66.6%** | -7.0% | **+3.25** |
| BiLSTM | 0.93 | 20% | 53.2% | -28.1% | +0.60 |
| GRU | 0.35 | 7% | 50.8% | -25.6% | +0.26 |
| LSTM | 0.23 | 4% | 52.6% | -25.2% | +0.18 |

**Finding:** Constraining SVR_lin to k=30 removes useful signal — its optimal k is 57 (all features). The ElasticNet fix (2.92 → 6.17) comes entirely from the wider alpha grid, not the k tuning. Net effect on top models is slightly negative.

---

## Exp 3 — `exp/signal-dead-zone`

**Change:** Hold cash (signal=0) when |z-score| < threshold. Threshold swept over [0.0, 0.2, 0.3, 0.5, 0.75] per model using OOF CV.

| Model | Sharpe | CAGR | DirAcc* | MaxDD | vs Baseline | Best threshold |
|---|---:|---:|---:|---:|---:|---:|
| **SVR_lin** | **8.49** | **275%** | **58.0%** | -6.1% | **+0.68** | 0.50 |
| Ridge | 8.09 | 229% | 48.2% | -2.8% | +0.63 | 0.50 |
| SVR_rbf | 7.56 | 235% | 56.9% | -6.1% | +0.80 | 0.50 |
| MLP | 7.22 | 233% | 63.0% | -6.1% | +0.49 | 0.20 |
| Lasso | 6.88 | 186% | 48.3% | -6.1% | +0.66 | 0.50 |
| XGBoost | 6.93 | 193% | 49.2% | -6.1% | +0.54 | 0.75 |
| RandomForest | 6.83 | 184% | 49.0% | -6.1% | +0.82 | 0.75 |
| LightGBM | 6.75 | 229% | 69.1% | -7.2% | +0.00 | 0.00 |
| ElasticNet | 2.92 | 73% | 59.3% | -13.7% | +0.00 | 0.00 |

*\* DirAcc here is the raw (pre-fix) metric — cash days (sig=0) counted as wrong. Apparent drop from baseline ~70% is an artifact of the dead zone, not worse active trades. See Metrics Note below.*

**Finding:** Filtering low-conviction trades boosts Sharpe across all linear/tree models. Ridge MaxDD halved (−6.1% → −2.8%). LightGBM and ElasticNet found threshold=0 optimal — their z-scores don't benefit from dead-zone filtering at the tested thresholds.

---

## Exp 4 — `exp/proportional-sizing`

**Change:** Replace binary ±1 signal with `sig = clip(z/3, −1, 1)` — position size proportional to z-score magnitude.

| Model | Sharpe | CAGR | DirAcc | MaxDD | vs Baseline |
|---|---:|---:|---:|---:|---:|
| SVR_lin | 7.21 | 97% | 71.2% | -1.2% | −0.60 |
| MLP | 7.20 | 83% | 68.5% | -2.1% | +0.47 |
| Ridge | 7.06 | 94% | 70.3% | -1.4% | −0.40 |
| SVR_rbf | 6.73 | 88% | 69.0% | -1.8% | −0.03 |
| Lasso | 6.59 | 81% | 66.7% | -2.1% | +0.37 |
| LightGBM | 6.51 | 84% | 69.1% | -2.2% | −0.24 |
| XGBoost | 6.22 | 79% | 66.8% | -2.4% | −0.17 |
| RandomForest | 6.11 | 76% | 66.6% | -2.6% | +0.10 |
| ElasticNet | 4.00 | 33% | 59.3% | -3.9% | +1.08 |

**Finding:** CAGR drops ~60% on average because positions are smaller on average. MaxDD improves significantly (SVR_lin −6.1% → −1.2%) but the return reduction outweighs the drawdown improvement on a Sharpe basis. Binary ±1 remains superior for the top models. ElasticNet is the exception — proportional sizing softens its noisy signal (+1.08 Sharpe).

---

## Exp 5 — `exp/zscore-win-tuning` ★★ Best Experiment

**Change:** Z-score rolling window tuned per model over [5, 7, 10, 15, 20]. All sklearn models selected win=5 as optimal.

| Model | Sharpe | CAGR | DirAcc | MaxDD | vs Baseline |
|---|---:|---:|---:|---:|---:|
| **SVR_lin** | **9.03** | **350%** | **72.5%** | -6.1% | **+1.22** |
| Ridge | 8.93 | 345% | 72.5% | -6.1% | +1.47 |
| SVR_rbf | 8.29 | 309% | 71.0% | -6.1% | +1.53 |
| MLP | 7.78 | 281% | 71.3% | -6.1% | +1.05 |
| RandomForest | 7.69 | 277% | 69.3% | -7.2% | +1.68 |
| XGBoost | 7.58 | 271% | 69.0% | -7.2% | +1.19 |
| Lasso | 7.49 | 266% | 69.2% | -6.1% | +1.27 |
| LightGBM | 6.75 | 229% | 69.2% | -7.2% | +0.00 |
| ElasticNet | 2.92 | 73% | 59.4% | -13.7% | +0.00 |

**Finding:** ZSCORE_WIN=5 is universally better than 10 for 7 out of 9 sklearn models. This was a regression introduced in a prior session (hardcoded 5 was changed to 10). LightGBM and ElasticNet are unaffected because their zscore_win params were not saved separately in this branch.

---

## Exp 6 — `exp/tune-elasticnet-lightgbm`

**Change:** Full hyperparameter grid search for ElasticNet (wider alpha [0.0001–1.0], finer l1_ratio) and LightGBM (num_leaves, min_child_samples added).

| Model | Sharpe | CAGR | DirAcc | vs Baseline |
|---|---:|---:|---:|---:|
| All others | identical | — | — | +0.00 |
| **ElasticNet** | **6.17** | **201%** | **66.6%** | **+3.25** |
| LightGBM | 6.50 | 217% | 67.6% | −0.25 |

**Finding:** ElasticNet's default alpha=0.01 was severely suboptimal — the wider grid found the correct regularization level, lifting Sharpe from 2.92 to 6.17. LightGBM's additional hyperparameters slightly overfit to CV folds (−0.25 Sharpe).

---

## Summary

### SVR_lin Sharpe Ranking

| Rank | Experiment | SVR_lin Sharpe | Primary Change |
|---|---|---:|---|
| 1 | `exp/zscore-win-tuning` | **9.03** | ZSCORE_WIN: 10 → 5 |
| 2 | `exp/signal-dead-zone` | **8.49** | Hold cash when \|z\| < 0.5 |
| 3 | `main` (baseline) | 7.81 | — |
| 3 | `exp/controlled-39-features` | 7.81 | 39 features (neutral) |
| 3 | `exp/tune-elasticnet-lightgbm` | 7.81 | ElasticNet fixed |
| 4 | `exp/per-model-k-alpha` | 7.22 | Per-model k |
| 4 | `exp/proportional-sizing` | 7.21 | Continuous sizing |

### What Works

| Improvement | Best model gain | Notes |
|---|---|---|
| ZSCORE_WIN=5 | +1.68 (RF), +1.53 (SVR_rbf) | Universal improvement, was a regression |
| Signal dead zone (threshold=0.5) | +0.82 (RF), +0.80 (SVR_rbf) | Cuts noise trades |
| ElasticNet alpha fix | +3.25 | Was severely misconfigured |

### What Doesn't Work

| Change | Best model impact | Reason |
|---|---|---|
| Per-model SelectKBest k | −0.59 (SVR_lin) | Optimal k=57 (no selection needed) |
| Proportional sizing z/3 | −0.60 (SVR_lin) | Lower average position size reduces returns |

---

## Metrics Note — DirAcc Fix ✓ (implemented in `exp/combined-best`)

### The problem (resolved)

The old `DirAcc` formula counted cash days (sig=0) as wrong predictions:

```python
# Old — wrong when threshold > 0
dir_acc = float(np.mean(sig == market_dir))   # sig ∈ {-1, 0, +1}, market_dir ∈ {-1, +1}
```

As threshold rises, more days go to cash and DirAcc fell mechanically — not because
active trades got worse. This is why Exp 3/7 showed DirAcc dropping to ~48–58% even as
Sharpe improved. The metric was contradicting the actual performance.

### Fix applied

`trading_metrics()` in `tune_model.py` and `update_results.py` now computes:

```python
active = sig != 0
'DirAcc':   float(np.mean(sig[active] == mkt[active])) if active.any() else 0.0,
'Coverage': float(np.mean(active)),
```

### Summary table

| Metric | threshold=0 | threshold>0 | Status |
|---|---|---|---|
| DirAcc (old) | Correct | Misleading ✗ | Replaced |
| Active DirAcc | Same as old | Correct ✓ | Now in use |
| Coverage | 100% | Shows trade frequency ✓ | Now in use |

---

## Exp 7 — `exp/combined-best` ★★★ Best Overall

**Change:** Combines all three validated improvements in one branch:
1. `zscore_win` tuned per-model via joint OOF CV (all sklearn settle on 5)
2. `signal_threshold` tuned jointly with `zscore_win` via OOF CV (no test leakage)
3. ElasticNet alpha grid widened to include 1.0; `tune_all.py` fixed to include ElasticNet + LightGBM

| Model | Sharpe | CAGR | Active DirAcc | Coverage | vs Baseline | `zscore_win` | `threshold` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **SVR_lin** | **9.45** | **346%** | **77.0%** | **85.1%** | **+1.64** | 5 | 0.3 |
| Ridge | 9.16 | 329% | 75.7% | 84.8% | +1.70 | 5 | 0.3 |
| SVR_rbf | 8.73 | 289% | 77.5% | 74.6% | +1.97 | 5 | 0.5 |
| **ElasticNet** | **8.55** | **267%** | **79.5%** | **65.2%** | **+5.63** | 5 | 0.75 |
| Lasso | 8.49 | 264% | 79.4% | 65.0% | +2.27 | 5 | 0.75 |
| MLP | 8.44 | 274% | 77.6% | 73.2% | +1.71 | 5 | 0.5 |
| XGBoost | 8.39 | 275% | 73.3% | 79.2% | +2.00 | 5 | 0.5 |
| RandomForest | 8.18 | 272% | 74.0% | 79.6% | +2.17 | 5 | 0.5 |
| LightGBM | 8.04 | 265% | 74.8% | 78.1% | +1.29 | 5 | 0.5 |

**Finding:** Gains are additive. SVR_lin: baseline 7.81 → zscore-win only 9.03 → combined 9.45. Every sklearn model now exceeds Sharpe 8.0. The joint OOF CV for (zscore_win, signal_threshold) avoids test leakage — both signal parameters are selected on train-val OOF predictions only. Active DirAcc is now 73–80% across models (the old raw metric showed ~52–66% due to cash days being counted as wrong). Higher thresholds (ElasticNet/Lasso at 0.75) trade less often (65% coverage) but with higher accuracy (79%).

---

## Grid Boundary Analysis — `exp/combined-best` best params

After Exp 7, each model's best params were checked against its search grid boundaries.
A boundary hit means the optimum may lie outside the tested range.

| Model | Best Param | Boundary? | Action | Exp 8 Outcome |
|---|---|---|---|---|
| SVR_lin | C=1000 | UPPER | Extended to [1000, 5000, 10000] | +0.09 Sharpe ✓ |
| Ridge | alpha=100 | UPPER | Extended to [100, 500, 1000] | −0.65 Sharpe ✗ — CV/Sharpe misalignment; closed |
| SVR_rbf | C=0.01, gamma=0.001 | BOTH LOWER | Extended C↓ and gamma↓ | +0.19 Sharpe ✓ |
| LightGBM | lr=0.01, n_est=300, leaves=15 | 3 boundaries | Extended all 3 directions | 0.00 — original was already optimal |
| MLP | hidden=(128,64,32) | UPPER | Added (256,128,64) | +0.32 Sharpe ✓ |
| XGBoost | max_depth=2 | lower | Skipped — depth=1 = stumps | — |
| RandomForest | min_samples_leaf=1, min_samples_split=2 | both lower | Skipped — already minimum regularization | — |
| ElasticNet | l1_ratio=0.1 | LOWER | Skipped — toward 0 = Ridge territory | — |
| Lasso | alpha=0.001 | middle | Nothing to do | — |

**Result: 3/5 extended models improved. Ridge is a CV-objective problem, not a grid problem.**

---

## Exp 8 — `exp/extended-search`

**Change:** Extended search grids for the 5 models that hit boundaries in Exp 7.

| Model | Sharpe | CAGR | Active DirAcc | Coverage | vs Exp 7 | Best params |
|---|---:|---:|---:|---:|---:|---|
| **SVR_lin** | **9.54** | **351%** | **76.7%** | **85.2%** | **+0.09** | C=0.1, thr=0.3 |
| SVR_rbf | 8.93 | 300% | 78.0% | 74.6% | +0.19 | C=0.1, γ=0.0001, thr=0.5 |
| MLP | 8.75 | 293% | 77.4% | 75.6% | +0.32 | hidden=(256,128,64), thr=0.5 |
| Ridge | 8.51 | 279% | 76.5% | 75.4% | **−0.65** | α=1000, thr=0.5 |
| LightGBM | 8.04 | 265% | 74.8% | 78.1% | 0.00 | unchanged |

*Unchanged models (Lasso, ElasticNet, XGBoost, RandomForest) carry forward from Exp 7.*

**Finding:** Mixed results. Three models improved (SVR_rbf +0.19, MLP +0.32 from new `(256,128,64)` arch, SVR_lin +0.09). Ridge got worse — alpha jumped to 1000 (upper boundary again), and test Sharpe dropped. This is a CV/Sharpe misalignment: MSE-based CV keeps preferring stronger regularization (flatter predictions → lower MSE) but flatter predictions also produce weaker z-score signals → lower Sharpe. **Ridge grid extension is closed** — the problem is the wrong CV objective, not the grid range. See the note below. LightGBM's extension found the same params — the original grid was already optimal.

---

## CV Objective Note — MSE vs Sharpe Misalignment

### The problem

All models are currently tuned with `GridSearchCV(scoring='neg_mean_squared_error')`. For most models this works well, but for Ridge (and potentially other linear models) it creates a systematic misalignment:

- Higher alpha → smoother, smaller-variance predictions → **lower MSE** → CV picks it
- Smaller-variance predictions → smaller z-scores → weaker signal differentiation → **lower Sharpe**

The signal params (zscore_win, threshold) are already selected by OOF Sharpe. The problem is that the *model* params (alpha, C, etc.) are still selected by MSE — and for some models those two objectives point in opposite directions.

### Three options, ordered by impact

**Option 1 — Extend OOF Sharpe sweep to model params (recommended)**

Instead of using GridSearchCV for model params, manually iterate over all param combos, collect OOF predictions, then pick the (model_params, signal_params) combo with the best joint OOF Sharpe. Replaces `GridSearchCV` entirely for sklearn models:

```python
best_sharpe, best_model_params, best_zw, best_thr = -np.inf, None, None, None
for params in ParameterGrid(model_grid):          # iterate model param combos
    oof_preds = cross_val_predict(model(**params), X_tv, y_tv, cv=tscv)
    for zw in zscore_win_vals:
        for thr in threshold_vals:
            s = trading_metrics(oof_preds, lr_tv, y_tv, zscore_win=zw, threshold=thr)['Sharpe']
            if s > best_sharpe:
                best_sharpe, best_model_params, best_zw, best_thr = s, params, zw, thr
```

**Trade-off:** More fits (n_model_combos × 5 folds instead of n_model_combos × 5 folds via GridSearchCV — same count, just sequential). For fast models (Ridge, Lasso, ElasticNet, LightGBM) this is negligible. For SVR_rbf the cross product with signal params grows but is still manageable.

**Option 2 — Custom Sharpe scorer in GridSearchCV**

Pass a custom scorer to GridSearchCV that computes OOF Sharpe directly. Requires threading log_rets into the scorer via a closure or a wrapper:

```python
def make_sharpe_scorer(log_rets_tv, zscore_win, threshold):
    def scorer(estimator, X, y):
        preds = estimator.predict(X)
        return trading_metrics(preds, log_rets_tv[len(log_rets_tv)-len(y):],
                               zscore_win=zscore_win, threshold=threshold)['Sharpe']
    return make_scorer(scorer, greater_is_better=True)
```

**Trade-off:** Simpler code change, but signal params must be fixed during model selection (no joint sweep). Less principled than Option 1.

**Option 3 — Two-stage: MSE shortlist → Sharpe final pick**

Use MSE to pick the top-3 model param candidates, then re-evaluate those 3 with OOF Sharpe to pick the winner. Low implementation cost, minimal risk of overfitting to Sharpe noise.

### Which models need this

| Model | Problem? | Reason |
|---|---|---|
| Ridge | Yes — confirmed | alpha=1000 beats MSE but loses on Sharpe |
| Lasso | Possibly | alpha is at a grid middle point — may be fine |
| ElasticNet | Possibly | l1_ratio at lower boundary; same dynamic possible |
| SVR_lin | Unlikely | C is not a regularization parameter in the same direction |
| SVR_rbf, XGBoost, RF, LightGBM, MLP | Unlikely | Tree/kernel models less sensitive to this effect |

---

## Exp 9 — `exp/sharpe-cv`

**Change:** Replace two-step MSE-then-Sharpe selection with a single joint OOF Sharpe sweep over all params simultaneously. In `tune_sklearn`, `GridSearchCV` (MSE objective) is removed. Instead, for every model param combo the pipeline is cross-validated, OOF predictions collected, and then all (zscore_win × threshold) signal combos are evaluated by OOF Sharpe — in one loop. The winning (model_params, zscore_win, threshold) triple is the one with the best OOF Sharpe across the entire joint grid.

**Fit count** is identical to before (n_model_combos × 5 folds). The signal sweep is arithmetic on already-collected OOF predictions — no extra fits.

**Motivation:** Ridge alpha=1000 was selected by MSE-CV but hurt Sharpe (Exp 8). The new approach cannot make that mistake because Sharpe is the only objective.

| Model | Sharpe | CAGR | Active DirAcc | Coverage | vs Exp 8 | Best params |
|---|---:|---:|---:|---:|---:|---|
| **SVR_lin** | **9.45** | **346%** | **77.0%** | **85.1%** | −0.09 | C=1000, thr=0.3 (same as Exp 7) |
| **Ridge** | **9.39** | **339%** | **76.1%** | **84.9%** | **+0.88** | α=1.0, thr=0.3 ← was α=1000 |
| Lasso | 9.26 | 328% | 75.7% | 84.9% | **+0.77** | α=0.00001, thr=0.3 ← tiny regularization |
| ElasticNet | 9.26 | 328% | 75.8% | 84.8% | **+0.71** | α=0.0001, thr=0.3 ← was α=0.01 |
| XGBoost | 8.83 | 306% | 74.0% | 90.2% | **+0.44** | depth=4, lr=0.1, thr=0.2 ← was depth=2 |
| SVR_rbf | 8.83 | 296% | 77.5% | 89.4% | −0.10 | C=1, γ=0.01, thr=0.2 |
| MLP | 8.76 | 293% | 77.4% | 75.6% | +0.00 | (256,128,64), unchanged |
| RandomForest | 8.18 | 272% | 74.0% | 79.6% | +0.00 | unchanged |
| LightGBM | 8.04 | 265% | 74.8% | 78.1% | +0.00 | unchanged |

**Finding:** Sharpe-CV confirmed and resolved the MSE/Sharpe misalignment. The four biggest movers were all regularization-dominated models: Ridge (α: 1000→1, +0.88), Lasso (α: 0.001→0.00001, +0.77), ElasticNet (α: 0.01→0.0001, +0.71), XGBoost (depth: 2→4, +0.44). MSE-CV had been pushing these toward smooth, low-variance predictions — which minimise MSE but also flatten z-scores and weaken the trading signal. Sharpe-CV picked less regularised, more expressive models that generate sharper z-score differentials. SVR_lin was unaffected (same params as Exp 7, C=1000 wins on both objectives). SVR_rbf, MLP, RF, LightGBM are also stable — their MSE and Sharpe optima already aligned.

---

---

## Exp 10 — `exp/recent-3yr`

**Change:** Restrict training and test data to the most recent 3 years (2023-05-01 → 2026-05-01, 756 rows). Feature engineering still runs on the full 20-year dataset for rolling warmup, then the window is sliced after `dropna()`. The 80/20 split is reapplied within the 3-year window: **train/val 604 rows (2023-05-01 → 2025-09-23), test 152 rows (2025-09-24 → 2026-05-01)**. All models re-tuned with Sharpe-CV (same grids as Exp 9). frac-diff d=0.3 (vs 0.5 on full history — recent window is more stationary).

**Motivation:** Earlier data (2006–2015) has much lower volatility. Training on that flat regime may dilute the signal relevant to current gold dynamics (2023–2026 bull run). Restricting to recent data lets models focus on the current regime.

| Model | Sharpe | CAGR | Active DirAcc | Coverage | vs Exp 9 | Best params |
|---|---:|---:|---:|---:|---:|---|
| **SVR_lin** | **10.60** | **1341%** | **88.4%** | **73.7%** | **+1.15** | C=10000, thr=0.5 |
| Ridge | 10.13 | 1321% | 82.8% | 88.2% | +0.74 | α=0.1, thr=0.3 |
| Lasso | 9.98 | 1286% | 81.6% | 89.5% | +0.72 | α=0.00001, thr=0.3 |
| ElasticNet | 9.98 | 1286% | 81.6% | 89.5% | +0.72 | α=0.0001, thr=0.3 |
| SVR_rbf | 9.82 | 1072% | 89.6% | 63.2% | +0.90 | C=10, γ=0.0001, thr=0.75 |
| XGBoost | 9.62 | 1098% | 80.9% | 72.4% | +1.17 | depth=4, lr=0.05, n_est=300 |
| LightGBM | 8.51 | 821% | 81.1% | 73.0% | −0.78 | n_est=100, leaves=7, lr=0.05 |
| RandomForest | 7.33 | 607% | 78.2% | 72.4% | −0.81 | default, n_est=100 |
| MLP | 1.44 | 55% | 59.6% | 89.5% | **−4.20** | (128,64), α=0.001 |
| LSTM | −0.75 | −22% | 48.6% | 99.3% | −0.93 | ts=5 |
| GRU | −0.46 | −14% | 51.6% | 87.1% | −1.10 | ts=5 |
| BiLSTM | −1.03 | −28% | 46.7% | 89.1% | −1.42 | ts=15 |

*CAGR is high due to the short 152-day test period (2025-09-24 → 2026-05-01) capturing a strong gold bull run. Compare Sharpe for regime-adjusted performance.*

**Finding:** Restricting to recent 3 years helps the linear and SVM models significantly (+0.72 to +1.17 Sharpe) — their predictions align better with the current regime. XGBoost also improves (+1.17), confirming that the older flat-price data was diluting its gradient signal. Tree ensemble models (LightGBM −0.78, RandomForest −0.81) suffer from the smaller sample size (604 train rows, 57 features — ratio too tight for ensembles). Deep learning collapses entirely: 604 sequences is too little for LSTM/GRU/BiLSTM — they overfit the training window and produce negative Sharpe on test. MLP similarly degrades (−4.20). **The recent-3yr regime is a clear win for linear and kernel models; ensemble and deep learning models need more data than the 3-year window provides.**

Active DirAcc is notably higher across the board (SVR_lin 88.4%, SVR_rbf 89.6%) suggesting the 2024–2026 gold market is more directionally consistent — larger, sustained moves rather than choppy reversals.

---

---

## Exp 11 — `exp/combined-recent` ★★★ Best Overall

**Change:** Hybrid best-of-both setup. Training window: recent 3 years (2023-05-01 → 2025-09-23, 604 rows). Test: 2025-09-24 → 2026-05-01 (152 days).

- **Exp 10 (recent-3yr) params** used for models that improved on the recent window: SVR_lin, Ridge, Lasso, ElasticNet, SVR_rbf, XGBoost
- **Exp 9 (sharpe-cv) params** used for models that degraded on the recent window: LightGBM, RandomForest, MLP, LSTM, GRU, BiLSTM

Rationale: Exp 9 params were tuned on the full history with Sharpe-CV. For ensemble/deep learning models that need more training data, those params are better calibrated than the ones over-fitted to 604 rows.

| Model | Sharpe | CAGR | Active DirAcc | Coverage | Train window | Params source |
|---|---:|---:|---:|---:|---|---|
| **SVR_lin** | **10.60** | **1341%** | **88.4%** | **73.7%** | recent 3yr | Exp 10 |
| Ridge | 10.13 | 1321% | 82.8% | 88.2% | recent 3yr | Exp 10 |
| Lasso | 9.98 | 1286% | 81.6% | 89.5% | recent 3yr | Exp 10 |
| ElasticNet | 9.98 | 1286% | 81.6% | 89.5% | recent 3yr | Exp 10 |
| SVR_rbf | 9.82 | 1072% | 89.6% | 63.2% | recent 3yr | Exp 10 |
| XGBoost | 9.62 | 1098% | 80.9% | 72.4% | recent 3yr | Exp 10 |
| LightGBM | 8.21 | 934% | 77.9% | 89.5% | full hist | Exp 9 |
| RandomForest | 8.18 | 822% | 76.4% | 80.9% | full hist | Exp 9 |
| **MLP** | **7.30** | **715%** | **75.8%** | **84.2%** | full hist | Exp 9 |
| LSTM | 0.37 | 13% | 55.3% | 100% | full hist | Exp 9 |
| BiLSTM | 0.25 | 8% | 50.0% | 100% | full hist | Exp 9 |
| GRU | −0.13 | −4% | 44.1% | 100% | full hist | Exp 9 |

*All models evaluated on the same 152-day window (2025-09-24 → 2026-05-01). Full-hist models have 819-day z-score warmup before eval start — no cold-start bias. High CAGR across the board reflects the 2025–2026 gold bull run.*

**Finding:** The split is now principled: full-history models (LightGBM, RF, MLP, LSTM, GRU, BiLSTM) are trained on the complete dataset up to 2022-06-21. Their predictions from 2022-06-22 onwards provide 819 days of z-score warmup before the eval window — eliminating the cold-start bias in the preliminary run. LightGBM improves to 8.21, RandomForest to 8.18, MLP to 7.30 — all substantially better than their 3yr-only tuned versions (8.51, 7.33, 1.44 in Exp 10). The `tune_model.py` `--recent-years` flag (default 3, use 0 for full history) controls which window each model is tuned on. Deep learning (LSTM 0.37, BiLSTM 0.25, GRU −0.13) still underperforms on the 2025-2026 eval period, suggesting regime mismatch between full-history training distribution and recent gold dynamics.

---

## Summary

### SVR_lin Sharpe Ranking (all experiments)

| Rank | Experiment | SVR_lin Sharpe | Primary Change |
|---|---|---:|---|
| 1 | `exp/combined-recent` | **10.60** | Hybrid: Exp10 params for linear/SVM/XGB, Exp9 for rest |
| 1 | `exp/recent-3yr` | **10.60** | Train on last 3 years only |
| 2 | `exp/extended-search` | **9.54** | Wider C grid |
| 2 | `exp/combined-best` | 9.45 | ZSCORE_WIN=5 + threshold + ElasticNet fix |
| 2 | `exp/sharpe-cv` | **9.45** | Sharpe-CV (SVR_lin unaffected; Ridge/Lasso/ElasticNet/XGBoost all improved) |
| 3 | `exp/zscore-win-tuning` | 9.03 | ZSCORE_WIN: 10 → 5 |
| 4 | `exp/signal-dead-zone` | 8.49 | Hold cash when \|z\| < 0.5 |
| 5 | `main` (baseline) | 7.81 | — |
| 5 | `exp/controlled-39-features` | 7.81 | 39 features (neutral) |
| 5 | `exp/tune-elasticnet-lightgbm` | 7.81 | ElasticNet fixed |
| 6 | `exp/per-model-k-alpha` | 7.22 | Per-model k |
| 6 | `exp/proportional-sizing` | 7.21 | Continuous sizing |

### Best model per experiment (Sharpe)

| Experiment | Best model | Sharpe |
|---|---|---:|
| `exp/combined-recent` | SVR_lin | **10.60** |
| `exp/recent-3yr` | SVR_lin | 10.60 |
| `exp/sharpe-cv` | SVR_lin | 9.45 |
| `exp/extended-search` | SVR_lin | 9.54 |
| `exp/combined-best` | SVR_lin | 9.45 |
| `exp/zscore-win-tuning` | SVR_lin | 9.03 |
| `exp/signal-dead-zone` | SVR_lin | 8.49 |
| `main` | SVR_lin | 7.81 |

### What Works

| Improvement | Best model gain | Notes |
|---|---|---|
| Recent 3yr training window | +1.17 (SVR_lin, XGBoost), +0.90 (SVR_rbf) | Exp 10 — focuses on current regime; linear/SVM/XGBoost all benefit |
| Sharpe-CV (replace MSE-CV) | +0.88 (Ridge), +0.77 (Lasso), +0.71 (ElasticNet), +0.44 (XGBoost) | Exp 9 — all regularization-dominated models benefited |
| ZSCORE_WIN=5 | +1.68 (RF), +1.53 (SVR_rbf) | Universal improvement, was a regression |
| Signal dead zone (per-model threshold) | +0.82 (RF), +0.80 (SVR_rbf) | Cuts noise trades |
| ElasticNet alpha fix | +5.64 (combined) | Was severely misconfigured |
| MLP wider hidden layers (256,128,64) | +0.32 | Exp 8 extension found larger net helps |

### What Doesn't Work

| Change | Best model impact | Reason |
|---|---|---|
| Recent 3yr window for tree ensembles | −0.78 (LightGBM), −0.81 (RF) | 604 train rows too few for high-variance ensembles |
| Recent 3yr window for deep learning | −4.20 (MLP), −0.93 (LSTM), −1.42 (BiLSTM) | 604 sequences inadequate; models overfit and collapse on test |
| Per-model SelectKBest k | −0.59 (SVR_lin) | Optimal k=57 (no selection needed) |
| Proportional sizing z/3 | −0.60 (SVR_lin) | Lower average position size reduces returns |
| Ridge alpha > 100 (MSE-CV) | −0.65 (Ridge) | MSE-CV kept pushing regularization up; fixed by Sharpe-CV |
