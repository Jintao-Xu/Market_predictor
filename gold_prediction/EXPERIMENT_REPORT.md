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

## Exp 9 — `exp/sharpe-cv` *(in progress)*

**Change:** Replace two-step MSE-then-Sharpe selection with a single joint OOF Sharpe sweep over all params simultaneously. In `tune_sklearn`, `GridSearchCV` (MSE objective) is removed. Instead, for every model param combo the pipeline is cross-validated, OOF predictions collected, and then all (zscore_win × threshold) signal combos are evaluated by OOF Sharpe — in one loop. The winning (model_params, zscore_win, threshold) triple is the one with the best OOF Sharpe across the entire joint grid.

**Fit count** is identical to before (n_model_combos × 5 folds). The signal sweep is arithmetic on already-collected OOF predictions — no extra fits.

**Motivation:** Ridge alpha=1000 was selected by MSE-CV but hurt Sharpe (Exp 8). The new approach cannot make that mistake because Sharpe is the only objective.

*Results to be filled in after tuning completes.*

---

## Summary

### SVR_lin Sharpe Ranking (all experiments)

| Rank | Experiment | SVR_lin Sharpe | Primary Change |
|---|---|---:|---|
| 1 | `exp/extended-search` | **9.54** | Wider C grid; best C shifted to 0.1 |
| 2 | `exp/combined-best` | 9.45 | ZSCORE_WIN=5 + threshold + ElasticNet fix |
| 3 | `exp/zscore-win-tuning` | 9.03 | ZSCORE_WIN: 10 → 5 |
| 4 | `exp/signal-dead-zone` | 8.49 | Hold cash when \|z\| < 0.5 |
| 5 | `main` (baseline) | 7.81 | — |
| 5 | `exp/controlled-39-features` | 7.81 | 39 features (neutral) |
| 5 | `exp/tune-elasticnet-lightgbm` | 7.81 | ElasticNet fixed |
| 6 | `exp/per-model-k-alpha` | 7.22 | Per-model k |
| 6 | `exp/proportional-sizing` | 7.21 | Continuous sizing |

### What Works

| Improvement | Best model gain | Notes |
|---|---|---|
| ZSCORE_WIN=5 | +1.68 (RF), +1.53 (SVR_rbf) | Universal improvement, was a regression |
| Signal dead zone (per-model threshold) | +0.82 (RF), +0.80 (SVR_rbf) | Cuts noise trades |
| ElasticNet alpha fix | +5.64 (combined) | Was severely misconfigured |
| MLP wider hidden layers (256,128,64) | +0.32 | Exp 8 extension found larger net helps |
| SVR_rbf lower gamma (0.0001) | +0.19 | Exp 8 found smoother kernel more effective |

### What Doesn't Work

| Change | Best model impact | Reason |
|---|---|---|
| Per-model SelectKBest k | −0.59 (SVR_lin) | Optimal k=57 (no selection needed) |
| Proportional sizing z/3 | −0.60 (SVR_lin) | Lower average position size reduces returns |
| Ridge alpha > 100 | −0.65 (Ridge) | Exp 8 pushed alpha to 1000; test Sharpe degraded — CV/Sharpe misalignment |
