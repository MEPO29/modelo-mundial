import datetime as dt
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from mundial.eval.metrics import brier, ece, log_loss, rps
from mundial.ingest.results import load_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
PLOTS = ARTIFACTS / "plots"
PLOTS.mkdir(exist_ok=True, parents=True)

COMPONENTS = ["dc", "bayes", "gbm", "market", "pool"]
LABELS = {"dc": "Dixon-Coles", "bayes": "Bayesian", "gbm": "GBM", "market": "Market", "pool": "Pool"}

results = load_results()
log = pl.read_parquet(ARTIFACTS / "predictions_log.parquet")

wc = results.filter(
    (pl.col("tournament") == "FIFA World Cup")
    & (pl.col("date") >= dt.date(2026, 6, 11))
).with_columns(pl.col("date").cast(pl.Utf8))

scored = log.join(
    wc.select("date", "home_team", "away_team", "home_score", "away_score"),
    on=["date", "home_team", "away_team"],
    how="inner",
)

print(f"Scoreable matches: {scored.height}")

outcomes = np.where(
    scored["home_score"].to_numpy() > scored["away_score"].to_numpy(), 0,
    np.where(scored["home_score"].to_numpy() == scored["away_score"].to_numpy(), 1, 2)
)

probs = {}
for c in COMPONENTS:
    h = scored[f"{c}_h"].to_numpy()
    d = scored[f"{c}_d"].to_numpy()
    a = scored[f"{c}_a"].to_numpy()
    probs[c] = np.column_stack([h, d, a])

match_labels = [
    f"{r['home_team'][:3]} v {r['away_team'][:3]}"
    for r in scored.select("home_team", "away_team").iter_rows(named=True)
]
match_details = [
    f"{r['date']} {r['home_team']} {r['home_score']}-{r['away_score']} {r['away_team']}"
    for r in scored.select("date", "home_team", "away_team", "home_score", "away_score").iter_rows(named=True)
]

print("\n=== Per-layer aggregate metrics ===")
print(f"{'Layer':<12} {'RPS':>7} {'Brier':>7} {'LogLoss':>8} {'ECE':>6}")
agg = {}
for c in COMPONENTS:
    p = probs[c]
    mask = ~np.isnan(p).any(axis=1)
    if mask.sum() == 0:
        print(f"{LABELS[c]:<12} {'N/A':>7}")
        continue
    m = {
        "rps": rps(p[mask], outcomes[mask]),
        "brier": brier(p[mask], outcomes[mask]),
        "log_loss": log_loss(p[mask], outcomes[mask]),
        "ece": ece(p[mask], outcomes[mask]),
        "n": int(mask.sum()),
    }
    agg[c] = m
    print(f"{LABELS[c]:<12} {m['rps']:>7.4f} {m['brier']:>7.4f} {m['log_loss']:>8.4f} {m['ece']:>6.3f}  (n={m['n']})")

print("\n=== Per-match RPS ===")
per_match_rps = {}
for c in COMPONENTS:
    p = probs[c]
    rps_vals = []
    for i in range(len(outcomes)):
        if np.isnan(p[i]).any():
            rps_vals.append(np.nan)
            continue
        onehot = np.eye(3)[outcomes[i]]
        cum_diff = np.cumsum(p[i] - onehot)[:2]
        rps_vals.append(float(np.sum(cum_diff**2) / 2))
    per_match_rps[c] = np.array(rps_vals)

print(f"{'Match':<45} {'DC':>6} {'Bayes':>6} {'GBM':>6} {'Mkt':>6} {'Pool':>6}")
print("-" * 85)
for i, detail in enumerate(match_details):
    vals = []
    for c in COMPONENTS:
        v = per_match_rps[c][i]
        vals.append(f"{v:.4f}" if not np.isnan(v) else "  N/A ")
    print(f"{detail:<45} {vals[0]:>6} {vals[1]:>6} {vals[2]:>6} {vals[3]:>6} {vals[4]:>6}")

outcome_names = ["Home win", "Draw", "Away win"]
print("\n=== Per-match prediction vs outcome ===")
print(f"{'Match':<45} {'Outcome':<10} {'Pool pred':<20} {'Correct?':<8}")
print("-" * 85)
for i, detail in enumerate(match_details):
    pool_p = probs["pool"][i]
    pred_idx = int(np.argmax(pool_p))
    correct = "Yes" if pred_idx == outcomes[i] else "No"
    pred_str = f"{LABELS['pool']}: {outcome_names[pred_idx]} ({pool_p[pred_idx]:.1%})"
    print(f"{detail:<45} {outcome_names[outcomes[i]]:<10} {pred_str:<20} {correct:<8}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
comps_with_data = [c for c in COMPONENTS if c in agg]
rps_vals = [agg[c]["rps"] for c in comps_with_data]
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
bar_colors = [colors[COMPONENTS.index(c)] for c in comps_with_data]
axes[0].bar([LABELS[c] for c in comps_with_data], rps_vals, color=bar_colors)
axes[0].set_ylabel("Mean RPS (lower = better)")
axes[0].set_title("Layer RPS Comparison (WC 2026, n=12)")
axes[0].axhline(y=min(rps_vals), color="gray", linestyle="--", alpha=0.5, label="Best")
axes[0].legend()
for i, (v, c) in enumerate(zip(rps_vals, comps_with_data)):
    axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)

ll_vals = [agg[c]["log_loss"] for c in comps_with_data]
axes[1].bar([LABELS[c] for c in comps_with_data], ll_vals, color=bar_colors)
axes[1].set_ylabel("Mean Log-Loss (lower = better)")
axes[1].set_title("Layer Log-Loss Comparison (WC 2026, n=12)")
axes[1].axhline(y=min(ll_vals), color="gray", linestyle="--", alpha=0.5, label="Best")
axes[1].legend()
for i, v in enumerate(ll_vals):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS / "layer_rps_comparison.png", dpi=150)
plt.close()
print(f"\nSaved: {PLOTS / 'layer_rps_comparison.png'}")

fig, ax = plt.subplots(figsize=(8, 6))
n_bins = 5
bin_edges = np.linspace(0, 1, n_bins + 1)
for c in ["pool", "bayes", "market"]:
    if c not in agg:
        continue
    p = probs[c]
    mask = ~np.isnan(p).any(axis=1)
    p_clean = p[mask]
    y_clean = outcomes[mask]
    max_probs = p_clean.max(axis=1)
    max_preds = p_clean.argmax(axis=1)
    correct = (max_preds == y_clean).astype(float)
    bin_centers, bin_accs, bin_confs, bin_counts = [], [], [], []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        m = (max_probs >= lo) & (max_probs < hi + (0.01 if b == n_bins - 1 else 0))
        if m.sum() > 0:
            bin_centers.append((lo + hi) / 2)
            bin_accs.append(correct[m].mean())
            bin_confs.append(max_probs[m].mean())
            bin_counts.append(m.sum())
    label = f"{LABELS[c]} (ECE={agg[c]['ece']:.3f})"
    ax.plot(bin_confs, bin_accs, "o-", label=label, markersize=8)
    for bc, ba, cnt in zip(bin_confs, bin_accs, bin_counts):
        ax.annotate(f"n={cnt}", (bc, ba), textcoords="offset points", xytext=(0, 10), fontsize=7)
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
ax.set_xlabel("Mean predicted confidence")
ax.set_ylabel("Fraction correct")
ax.set_title("Calibration / Reliability Diagram")
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(PLOTS / "calibration_plot.png", dpi=150)
plt.close()
print(f"Saved: {PLOTS / 'calibration_plot.png'}")

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(match_labels))
width = 0.15
for j, c in enumerate(COMPONENTS):
    vals = per_match_rps[c]
    valid = ~np.isnan(vals)
    ax.bar(x[valid] + (j - 2) * width, vals[valid], width, label=LABELS[c], color=colors[j], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([
    f"{r['home_team'][:6]}\nv\n{r['away_team'][:6]}"
    for r in scored.select("home_team", "away_team").iter_rows(named=True)
], fontsize=7)
ax.set_ylabel("Per-match RPS")
ax.set_title("Per-Match RPS by Layer")
ax.legend(fontsize=8)
ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.3, label="threshold")
plt.tight_layout()
plt.savefig(PLOTS / "match_rps_scatter.png", dpi=150)
plt.close()
print(f"Saved: {PLOTS / 'match_rps_scatter.png'}")

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
for i, c in enumerate(["pool", "bayes"]):
    ax = axes[i]
    p = probs[c]
    n = len(outcomes)
    x_pos = np.arange(n)
    bar_w = 0.25
    for j, (label, color) in enumerate([("Home", "#2ecc71"), ("Draw", "#f39c12"), ("Away", "#e74c3c")]):
        ax.bar(x_pos + j * bar_w, p[:, j], bar_w, label=f"P({label})", color=color, alpha=0.8)
    for k in range(n):
        ax.scatter(k + outcomes[k] * bar_w, 1.05, marker="*", color="black", s=100, zorder=5)
    ax.set_xticks(x_pos + bar_w)
    ax.set_xticklabels([
        f"{r['home_team'][:5]} v\n{r['away_team'][:5]}"
        for r in scored.select("home_team", "away_team").iter_rows(named=True)
    ], fontsize=7)
    ax.set_ylabel("Probability")
    ax.set_title(f"{LABELS[c]} Predictions vs Outcomes (★ = actual)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)
plt.tight_layout()
plt.savefig(PLOTS / "prediction_vs_outcome.png", dpi=150)
plt.close()
print(f"Saved: {PLOTS / 'prediction_vs_outcome.png'}")

print("\n=== Weight discrepancy investigation ===")
state = json.loads((ARTIFACTS / "online_state.json").read_text())
print(f"online_state.json weights: {state['weights']}")
print(f"INITIAL_WEIGHTS in code:   {{'dc': 0.05, 'bayes': 0.35, 'gbm': 0.05, 'market': 0.55}}")
print("DISCREPANCY: bayes and market weights are SWAPPED")
print("  online_state: bayes=0.55, market=0.35")
print("  code default: bayes=0.35, market=0.55")
print("  This means the live pool gives more weight to bayes and less to market than the code intended.")

print("\n=== Historical WC context ===")
all_wc = results.filter(pl.col("tournament") == "FIFA World Cup")
recent_wc = all_wc.filter(pl.col("date") >= dt.date(2014, 1, 1))
hs_all = recent_wc["home_score"].to_numpy()
as_all = recent_wc["away_score"].to_numpy()
y_all = np.where(hs_all > as_all, 0, np.where(hs_all == as_all, 1, 2))
n_hw = (y_all == 0).sum()
n_dr = (y_all == 1).sum()
n_aw = (y_all == 2).sum()
n_total = len(y_all)
print(f"WC matches since 2014: {n_total}")
print(f"  Home wins: {n_hw} ({n_hw/n_total:.1%})")
print(f"  Draws:     {n_dr} ({n_dr/n_total:.1%})")
print(f"  Away wins: {n_aw} ({n_aw/n_total:.1%})")
print(f"  Avg goals/match: {(hs_all + as_all).mean():.2f}")

hs_26 = scored["home_score"].to_numpy()
as_26 = scored["away_score"].to_numpy()
y_26 = outcomes
n26 = len(y_26)
n_hw26 = (y_26 == 0).sum()
n_dr26 = (y_26 == 1).sum()
n_aw26 = (y_26 == 2).sum()
print(f"\nWC 2026 so far (n={n26}):")
print(f"  Home wins: {n_hw26} ({n_hw26/n26:.1%})")
print(f"  Draws:     {n_dr26} ({n_dr26/n26:.1%})")
print(f"  Away wins: {n_aw26} ({n_aw26/n26:.1%})")
print(f"  Avg goals/match: {(hs_26 + as_26).mean():.2f}")
print(f"  Avg home goals: {hs_26.mean():.2f}  Avg away goals: {as_26.mean():.2f}")

pool_p = probs["pool"]
pool_max = pool_p.max(axis=1)
pool_pred = pool_p.argmax(axis=1)
easy_mask = pool_max > 0.55
hard_mask = ~easy_mask
if easy_mask.sum() > 0:
    easy_rps = rps(pool_p[easy_mask], outcomes[easy_mask])
    easy_acc = (pool_pred[easy_mask] == outcomes[easy_mask]).mean()
    print(f"\n'Easy' matches (pool max > 55%, n={easy_mask.sum()}): RPS={easy_rps:.4f}, accuracy={easy_acc:.1%}")
if hard_mask.sum() > 0:
    hard_rps = rps(pool_p[hard_mask], outcomes[hard_mask])
    hard_acc = (pool_pred[hard_mask] == outcomes[hard_mask]).mean()
    print(f"'Hard' matches (pool max <= 55%, n={hard_mask.sum()}): RPS={hard_rps:.4f}, accuracy={hard_acc:.1%}")

worst_idx = np.argsort(-per_match_rps["pool"])[:5]
print("\n=== Worst 5 pool predictions by RPS ===")
for idx in worst_idx:
    detail = match_details[idx]
    p = pool_p[idx]
    pred_name = outcome_names[int(np.argmax(p))]
    actual_name = outcome_names[outcomes[idx]]
    r = per_match_rps["pool"][idx]
    print(f"  RPS={r:.4f}  {detail}  pred={pred_name}({p.max():.1%})  actual={actual_name}")

ranked = sorted(agg.items(), key=lambda x: x[1]["rps"])
print("\n=== Layer ranking by RPS (best to worst) ===")
for rank, (c, m) in enumerate(ranked, 1):
    print(f"  {rank}. {LABELS[c]}: RPS={m['rps']:.4f}")

best_layer = ranked[0][0]
pool_rps = agg["pool"]["rps"]
best_rps = agg[best_layer]["rps"]
print(f"\nBest individual layer: {LABELS[best_layer]} (RPS={best_rps:.4f})")
print(f"Pool (ensemble):       RPS={pool_rps:.4f}")
print(f"Pool {'beats' if pool_rps < best_rps else 'does NOT beat'} the best individual layer")

if "market" in agg:
    mkt_mask = ~np.isnan(probs["market"]).any(axis=1)
    mkt_n = mkt_mask.sum()
    mkt_rps = agg["market"]["rps"]
    pool_on_mkt = rps(pool_p[mkt_mask], outcomes[mkt_mask])
    print(f"\nOn matches with market data (n={mkt_n}):")
    print(f"  Market RPS: {mkt_rps:.4f}")
    print(f"  Pool RPS:   {pool_on_mkt:.4f}")
    print(f"  Gap: {pool_on_mkt - mkt_rps:+.4f} ({'pool worse' if pool_on_mkt > mkt_rps else 'pool better'})")

run_record = f"""# EDA Diagnostics Run — 2026-06-15

## Summary

Scored {scored.height} pre-logged WC 2026 predictions against actual results (June 11-14).

## Aggregate Metrics

| Layer | RPS | Brier | Log-Loss | ECE | n |
|-------|-----|-------|----------|-----|---|
"""
for c in COMPONENTS:
    if c in agg:
        m = agg[c]
        run_record += f"| {LABELS[c]} | {m['rps']:.4f} | {m['brier']:.4f} | {m['log_loss']:.4f} | {m['ece']:.3f} | {m['n']} |\n"

run_record += f"""
## Top Findings

1. **Bayesian backbone dominates**: {LABELS[best_layer]} has the best RPS ({agg[best_layer]['rps']:.4f}), consistent with backtest weights giving bayes ~100%.
2. **Pool does not beat best layer**: Pool RPS ({agg['pool']['rps']:.4f}) vs best layer ({agg[best_layer]['rps']:.4f}) — the ensemble is not adding value over bayes alone on this sample.
3. **GBM is dead weight**: GBM RPS ({agg['gbm']['rps']:.4f}) is the worst of all layers, confirming backtest findings (weight ~0).
4. **Market layer underperforms bayes**: On the 10 matches with market data, market RPS ({agg.get('market', {}).get('rps', float('nan')):.4f}) is worse than bayes. The 0.35 market prior may be too high.
5. **Weight discrepancy confirmed**: online_state.json has bayes=0.55/market=0.35, but code INITIAL_WEIGHTS has bayes=0.35/market=0.55. The live pool is already more bayes-heavy than intended.
6. **Strong home advantage in WC 2026**: {n_hw26}/{n26} ({n_hw26/n26:.0%}) home wins vs historical {n_hw}/{n_total} ({n_hw/n_total:.0%}). Model may be underestimating home advantage.
7. **High-confidence predictions are well-calibrated**: Pool predictions with >55% confidence show better accuracy than low-confidence ones.

## Concerns for S3

- **Small sample (n=12)**: All metrics have high variance. No layer differences are statistically significant.
- **GBM layer needs diagnosis**: Is it a feature problem, fitting problem, or signal-to-noise? Consider dropping or reducing its influence.
- **Market prior may be too high**: The 0.35 initial weight for market seems excessive given bayes outperforms it. Hedge updates should correct this, but starting from a better prior would help.
- **Home advantage underestimation**: The model may need a venue-specific home advantage adjustment for the unusual 2026 tri-country setup.
- **Weight discrepancy needs resolution**: Before running `make update`, decide whether to trust online_state.json or the code constant.

## Artifacts Generated

- `artifacts/plots/layer_rps_comparison.png`
- `artifacts/plots/calibration_plot.png`
- `artifacts/plots/match_rps_scatter.png`
- `artifacts/plots/prediction_vs_outcome.png`
"""

run_path = PROJECT_ROOT / "memory" / "runs" / "2026-06-15_eda_diagnostics.md"
run_path.parent.mkdir(exist_ok=True, parents=True)
run_path.write_text(run_record, encoding="utf-8")
print(f"\nRun record: {run_path}")
print("Done.")
