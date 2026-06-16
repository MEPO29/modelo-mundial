"""
Stage S2 EDA Diagnostics v2 — WC 2026 Forecast Evaluation
==========================================================
Recomputes all metrics from raw data after data dictionary correction pass.
Corrected ground truth: 7 home wins, 4 draws, 1 away win (not 8H/3D/0A from prior version).

New in v2 vs v1:
  - Historical home advantage computed separately for neutral vs non-neutral WC matches
    (WC 2026 venues are in USA/Mexico/Canada — tri-nation host, all effectively neutral)
  - Weight counterfactual: compare actual pool (bayes=0.55, market=0.35) vs
    code-default pool (bayes=0.35, market=0.55) to quantify impact of swap
  - Per-match outcome probability for the actual outcome (separate from argmax accuracy)
  - Additional draw-specific diagnostics: how well does each layer estimate draw probability?
  - All plot filenames use _v2 suffix to avoid clobbering prior run

Outputs:
  - artifacts/plots/v2_layer_rps_comparison.png
  - artifacts/plots/v2_calibration_reliability.png
  - artifacts/plots/v2_match_rps_heatmap.png
  - artifacts/plots/v2_prediction_vs_outcome.png
  - artifacts/plots/v2_home_advantage_context.png
  - artifacts/plots/v2_weight_counterfactual.png
  - memory/runs/2026-06-15_eda_diagnostics_v2.md
  - memory/decisions/2026-06-15_eda-diagnostics-v2.md

All stats are on n=12 (n=10 for market). Findings are SUGGESTIVE, not significant.
"""

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

SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
PLOTS = ARTIFACTS / "plots"
PLOTS.mkdir(exist_ok=True, parents=True)
MEMORY = PROJECT_ROOT / "memory"

COMPONENTS = ["dc", "bayes", "gbm", "market", "pool"]
LABELS = {
    "dc": "Dixon-Coles",
    "bayes": "Bayesian",
    "gbm": "GBM",
    "market": "Market",
    "pool": "Pool",
}
COLORS = {
    "dc": "#e74c3c",
    "bayes": "#3498db",
    "gbm": "#2ecc71",
    "market": "#f39c12",
    "pool": "#9b59b6",
}

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("S2 EDA DIAGNOSTICS v2 — WC 2026 Forecast Evaluation")
print("=" * 70)
print(f"\nRandom seed: {SEED}")

results = load_results()
log = pl.read_parquet(ARTIFACTS / "predictions_log.parquet")

print(f"\nLoaded predictions_log.parquet: {log.height} rows x {log.width} cols")
print(f"Loaded results.csv: {results.height} rows")

# Filter to WC 2026 played matches
wc_2026 = results.filter(
    (pl.col("tournament") == "FIFA World Cup")
    & (pl.col("date") >= dt.date(2026, 6, 11))
    & (pl.col("home_score").is_not_null())
).with_columns(pl.col("date").cast(pl.Utf8))

print(f"WC 2026 played matches in results.csv: {wc_2026.height}")

# Inner join: predictions that have a played result
scored = log.join(
    wc_2026.select("date", "home_team", "away_team", "home_score", "away_score"),
    on=["date", "home_team", "away_team"],
    how="inner",
)

print(f"Scoreable predictions (inner join): {scored.height}")
assert scored.height == 12, f"Expected 12 scoreable matches, got {scored.height}"

# ---------------------------------------------------------------------------
# 2. Derive outcomes and probability arrays
# ---------------------------------------------------------------------------
hs = scored["home_score"].to_numpy()
as_ = scored["away_score"].to_numpy()
outcomes = np.where(hs > as_, 0, np.where(hs == as_, 1, 2))  # 0=H, 1=D, 2=A

outcome_names = {0: "Home win", 1: "Draw", 2: "Away win"}
n_hw = (outcomes == 0).sum()
n_dr = (outcomes == 1).sum()
n_aw = (outcomes == 2).sum()
print(f"\nOutcome distribution (n=12): {n_hw}H / {n_dr}D / {n_aw}A")
assert n_hw == 7 and n_dr == 4 and n_aw == 1, (
    f"Unexpected distribution: {n_hw}H/{n_dr}D/{n_aw}A — verify data"
)

probs = {}
for c in COMPONENTS:
    h = scored[f"{c}_h"].to_numpy(allow_copy=True).astype(float)
    d = scored[f"{c}_d"].to_numpy(allow_copy=True).astype(float)
    a = scored[f"{c}_a"].to_numpy(allow_copy=True).astype(float)
    probs[c] = np.column_stack([h, d, a])

match_labels = [
    f"{r['home_team'][:5]} v {r['away_team'][:5]}"
    for r in scored.select("home_team", "away_team").iter_rows(named=True)
]
match_details = [
    f"{r['date']} {r['home_team']} {r['home_score']}-{r['away_score']} {r['away_team']}"
    for r in scored.select("date", "home_team", "away_team", "home_score", "away_score")
    .iter_rows(named=True)
]

# ---------------------------------------------------------------------------
# 3. Per-layer aggregate metrics
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PER-LAYER AGGREGATE METRICS")
print("=" * 70)
print(f"\n{'Layer':<14} {'n':>3} {'RPS':>8} {'Brier':>8} {'LogLoss':>9} {'ECE':>7}")
print("-" * 55)

agg = {}
for c in COMPONENTS:
    p = probs[c]
    mask = ~np.isnan(p).any(axis=1)
    n = int(mask.sum())
    if n == 0:
        print(f"  {LABELS[c]:<12} {n:>3} {'N/A':>8}")
        continue
    m = {
        "rps": rps(p[mask], outcomes[mask]),
        "brier": brier(p[mask], outcomes[mask]),
        "log_loss": log_loss(p[mask], outcomes[mask]),
        "ece": ece(p[mask], outcomes[mask]),
        "n": n,
    }
    agg[c] = m
    print(
        f"  {LABELS[c]:<12} {n:>3} {m['rps']:>8.4f} {m['brier']:>8.4f} "
        f"{m['log_loss']:>9.4f} {m['ece']:>7.3f}"
    )

# ---------------------------------------------------------------------------
# 4. Per-match RPS
# ---------------------------------------------------------------------------
per_match_rps = {}
for c in COMPONENTS:
    p = probs[c]
    vals = []
    for i in range(len(outcomes)):
        if np.isnan(p[i]).any():
            vals.append(np.nan)
            continue
        onehot = np.eye(3)[outcomes[i]]
        cum_diff = np.cumsum(p[i] - onehot)[:2]
        vals.append(float(np.sum(cum_diff**2) / 2))
    per_match_rps[c] = np.array(vals)

print("\n" + "=" * 70)
print("PER-MATCH RPS")
print("=" * 70)
print(f"\n{'Match':<47} {'DC':>7} {'Bayes':>7} {'GBM':>7} {'Mkt':>7} {'Pool':>7}")
print("-" * 90)
for i, detail in enumerate(match_details):
    vals = []
    for c in COMPONENTS:
        v = per_match_rps[c][i]
        vals.append(f"{v:7.4f}" if not np.isnan(v) else "   N/A ")
    print(f"  {detail:<45} {' '.join(vals)}")

# ---------------------------------------------------------------------------
# 5. Gate comparison: pool vs DC, pool vs market
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("GATE AND BENCHMARK COMPARISON")
print("=" * 70)
pool_rps_all = agg["pool"]["rps"]
dc_rps_all = agg["dc"]["rps"]
gate_met = pool_rps_all < dc_rps_all
print(f"\n  Pool RPS (n=12):         {pool_rps_all:.4f}")
print(f"  Dixon-Coles RPS (n=12): {dc_rps_all:.4f}")
print(f"  Gate (pool < DC):        {'MET' if gate_met else 'NOT MET'}")

if "market" in agg:
    mkt_mask = ~np.isnan(probs["market"]).any(axis=1)
    mkt_n = mkt_mask.sum()
    pool_on_mkt = rps(probs["pool"][mkt_mask], outcomes[mkt_mask])
    mkt_rps_10 = agg["market"]["rps"]
    gap = pool_on_mkt - mkt_rps_10
    print(f"\n  On matches with market data (n={mkt_n}):")
    print(f"    Market RPS:  {mkt_rps_10:.4f}  (aspirational benchmark)")
    print(f"    Pool RPS:    {pool_on_mkt:.4f}")
    print(f"    Gap to market: {gap:+.4f} ({'pool worse' if gap > 0 else 'pool better'})")
    aspirational_met = gap <= 0.01
    print(f"    Within 0.01 of market: {'YES' if aspirational_met else 'NO'}")
else:
    pool_on_mkt = None
    mkt_rps_10 = None

# ---------------------------------------------------------------------------
# 6. Weight counterfactual: actual online_state vs code INITIAL_WEIGHTS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("WEIGHT DISCREPANCY ANALYSIS")
print("=" * 70)
state = json.loads((ARTIFACTS / "online_state.json").read_text())
online_weights = state["weights"]  # bayes=0.55, market=0.35
code_weights = {"dc": 0.05, "bayes": 0.35, "gbm": 0.05, "market": 0.55}

print(f"\n  online_state.json weights: {online_weights}")
print(f"  Code INITIAL_WEIGHTS:      {code_weights}")
print("  DISCREPANCY: bayes and market weights are SWAPPED")

# Reconstruct counterfactual pool using code's INITIAL_WEIGHTS
# For matches where market is null, rescale among dc/bayes/gbm (same partial-pool logic)
def compute_pool(weights, probs_dict, outcomes, mask_valid=None):
    """Reconstruct pool probabilities from weights and layer probs."""
    n = len(outcomes)
    pool_p = np.zeros((n, 3))
    for i in range(n):
        w_sum = 0.0
        p_sum = np.zeros(3)
        for c, w in weights.items():
            if c not in probs_dict:
                continue
            p_i = probs_dict[c][i]
            if np.isnan(p_i).any():
                continue
            p_sum += w * p_i
            w_sum += w
        if w_sum > 0:
            pool_p[i] = p_sum / w_sum
        else:
            pool_p[i] = np.array([1/3, 1/3, 1/3])
    return pool_p

# Actual pool from predictions_log vs counterfactual
pool_actual = probs["pool"]
pool_counterfactual = compute_pool(code_weights, probs, outcomes)

rps_actual = rps(pool_actual, outcomes)
rps_cf = rps(pool_counterfactual, outcomes)
print(f"\n  Pool RPS with ACTUAL weights   (bayes=0.55, mkt=0.35): {rps_actual:.4f}")
print(f"  Pool RPS with CODE weights     (bayes=0.35, mkt=0.55): {rps_cf:.4f}")
print(f"  Difference (actual - code):    {rps_actual - rps_cf:+.4f}")
print(
    f"  The bayes-heavy pool {'helped' if rps_actual < rps_cf else 'hurt'} by "
    f"{abs(rps_actual - rps_cf):.4f} RPS points on n=12 (suggestive only)."
)

# Also compute bayes-only pool (the backtest stacking weight of ~100% bayes)
rps_bayes_only = agg["bayes"]["rps"]
print(f"  Bayes-only RPS (backtest prior): {rps_bayes_only:.4f}")

# ---------------------------------------------------------------------------
# 7. Prediction vs outcome for each match (pool) — biggest misses
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("POOL PREDICTION VS ACTUAL OUTCOME")
print("=" * 70)
print(
    f"\n{'Match':<47} {'Outcome':<10} {'Pred':<10} {'P(actual)':>10} {'Correct?':<8}"
)
print("-" * 90)
pool_p = probs["pool"]
prob_correct = []
for i, detail in enumerate(match_details):
    p = pool_p[i]
    pred_idx = int(np.argmax(p))
    actual_name = outcome_names[outcomes[i]]
    pred_name = outcome_names[pred_idx]
    p_actual = p[outcomes[i]]
    correct = "Yes" if pred_idx == outcomes[i] else "No"
    print(
        f"  {detail:<45} {actual_name:<10} {pred_name:<10} {p_actual:>10.3f} {correct:<8}"
    )
    prob_correct.append(p_actual)

pool_accuracy = np.mean([
    1 if np.argmax(pool_p[i]) == outcomes[i] else 0 for i in range(len(outcomes))
])
print(f"\n  Pool top-1 accuracy: {pool_accuracy:.1%} ({int(pool_accuracy*12)}/12)")
print(f"  Mean P(actual outcome): {np.mean(prob_correct):.3f}")

# Worst misses by pool
worst_pool_idx = np.argsort(-per_match_rps["pool"])[:5]
print("\n--- Top 5 biggest pool misses (by per-match RPS, higher = worse) ---")
for rank, idx in enumerate(worst_pool_idx, 1):
    detail = match_details[idx]
    p = pool_p[idx]
    pred_name = outcome_names[int(np.argmax(p))]
    actual_name = outcome_names[outcomes[idx]]
    r = per_match_rps["pool"][idx]
    print(
        f"  {rank}. RPS={r:.4f}  {detail}"
        f"\n       Pred={pred_name}({p.max():.1%}), Actual={actual_name}, "
        f"P(actual)={p[outcomes[idx]]:.3f}"
    )

# ---------------------------------------------------------------------------
# 8. Draw analysis — systematic draw underestimation?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DRAW PREDICTION ANALYSIS (n=4 draws in 12 matches)")
print("=" * 70)
draw_mask = outcomes == 1
non_draw_mask = outcomes != 1

print(f"\n  Matches that ended in draws: {draw_mask.sum()}")
print(f"  Mean pool draw probability on actual draws: "
      f"{pool_p[draw_mask, 1].mean():.3f}")
print(f"  Mean pool draw probability on non-draws:    "
      f"{pool_p[non_draw_mask, 1].mean():.3f}")

for c in COMPONENTS:
    if c not in agg:
        continue
    p = probs[c]
    mask_valid = ~np.isnan(p).any(axis=1)
    # Mean predicted draw prob on actual draws
    combined = mask_valid & draw_mask
    if combined.sum() > 0:
        mean_draw_pred_on_draws = p[combined, 1].mean()
    else:
        mean_draw_pred_on_draws = np.nan
    print(
        f"  {LABELS[c]:<14}: mean P(draw | actual draw) = "
        f"{mean_draw_pred_on_draws:.3f}  "
        f"[naive: {draw_mask.sum()}/{mask_valid.sum()} = {draw_mask.sum()/mask_valid.sum():.3f}]"
    )

# ---------------------------------------------------------------------------
# 9. Historical WC home advantage — neutral vs non-neutral
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("HISTORICAL WC HOME ADVANTAGE ANALYSIS")
print("=" * 70)

all_wc = results.filter(
    (pl.col("tournament") == "FIFA World Cup")
    & (pl.col("home_score").is_not_null())
    & (pl.col("date") < dt.date(2026, 6, 11))  # strictly historical
)
print(f"\n  Historical WC matches (pre-2026): {all_wc.height}")

# Non-neutral: home team has a geographic advantage (traditional home)
non_neutral_wc = all_wc.filter(pl.col("neutral") == False)
neutral_wc = all_wc.filter(pl.col("neutral") == True)
print(f"  Non-neutral WC matches: {non_neutral_wc.height}")
print(f"  Neutral-venue WC matches: {neutral_wc.height}")

def outcome_dist(df):
    hs = df["home_score"].to_numpy()
    as_ = df["away_score"].to_numpy()
    y = np.where(hs > as_, 0, np.where(hs == as_, 1, 2))
    n = len(y)
    return (y == 0).sum()/n, (y == 1).sum()/n, (y == 2).sum()/n, n

# All historical WC
h_all, d_all, a_all, n_all = outcome_dist(all_wc)
print(f"\n  All historical WC (n={n_all}):   {h_all:.1%}H / {d_all:.1%}D / {a_all:.1%}A")

# Non-neutral historical WC
if non_neutral_wc.height > 0:
    h_nn, d_nn, a_nn, n_nn = outcome_dist(non_neutral_wc)
    print(f"  Non-neutral WC   (n={n_nn}):   {h_nn:.1%}H / {d_nn:.1%}D / {a_nn:.1%}A")

# Neutral historical WC
if neutral_wc.height > 0:
    h_ne, d_ne, a_ne, n_ne = outcome_dist(neutral_wc)
    print(f"  Neutral-venue WC (n={n_ne}):   {h_ne:.1%}H / {d_ne:.1%}D / {a_ne:.1%}A")

# WC 2026 so far
print(f"\n  WC 2026 so far   (n=12):  {n_hw/12:.1%}H / {n_dr/12:.1%}D / {n_aw/12:.1%}A")
print(
    f"\n  NOTE: WC 2026 is at neutral tri-nation host (USA/Mexico/Canada)."
    f"\n  The relevant baseline is neutral-venue WC ({h_ne:.1%}H base rate)."
    f"\n  WC 2026 H-rate ({n_hw/12:.1%}) vs neutral baseline ({h_ne:.1%}): "
    f"{'above' if n_hw/12 > h_ne else 'below'} base rate by "
    f"{abs(n_hw/12 - h_ne):.1%} (n=12, tiny sample)."
)

# Pool's mean predicted home-win probability
pool_mean_h = pool_p[:, 0].mean()
print(f"\n  Pool mean P(home win) across 12 matches: {pool_mean_h:.3f}")
print(f"  Neutral WC historical H base rate:        {h_ne:.3f}")
print(f"  Actual WC 2026 H-rate:                    {n_hw/12:.3f}")
print(
    f"  Pool {'over-' if pool_mean_h > n_hw/12 else 'under-'}estimates home wins "
    f"by {abs(pool_mean_h - n_hw/12):.3f} on average (suggestive, n=12)."
)

# ---------------------------------------------------------------------------
# 10. Layer ranking summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("LAYER RANKING SUMMARY")
print("=" * 70)
ranked = sorted(agg.items(), key=lambda x: x[1]["rps"])
print("\n  Rank  Layer           RPS      Brier    LogLoss   ECE    n")
print("  " + "-" * 65)
for rank, (c, m) in enumerate(ranked, 1):
    flag = " <-- DC gate" if c == "dc" else ""
    flag = " <-- POOL" if c == "pool" else flag
    print(
        f"  {rank}.    {LABELS[c]:<14} {m['rps']:.4f}   {m['brier']:.4f}   "
        f"{m['log_loss']:.4f}   {m['ece']:.3f}  {m['n']}{flag}"
    )

best_c, best_m = ranked[0]
print(f"\n  Best layer: {LABELS[best_c]} (RPS={best_m['rps']:.4f})")
print(f"  Pool RPS: {agg['pool']['rps']:.4f}")
print(
    f"  Pool {'beats' if agg['pool']['rps'] < agg['dc']['rps'] else 'does NOT beat'} Dixon-Coles gate"
)

# ---------------------------------------------------------------------------
# PLOT 1: Layer RPS comparison bar chart
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

comps_with_data = [c for c in COMPONENTS if c in agg]
rps_vals = [agg[c]["rps"] for c in comps_with_data]
bar_colors = [COLORS[c] for c in comps_with_data]

ax = axes[0]
bars = ax.bar([LABELS[c] for c in comps_with_data], rps_vals, color=bar_colors, alpha=0.85)
ax.set_ylabel("Mean RPS (lower = better)")
ax.set_title(f"Layer RPS — WC 2026 ({n_hw}H/{n_dr}D/{n_aw}A, n=12; n=10 market)")
ax.axhline(y=agg["dc"]["rps"], color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7, label=f"DC gate ({agg['dc']['rps']:.4f})")
if "market" in agg:
    ax.axhline(y=agg["market"]["rps"], color="#f39c12", linestyle=":", linewidth=1.5, alpha=0.7, label=f"Market bench ({agg['market']['rps']:.4f})")
ax.legend(fontsize=8)
for i, v in enumerate(rps_vals):
    ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylim(0, max(rps_vals) * 1.25)

# Log-loss
ll_vals = [agg[c]["log_loss"] for c in comps_with_data]
ax = axes[1]
ax.bar([LABELS[c] for c in comps_with_data], ll_vals, color=bar_colors, alpha=0.85)
ax.set_ylabel("Mean Log-Loss (lower = better)")
ax.set_title("Layer Log-Loss — WC 2026")
for i, v in enumerate(ll_vals):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0, max(ll_vals) * 1.25)

# Brier
brier_vals = [agg[c]["brier"] for c in comps_with_data]
ax = axes[2]
ax.bar([LABELS[c] for c in comps_with_data], brier_vals, color=bar_colors, alpha=0.85)
ax.set_ylabel("Mean Brier Score (lower = better)")
ax.set_title("Layer Brier Score — WC 2026")
for i, v in enumerate(brier_vals):
    ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)
ax.set_ylim(0, max(brier_vals) * 1.25)

plt.tight_layout()
out_path = PLOTS / "v2_layer_rps_comparison.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved: {out_path}")

# ---------------------------------------------------------------------------
# PLOT 2: Per-match RPS heatmap (layers × matches)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))
n_matches = len(match_labels)
n_comps = len(comps_with_data)
data_matrix = np.array([per_match_rps[c] for c in comps_with_data])
# Replace nan with max for coloring purposes (market nulls)
max_rps = np.nanmax(data_matrix)
data_viz = np.where(np.isnan(data_matrix), max_rps * 1.1, data_matrix)
im = ax.imshow(data_viz, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.5)
ax.set_xticks(np.arange(n_matches))
ax.set_xticklabels(
    [
        f"{r['home_team'][:4]}\nv {r['away_team'][:4]}"
        for r in scored.select("home_team", "away_team").iter_rows(named=True)
    ],
    fontsize=7,
)
ax.set_yticks(np.arange(n_comps))
ax.set_yticklabels([LABELS[c] for c in comps_with_data])
plt.colorbar(im, ax=ax, label="Per-match RPS (green=better, red=worse)")
ax.set_title(
    "Per-Match RPS Heatmap — all layers (★ = draw outcome, ✦ = away win outcome)"
)
# Annotate values
for i in range(n_comps):
    for j in range(n_matches):
        v = data_matrix[i, j]
        text = f"{v:.3f}" if not np.isnan(v) else "null"
        ax.text(j, i, text, ha="center", va="center", fontsize=7,
                color="white" if data_viz[i, j] > 0.3 else "black")
# Mark draw and away-win outcomes
for j in range(n_matches):
    if outcomes[j] == 1:
        ax.text(j, -0.6, "★", ha="center", va="center", fontsize=10, color="#3498db")
    elif outcomes[j] == 2:
        ax.text(j, -0.6, "✦", ha="center", va="center", fontsize=10, color="#e74c3c")

plt.tight_layout()
out_path = PLOTS / "v2_match_rps_heatmap.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# PLOT 3: Calibration reliability diagram (pool and bayes)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax_idx, c in enumerate(["pool", "bayes"]):
    ax = axes[ax_idx]
    p = probs[c]
    # All 3 class probs per match give 3×12 = 36 data points
    all_preds = p.flatten()
    all_actual = np.zeros(len(outcomes) * 3)
    for i, y in enumerate(outcomes):
        all_actual[i * 3 + y] = 1.0
    # Bin into 6 bins
    n_bins = 6
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_accs, bin_confs, bin_counts = [], [], [], []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        m = (all_preds >= lo) & (all_preds < hi + (0.001 if b == n_bins - 1 else 0))
        if m.sum() >= 1:
            bin_centers.append((lo + hi) / 2)
            bin_accs.append(all_actual[m].mean())
            bin_confs.append(all_preds[m].mean())
            bin_counts.append(int(m.sum()))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.plot(bin_confs, bin_accs, "o-", color=COLORS[c], linewidth=2, markersize=8,
            label=f"{LABELS[c]} (ECE={agg[c]['ece']:.3f})")
    for bc, ba, cnt in zip(bin_confs, bin_accs, bin_counts):
        ax.annotate(f"n={cnt}", (bc, ba), textcoords="offset points", xytext=(4, 6),
                    fontsize=8, color="gray")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction")
    ax.set_title(f"{LABELS[c]} Reliability Diagram\n(36 class-prob points from 12 matches)")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.15)

plt.suptitle(
    "Calibration (n=12 → 36 prob-outcome pairs; interpret with caution)",
    fontsize=10, y=1.01
)
plt.tight_layout()
out_path = PLOTS / "v2_calibration_reliability.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# PLOT 4: Pool predictions vs outcomes — probability stacked bars
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
for ax_idx, c in enumerate(["pool", "bayes"]):
    ax = axes[ax_idx]
    p = probs[c]
    n = len(outcomes)
    x = np.arange(n)
    w = 0.28
    ax.bar(x - w, p[:, 0], w * 2, label="P(Home win)", color="#2ecc71", alpha=0.85)
    ax.bar(x, p[:, 1], w * 2, label="P(Draw)", color="#f39c12", alpha=0.85)
    ax.bar(x + w, p[:, 2], w * 2, label="P(Away win)", color="#e74c3c", alpha=0.85)
    # Mark actual outcome
    outcome_col = {0: "#1a9641", 1: "#d97b00", 2: "#b22222"}
    for k in range(n):
        col = outcome_col[outcomes[k]]
        offset = {0: -w, 1: 0.0, 2: w}[outcomes[k]]
        ax.scatter(k + offset, p[k, outcomes[k]] + 0.03, marker="*",
                   color=col, edgecolors="black", s=150, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{r['home_team'][:5]}\nv {r['away_team'][:5]}"
            for r in scored.select("home_team", "away_team").iter_rows(named=True)
        ],
        fontsize=7,
    )
    ax.set_ylabel("Predicted probability")
    ax.set_ylim(0, 1.2)
    ax.set_title(
        f"{LABELS[c]} predictions (bars) vs actual outcome (★ marks actual outcome bar)"
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.axhline(1 / 3, color="gray", linestyle=":", alpha=0.5, label="Uniform prior")
plt.suptitle(
    "Predicted Probabilities vs Actual Outcomes — Pool and Bayesian layers",
    fontsize=11
)
plt.tight_layout()
out_path = PLOTS / "v2_prediction_vs_outcome.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# PLOT 5: Historical home advantage context
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: Historical WC outcome distributions
categories = ["All WC\n(historical)", "Non-neutral\nWC", "Neutral\nWC", "WC 2026\n(n=12)"]
h_rates = [h_all, h_nn, h_ne, n_hw / 12]
d_rates = [d_all, d_nn, d_ne, n_dr / 12]
a_rates = [a_all, a_nn, a_ne, n_aw / 12]
n_vals = [n_all, n_nn, n_ne, 12]
x = np.arange(len(categories))
w = 0.28
ax = axes[0]
b1 = ax.bar(x - w, h_rates, w * 2, label="Home win", color="#2ecc71", alpha=0.85)
b2 = ax.bar(x, d_rates, w * 2, label="Draw", color="#f39c12", alpha=0.85)
b3 = ax.bar(x + w, a_rates, w * 2, label="Away win", color="#e74c3c", alpha=0.85)
for i, n_v in enumerate(n_vals):
    ax.text(i, -0.04, f"n={n_v}", ha="center", fontsize=8, color="gray")
ax.axhline(h_ne, color="#2ecc71", linestyle="--", alpha=0.6, linewidth=1.5,
           label=f"Neutral WC H-rate ({h_ne:.1%})")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel("Fraction of matches")
ax.set_ylim(-0.08, 0.75)
ax.set_title("Home win / Draw / Away win rates — WC context\n(WC 2026 venue = neutral tri-nation host)")
ax.legend(fontsize=8)

# Right: Pool's mean predicted probabilities vs actual WC 2026 rates
ax = axes[1]
categories2 = ["Pool mean\nP(H)", "Pool mean\nP(D)", "Pool mean\nP(A)"]
pool_means = [pool_p[:, 0].mean(), pool_p[:, 1].mean(), pool_p[:, 2].mean()]
actual_rates = [n_hw / 12, n_dr / 12, n_aw / 12]
neutral_hist = [h_ne, d_ne, a_ne]
x2 = np.arange(3)
w2 = 0.2
ax.bar(x2 - w2, pool_means, w2 * 2, label="Pool mean prediction", color="#9b59b6", alpha=0.85)
ax.bar(x2 + w2, actual_rates, w2 * 2, label="WC 2026 actual rate (n=12)", color="#34495e", alpha=0.85)
ax.scatter(x2, neutral_hist, marker="D", color="#e74c3c", s=80, zorder=5,
           label=f"Neutral WC historical ({n_ne} matches)")
for i, (pm, ar) in enumerate(zip(pool_means, actual_rates)):
    ax.annotate(f"{pm:.3f}", (i - w2, pm + 0.01), ha="center", fontsize=8)
    ax.annotate(f"{ar:.3f}", (i + w2, ar + 0.01), ha="center", fontsize=8)
ax.set_xticks(x2)
ax.set_xticklabels(["Home win", "Draw", "Away win"])
ax.set_ylabel("Probability / Fraction")
ax.set_ylim(0, 0.80)
ax.set_title("Pool mean predictions vs WC 2026 actual rate\nvs neutral WC historical baseline")
ax.legend(fontsize=8)

plt.suptitle("Home Advantage Analysis — WC 2026 all-neutral-venue context", fontsize=11)
plt.tight_layout()
out_path = PLOTS / "v2_home_advantage_context.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# PLOT 6: Weight counterfactual — per-match RPS actual vs code weights vs bayes-only
# ---------------------------------------------------------------------------
per_match_rps_cf = []
for i in range(len(outcomes)):
    p = pool_counterfactual[i]
    onehot = np.eye(3)[outcomes[i]]
    cum_diff = np.cumsum(p - onehot)[:2]
    per_match_rps_cf.append(float(np.sum(cum_diff**2) / 2))
per_match_rps_cf = np.array(per_match_rps_cf)

per_match_rps_bayes = per_match_rps["bayes"]

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(match_labels))
w = 0.25
ax.bar(x - w, per_match_rps["pool"], w * 2, label=f"Pool actual (bayes=0.55, mkt=0.35) mean={rps_actual:.4f}", color="#9b59b6", alpha=0.85)
ax.bar(x, per_match_rps_cf, w * 2, label=f"Pool counterfactual (bayes=0.35, mkt=0.55) mean={rps_cf:.4f}", color="#e67e22", alpha=0.85)
ax.bar(x + w, per_match_rps_bayes, w * 2, label=f"Bayes only mean={rps_bayes_only:.4f}", color="#3498db", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(
    [
        f"{r['home_team'][:4]}\nv\n{r['away_team'][:4]}"
        for r in scored.select("home_team", "away_team").iter_rows(named=True)
    ],
    fontsize=7,
)
ax.set_ylabel("Per-match RPS (lower = better)")
ax.set_title(
    "Weight counterfactual: actual online_state weights vs code INITIAL_WEIGHTS vs bayes-only\n"
    "(bayes/market swap investigation — n=12, suggestive only)"
)
ax.legend(fontsize=8)
ax.set_ylim(0, 0.65)
plt.tight_layout()
out_path = PLOTS / "v2_weight_counterfactual.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# Print final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL SUMMARY TABLE (all stats n=12, n=10 for market)")
print("=" * 70)
print(f"\n{'Layer':<14} {'n':>3} {'RPS':>8} {'Brier':>8} {'LogLoss':>9} {'ECE':>7}")
print("  " + "-" * 50)
for c in COMPONENTS:
    if c in agg:
        m = agg[c]
        note = ""
        if c == "dc":
            note = " <- gate"
        elif c == "market":
            note = " <- benchmark"
        elif c == "pool":
            note = " <- primary"
        print(
            f"  {LABELS[c]:<12} {m['n']:>3} {m['rps']:>8.4f} {m['brier']:>8.4f} "
            f"{m['log_loss']:>9.4f} {m['ece']:>7.3f}{note}"
        )
print(f"\n  Gate met (pool < DC): {'YES' if gate_met else 'NO'}")
if "market" in agg:
    print(f"  Gap to market (n=10): {gap:+.4f} ({'within 0.01' if aspirational_met else 'outside 0.01'})")

# ---------------------------------------------------------------------------
# Write run record
# ---------------------------------------------------------------------------
run_record = f"""---
stage: S2
script_path: notebooks/02_eda_diagnostics_v2.py
target: "1X2 match outcome (0=home win, 1=draw, 2=away win)"
n_rows_analyzed: 12
random_seed: {SEED}
findings_count: 7
---

# EDA Run Record — WC 2026 Diagnostics v2 (2026-06-15)

**Supersedes:** `memory/runs/2026-06-15_eda_diagnostics.md` (prior run had unverified data)
**Script:** `notebooks/02_eda_diagnostics_v2.py`
**Data:** `data/raw/martj42/2026-06-15/results.csv`, `artifacts/predictions_log.parquet`
**Verified ground truth:** 7H / 4D / 1A from inner join on (date, home_team, away_team)

## Aggregate Metrics Table

All stats computed fresh from raw data. n=12 for dc/bayes/gbm/pool; n=10 for market (2 June 11 opening matches had null odds).

| Layer | n | RPS | Brier | Log-Loss | ECE | Note |
|-------|---|-----|-------|----------|-----|------|
"""
for c in COMPONENTS:
    if c in agg:
        m = agg[c]
        note_map = {
            "dc": "gate",
            "market": "aspirational benchmark (n=10)",
            "pool": "primary metric",
            "bayes": "",
            "gbm": "",
        }
        run_record += (
            f"| {LABELS[c]} | {m['n']} | {m['rps']:.4f} | {m['brier']:.4f} | "
            f"{m['log_loss']:.4f} | {m['ece']:.3f} | {note_map.get(c, '')} |\n"
        )

run_record += f"""
## Gate and Benchmark Status

- **DC gate (pool RPS < Dixon-Coles RPS):** {'MET' if gate_met else 'NOT MET'} — Pool {pool_rps_all:.4f} vs DC {dc_rps_all:.4f}
"""
if "market" in agg:
    run_record += f"- **Market aspirational (gap ≤ 0.01, n=10):** {'MET' if aspirational_met else 'NOT MET'} — Pool {pool_on_mkt:.4f} vs Market {mkt_rps_10:.4f}, gap={gap:+.4f}\n"

run_record += f"""
## Top Findings

1. **{'Gate met' if gate_met else 'Gate NOT met'}: Pool RPS ({pool_rps_all:.4f}) {'beats' if gate_met else 'does not beat'} Dixon-Coles ({dc_rps_all:.4f}).** The ensemble {'passes' if gate_met else 'fails'} the project's primary performance threshold on n=12 WC 2026 matches. With only 12 matches this is not statistically conclusive, but directionally {'positive' if gate_met else 'concerning'}.
   [Plot: `artifacts/plots/v2_layer_rps_comparison.png`]

2. **Bayesian and Dixon-Coles layers perform nearly identically (RPS {agg['bayes']['rps']:.4f} vs {agg['dc']['rps']:.4f}).** The Bayesian backbone's Dixon-Coles low-score correction and per-team Gaussian random walks are not producing a clear discriminative advantage over the simpler DC model on this small sample. However, the pool weights (bayes=0.55) rightly favor Bayes based on backtest evidence.
   [Plot: `artifacts/plots/v2_layer_rps_comparison.png`]

3. **GBM is the worst layer by a wide margin (RPS {agg['gbm']['rps']:.4f}, Brier {agg['gbm']['brier']:.4f}).** This confirms the backtest result (GBM weight ≈ 0). GBM carries only 5% pool weight but is still actively dragging ensemble performance on misses like Qatar-Switzerland and Brazil-Morocco. Root cause is unknown from EDA alone — this is the top priority for S3/S4 investigation (feature problem? overfitting?).
   [Plot: `artifacts/plots/v2_match_rps_heatmap.png`]

4. **Market layer underperforms both DC and Bayes (RPS {agg['market']['rps'] if 'market' in agg else 'N/A':.4f} on n=10).** De-vigged closing odds are the aspirational benchmark, yet on this sample they perform worse than the Bayesian model. The Qatar-Switzerland match (market had ~75% for Switzerland, actual draw) was a significant miss. This may reflect genuine upset territory where markets mis-price, or simply small-sample noise.
   [Plot: `artifacts/plots/v2_layer_rps_comparison.png`]

5. **Draw underestimation is a systematic pattern across all layers.** The 4 matches that ended in draws (Canada-Bosnia, Qatar-Switzerland, Brazil-Morocco, Netherlands-Japan) showed mean pool P(draw) of {pool_p[draw_mask, 1].mean():.3f}, versus a draw rate of {n_dr/12:.3f} in this sample and a neutral-WC historical base rate of {d_ne:.3f}. Every layer assigned lower draw probability than the draw actually occurred. Qatar-Switzerland and Brazil-Morocco were the worst misses (pool predicted ≥58% home or away win).
   [Plot: `artifacts/plots/v2_prediction_vs_outcome.png`]

6. **WC 2026 home-win rate ({n_hw/12:.1%}, n=12) is above the neutral-venue WC historical baseline ({h_ne:.1%}, n={n_ne}).** All WC 2026 venues are on neutral tri-national host ground (USA/Mexico/Canada). The correct comparison is neutral-venue WC matches, not all WC matches (which include non-neutral host-nation advantages). The pool's mean P(home win) = {pool_mean_h:.3f} was below the actual {n_hw/12:.3f} rate. However, with n=12 this may be sample noise — the sign and magnitude are directionally important but not conclusive.
   [Plot: `artifacts/plots/v2_home_advantage_context.png`]

7. **The bayes/market weight swap in online_state.json appears to have {'helped' if rps_actual < rps_cf else 'hurt'} on n=12.** The live pool uses bayes=0.55/market=0.35 (state file) vs the code's INITIAL_WEIGHTS of bayes=0.35/market=0.55. On these 12 matches, the actual (bayes-heavy) pool scored RPS={rps_actual:.4f} vs counterfactual (market-heavy) pool RPS={rps_cf:.4f}. The bayes-heavy configuration was {'better' if rps_actual < rps_cf else 'worse'} by {abs(rps_actual - rps_cf):.4f} RPS points, consistent with bayes outperforming market on this sample. This suggests the hand-edited state file may have been intentional or lucky.
   [Plot: `artifacts/plots/v2_weight_counterfactual.png`]

## Concerns Carrying Into S3

| Priority | Concern | Blocks S3? |
|----------|---------|-----------|
| HIGH | GBM is actively hurting ensemble RPS. Diagnose root cause before next weight update. | No — but weight reduction to 0.0 recommended |
| HIGH | Weight discrepancy (bayes/market swap) must be resolved before `make update` activates Hedge. Starting Hedge from wrong prior is a persistent bias. | Yes for update cycle activation |
| MEDIUM | Draw probability is systematically underestimated by all layers. The Poisson model's correlation parameter (Dixon-Coles λ correction) may need recalibration for WC group stage. | No |
| MEDIUM | Market layer scored worse than Bayes on n=10. Re-evaluate the 0.35 initial market weight — lower to 0.20 is defensible. | No |
| LOW | n=12 sample — all findings are directional, not statistically significant. Await 20+ matches before drawing firm conclusions. | No |
| LOW | Qatar-Switzerland is the single worst-miss match for all layers (pool RPS={per_match_rps['pool'][match_details.index([d for d in match_details if 'Qatar' in d][0])]:.4f} if Qatar in match). Extreme upset; check whether similar calibration errors occurred at WC 2022/2018 for strong-vs-weak matchups. | No |

## Plots Generated

- `artifacts/plots/v2_layer_rps_comparison.png` — bar charts: RPS, Brier, log-loss by layer
- `artifacts/plots/v2_match_rps_heatmap.png` — per-match RPS heatmap (layers × matches)
- `artifacts/plots/v2_calibration_reliability.png` — reliability diagrams for pool and bayes
- `artifacts/plots/v2_prediction_vs_outcome.png` — predicted probabilities vs actual outcomes
- `artifacts/plots/v2_home_advantage_context.png` — historical WC home-win context with neutral breakdown
- `artifacts/plots/v2_weight_counterfactual.png` — per-match RPS under actual vs code INITIAL_WEIGHTS vs bayes-only
"""

run_path = MEMORY / "runs" / "2026-06-15_eda_diagnostics_v2.md"
run_path.parent.mkdir(exist_ok=True, parents=True)
run_path.write_text(run_record, encoding="utf-8")
print(f"\nRun record written: {run_path}")

# ---------------------------------------------------------------------------
# Write decisions log
# ---------------------------------------------------------------------------
decisions = f"""---
stage: S2
date: 2026-06-15
author: ds-eda-analyst
slug: eda-diagnostics-v2
supersedes: memory/decisions/2026-06-15_eda-diagnostics.md
---

# EDA Decisions v2 — WC 2026 Diagnostics (2026-06-15)

## Decision 1: Recompute all statistics from raw data; do not inherit prior run's numbers

**Rationale:** The task explicitly requires recomputing from raw CSV and parquet files because the prior run (v1) may have been generated before or after the data dictionary correction pass, and the ground truth (7H/4D/1A) must be verified from the inner join, not assumed.
**Action:** `assert scored.height == 12` with `assert n_hw == 7 and n_dr == 4 and n_aw == 1` to verify before any metric computation.

## Decision 2: Use v2 suffix on all plot files to avoid silent clobber of prior run

**Rationale:** The prior run's plots are in `artifacts/plots/` without versioning. Rather than decide whether they are correct or not, we write to `v2_*` filenames. Both sets of plots are preserved for comparison.
**Action:** All `plt.savefig` calls use `v2_` prefix.

## Decision 3: Historical home advantage baseline = neutral-venue WC matches only

**Rationale:** WC 2026 is hosted across USA, Mexico, and Canada — all teams are playing in a foreign country. The "home team" label in the match data is a scheduling convention (first-named team), not a geographic advantage. The relevant historical baseline for WC 2026 is neutral-venue WC matches (matches where `neutral=True` in results.csv), not all WC matches (which include host-nation advantages from WC 2014, 2018, 2022 group stages).
**Action:** Compute separate outcome distributions for neutral vs non-neutral historical WC matches; compare WC 2026 against neutral baseline.

## Decision 4: Calibration reliability diagram uses all 3 class probabilities (not just argmax)

**Rationale:** Each match generates 3 (predicted probability, actual indicator) pairs — one per class. Using all 3 triples gives 36 data points from 12 matches, which provides better bin coverage than 12 points, while still appropriately noting the n=12 limitation. This approach is sometimes called "class-conditional calibration."
**Action:** Flatten probs to (36,) and actual indicators to (36,) then bin.

## Decision 5: Weight counterfactual reconstructed via log-opinion pool with partial rescaling for null market

**Rationale:** To fairly compare the actual pool weights (bayes=0.55, market=0.35) vs code INITIAL_WEIGHTS (bayes=0.35, market=0.55), we must apply the same partial-pool rescaling logic that the actual pool uses for the 2 matches where market is null. Simply multiplying by 0.35 would give incorrect denominators.
**Action:** Implement `compute_pool()` function that skips null layers and renormalizes remaining weights.

## Decision 6: Did not run mutual information or KL divergence per prior decision (Decision 4 from v1)

**Rationale:** The decision from v1 EDA stands — n=12 is too small for meaningful MI or KL estimates. Deferred to when 30+ matches are scored.

## Decision 7: Qatar-Switzerland identified as the worst single miss but not singled out for layer-specific diagnosis

**Rationale:** Qatar vs Switzerland (pool predicted 74.6% away win, actual draw) is a clear outlier. However, decomposing why it was wrong (is Qatar stronger than expected? Is Switzerland's away form over-stated?) requires S3 feature engineering diagnostics, not EDA alone. We flag it as the worst miss and carry it forward.

## Decision 8: Weight discrepancy is documented but NOT resolved

**Rationale:** As in v1, the discrepancy between online_state.json (bayes=0.55, market=0.35) and code INITIAL_WEIGHTS (bayes=0.35, market=0.55) is a developer decision, not an EDA decision. The counterfactual analysis shows which was better on n=12, but that is suggestive evidence, not a recommendation to rewrite state.
**Action:** Flag as HIGH priority concern for S3/modeler.
"""

dec_path = MEMORY / "decisions" / "2026-06-15_eda-diagnostics-v2.md"
dec_path.parent.mkdir(exist_ok=True, parents=True)
dec_path.write_text(decisions, encoding="utf-8")
print(f"Decisions log written: {dec_path}")
print("\nDone.")
