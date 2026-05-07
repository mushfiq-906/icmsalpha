"""
visualize_classifier.py
────────────────────────────────────────────────────────────────────────────
Plots for the fragment classifier (dependent vs independent):

  1. Per-fragment trajectory plot — for top-N gcids by number of decision
     events, draws each gcid as a horizontal lane along decision_index;
     correct predictions are green dots, wrong ones are red Xs. Directly
     visualizes the user's mental picture: "for fragment X, did the model
     correctly classify each successive change event?"
  2. Confusion-matrix heatmaps per model.
  3. Label-drift plot: rolling fraction of label==1 (dependent) per
     verification_revision.
  4. Feature-importance bar chart for the best model.

Inputs are produced by walk_forward_classifier.py.

Usage
─────
  python ml/visualize_classifier.py \\
    --predictions ml/results/classifier_predictions_Ctags_Type3_Block.csv
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def derive_tag_from_predictions(pred_path: Path) -> str:
    name = pred_path.stem  # classifier_predictions_<tag>
    return name.replace("classifier_predictions_", "")


def pick_best_model(metrics_path: Path) -> str:
    if not metrics_path.exists():
        return None
    m = pd.read_csv(metrics_path)
    if m.empty:
        return None
    best = m.loc[m["mcc"].idxmax()]
    return str(best["model"])


def plot_trajectory(pred_df: pd.DataFrame, model: str, tag: str, out_path: Path,
                    top_n: int = 12):
    sub = pred_df[pred_df["model"] == model].copy()
    if sub.empty:
        print(f"  [WARN] No predictions for model {model}; skipping trajectory plot.")
        return
    counts = sub["gcid"].value_counts().head(top_n)
    if counts.empty:
        print("  [WARN] No gcids to plot.")
        return
    top_gcids = counts.index.tolist()

    fig, ax = plt.subplots(figsize=(14, max(4, 0.6 * len(top_gcids))))

    for lane, gcid in enumerate(top_gcids):
        rows = sub[sub["gcid"] == gcid].sort_values("decision_index")
        x = rows["decision_index"].values
        y = np.full_like(x, lane, dtype=float)
        correct = (rows["y_pred"].values == rows["y_true"].values)
        # background line for the lane
        ax.plot(x, y, color="lightgray", linewidth=1.0, zorder=1)
        # correct: green dot
        if correct.any():
            ax.scatter(x[correct], y[correct], c="#2ca02c", s=70, marker="o",
                       zorder=3, edgecolors="white", linewidths=0.6)
        # wrong: red X
        if (~correct).any():
            ax.scatter(x[~correct], y[~correct], c="#d62728", s=90, marker="X",
                       zorder=3, edgecolors="white", linewidths=0.6)

    ax.set_yticks(range(len(top_gcids)))
    ax.set_yticklabels([f"gcid {g} (n={counts[g]})" for g in top_gcids])
    ax.set_xlabel("Decision index (k = k-th change event of this fragment)")
    ax.set_title(f"Per-fragment classification trajectory — {tag}  [{model}]")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markersize=9, label="Correct"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#d62728",
               markersize=10, label="Wrong"),
    ]
    ax.legend(handles=legend, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Trajectory plot -> {out_path}")


def plot_confusion(pred_df: pd.DataFrame, tag: str, out_path: Path):
    models = pred_df["model"].unique().tolist()
    n = len(models)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)

    for idx, m in enumerate(models):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        sub = pred_df[pred_df["model"] == m]
        cm = pd.crosstab(sub["y_true"], sub["y_pred"], dropna=False)
        cm = cm.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        im = ax.imshow(cm.values, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["independent (0)", "dependent (1)"])
        ax.set_yticklabels(["independent (0)", "dependent (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(m)
        for i in range(2):
            for j in range(2):
                v = cm.values[i, j]
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > cm.values.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.suptitle(f"Confusion matrices — {tag}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion plot  -> {out_path}")


def plot_label_drift(pred_df: pd.DataFrame, tag: str, out_path: Path):
    # Use one model (any) — y_true is the same across models per row
    one_model = pred_df["model"].iloc[0]
    sub = pred_df[pred_df["model"] == one_model].copy()
    rev_lab = sub.groupby("verification_revision")["y_true"].agg(["sum", "count"])
    rev_lab["frac_dependent"] = rev_lab["sum"] / rev_lab["count"]
    rev_lab["rolling"] = rev_lab["frac_dependent"].rolling(window=10, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(rev_lab.index, rev_lab["frac_dependent"], color="lightgray",
            linewidth=0.8, label="Per-revision")
    ax.plot(rev_lab.index, rev_lab["rolling"], color="#1f77b4",
            linewidth=1.6, label="Rolling (window=10)")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Verification revision")
    ax.set_ylabel("Fraction dependent (label=1)")
    ax.set_title(f"Label drift over time — {tag}")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Label drift     -> {out_path}")


def plot_feature_importance(imp_path: Path, tag: str, model: str, out_path: Path,
                            top_k: int = 20):
    if not imp_path.exists():
        print(f"  [WARN] No importance file at {imp_path}; skipping.")
        return
    imp = pd.read_csv(imp_path).head(top_k)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(imp))))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="#1f77b4")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-{top_k} features ({model}) — {tag}")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Feature imp plot-> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize fragment-classifier walk-forward results.")
    parser.add_argument("--predictions", required=True,
                        help="Path to classifier_predictions_<tag>.csv")
    parser.add_argument("--model", default=None,
                        help="Model to plot trajectory for (default: best by global MCC)")
    parser.add_argument("--top-n", type=int, default=12,
                        help="Number of top gcids in trajectory plot (default: 12)")
    parser.add_argument("--output", default=None,
                        help="Output directory for plots (default: same dir as predictions/plots)")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    tag = derive_tag_from_predictions(pred_path)
    results_dir = pred_path.parent
    out_dir = Path(args.output) if args.output else results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions: {pred_path}")
    pred_df = pd.read_csv(pred_path)
    print(f"  Rows: {len(pred_df):,}, Models: {pred_df['model'].nunique()}, "
          f"Unique gcids: {pred_df['gcid'].nunique()}")

    # Pick best model
    metrics_path = results_dir / f"classifier_metrics_{tag}.csv"
    chosen_model = args.model or pick_best_model(metrics_path)
    if chosen_model is None:
        chosen_model = pred_df["model"].iloc[0]
    print(f"  Using model for trajectory + importance: {chosen_model}")

    # 1. Trajectory
    plot_trajectory(pred_df, chosen_model, tag,
                    out_dir / f"classifier_trajectory_{tag}_{chosen_model}.png",
                    top_n=args.top_n)

    # 2. Confusion matrices
    plot_confusion(pred_df, tag,
                   out_dir / f"classifier_confusion_{tag}.png")

    # 3. Label drift
    plot_label_drift(pred_df, tag,
                     out_dir / f"classifier_label_drift_{tag}.png")

    # 4. Feature importance for best model
    imp_path = results_dir / f"classifier_feature_importance_{tag}_{chosen_model}.csv"
    plot_feature_importance(imp_path, tag, chosen_model,
                            out_dir / f"classifier_feature_importance_{tag}_{chosen_model}.png")


if __name__ == "__main__":
    main()
