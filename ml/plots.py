"""
plots.py — Analysis graphs for clone fragment change forecasting.
"""
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)


def plot_rolling_mcc(metrics_df, model_names, tag, output_dir):
    """Rolling MCC over revisions, one line per model."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for mn in model_names:
        sub = metrics_df[metrics_df["model"] == mn].sort_values("revision")
        if sub.empty:
            continue
        roll = sub["mcc"].rolling(window=10, min_periods=1).mean()
        ax.plot(sub["revision"], roll, label=mn, linewidth=1.3, alpha=0.85)
    ax.set_xlabel("Revision")
    ax.set_ylabel("Rolling MCC (window=10)")
    ax.set_title(f"Fragment-Level Standing Prediction — MCC Over Revisions ({tag})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    p = output_dir / f"standing_{tag}_rolling_mcc.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Plot -> {p}")


def plot_confusion_matrices(global_data, model_names, tag, output_dir):
    """Confusion matrix heatmap for each model."""
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, mn in zip(axes, model_names):
        d = global_data[mn]
        cm = confusion_matrix(d["y_true"], d["y_pred"], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Ind(0)", "Dep(1)"])
        ax.set_yticklabels(["Ind(0)", "Dep(1)"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(mn)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.suptitle(f"Confusion Matrices ({tag})", fontsize=13)
    p = output_dir / f"standing_{tag}_confusion_matrix.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Plot -> {p}")


def plot_roc_curves(global_data, model_names, tag, output_dir):
    """ROC curves for all models on one figure."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for mn in model_names:
        d = global_data[mn]
        yt, yp = np.array(d["y_true"]), np.array(d["y_proba"])
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{mn} (AUC={roc_auc:.3f})", linewidth=1.3)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curves ({tag})")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    p = output_dir / f"standing_{tag}_roc_curve.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Plot -> {p}")


def plot_pr_curves(global_data, model_names, tag, output_dir):
    """Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for mn in model_names:
        d = global_data[mn]
        yt, yp = np.array(d["y_true"]), np.array(d["y_proba"])
        if len(np.unique(yt)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(yt, yp)
        ap = average_precision_score(yt, yp)
        ax.plot(rec, prec, label=f"{mn} (AP={ap:.3f})", linewidth=1.3)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves ({tag})")
    ax.legend(loc="lower left"); ax.grid(True, alpha=0.3)
    p = output_dir / f"standing_{tag}_pr_curve.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Plot -> {p}")


def plot_feature_importance(models, feature_cols, tag, output_dir):
    """Feature importance bar chart for each model."""
    for mn, model in models.items():
        if not hasattr(model, "feature_importances_"):
            continue
        imp = pd.DataFrame({"feature": feature_cols,
                            "importance": model.feature_importances_})
        imp = imp.sort_values("importance", ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp["feature"], imp["importance"], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top 15 Features — {mn} ({tag})")
        p = output_dir / f"standing_{tag}_feature_importance_{mn}.png"
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        # Also save CSV
        full_imp = pd.DataFrame({"feature": feature_cols,
                                 "importance": model.feature_importances_})
        full_imp.sort_values("importance", ascending=False).to_csv(
            output_dir / f"standing_{tag}_feature_importance_{mn}.csv", index=False)
    print(f"  Feature importance plots saved")


def plot_model_comparison(global_metrics_df, tag, output_dir):
    """Bar chart comparing all models across key metrics."""
    metrics = ["mcc", "balanced_acc", "gmean", "f1", "auc_roc"]
    labels = ["MCC", "Bal. Acc", "G-mean", "F1", "AUC-ROC"]
    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (_, row) in enumerate(global_metrics_df.iterrows()):
        vals = [row.get(m, 0) for m in metrics]
        ax.bar(x + i * width, vals, width, label=row["model"], alpha=0.85)
    ax.set_xticks(x + width * (len(global_metrics_df) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Model Comparison ({tag})")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    p = output_dir / f"standing_{tag}_model_comparison.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Plot -> {p}")


def plot_label_distribution(y_all, tag, output_dir):
    """Pie chart of actual label distribution."""
    dep = int(np.sum(np.array(y_all) == 1))
    ind = int(np.sum(np.array(y_all) == 0))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie([dep, ind], labels=[f"Dependent ({dep})", f"Independent ({ind})"],
           autopct="%1.1f%%", colors=["#4A90D9", "#D9534F"], startangle=90)
    ax.set_title(f"Actual Label Distribution ({tag})")
    p = output_dir / f"standing_{tag}_label_distribution.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Plot -> {p}")
