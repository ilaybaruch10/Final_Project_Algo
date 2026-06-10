from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_true_vs_pred(y_true, y_pred, out_path, title="RR: True vs Predicted"):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.scatter(y_true, y_pred, s=18)
    mn = min(np.min(y_true), np.min(y_pred)); mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True RR [breaths/min]"); plt.ylabel("Predicted RR [breaths/min]")
    plt.title(title); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def plot_time_series(y_true, y_pred, out_path, title="RR over clips"):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(y_true, label="True RR"); plt.plot(y_pred, label="Predicted RR")
    plt.xlabel("Clip index"); plt.ylabel("RR [breaths/min]")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def plot_bland_altman(y_true, y_pred, out_path, title="Bland-Altman RR"):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    mean = (np.asarray(y_true) + np.asarray(y_pred)) / 2
    diff = np.asarray(y_pred) - np.asarray(y_true)
    md = diff.mean(); sd = diff.std()
    plt.figure()
    plt.scatter(mean, diff, s=18)
    plt.axhline(md, linestyle="-"); plt.axhline(md + 1.96*sd, linestyle="--"); plt.axhline(md - 1.96*sd, linestyle="--")
    plt.xlabel("Mean RR [breaths/min]"); plt.ylabel("Predicted - True [breaths/min]")
    plt.title(title); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
