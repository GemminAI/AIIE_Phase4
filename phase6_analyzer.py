"""
phase6_analyzer.py
Gemmina Intelligence LLC / Pure Information Laboratory
AIES 2026 — Control-Theoretic Edition
Phase 6 Expansion: Day 4 Analysis & Figure Generation

Input:  JSON files produced by phase6_ccr_full.py (RunPod)
Output: AIES-ready figures (PNG, 300dpi) + statistical summary (JSON)

Usage:
  python3 phase6_analyzer.py --data_dir /path/to/results --out_dir /path/to/figures
"""

import os
import json
import argparse
import numpy as np
from scipy import stats
from scipy.interpolate import make_interp_spline
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# AIES / AAAI style
plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.size":      12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize":11,
    "xtick.labelsize":11,
    "ytick.labelsize":11,
    "figure.dpi":     300,
})

EMERALD = "#10b981"
ROSE    = "#ef4444"
BLUE    = "#3b82f6"
NAVY    = "#1e3a8a"


# ===========================================================================
# Utilities
# ===========================================================================

def _cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / (pooled + 1e-12)


def _bootstrap_auc_ci(labels, scores, n=10000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    arr_l, arr_s = np.array(labels), np.array(scores)
    N = len(arr_l)
    for _ in range(n):
        idx = rng.randint(0, N, N)
        lb, sc = arr_l[idx], arr_s[idx]
        if len(np.unique(lb)) < 2:
            continue
        aucs.append(roc_auc_score(lb, sc))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _extract_a_l(records, layer_key="Layer_16"):
    """Extract A_l (Attractor Absorption Coefficient) for Correct and TypeC."""
    correct, typec = [], []
    for r in records:
        ls = r.get("layer_scans", {})
        if layer_key not in ls:
            continue
        correct.append(ls[layer_key]["Correct"]["coefficients"]["Attractor_Absorption_A_l"])
        typec.append(  ls[layer_key]["TypeC"  ]["coefficients"]["Attractor_Absorption_A_l"])
    return np.array(correct), np.array(typec)


# ===========================================================================
# Figure 1: Variance Collapse (Violin) — Exp A main result
# ===========================================================================

def fig_variance_collapse(data_files, out_dir, layer_key="Layer_16"):
    """
    [Fig 1] Attractor Absorption Coefficient distribution.
    Proves: Type C exhibits variance collapse (fixation) vs. Correct (controllable).
    Aggregates across all model-domain pairs for cross-model/cross-domain claim.
    """
    print("[Fig 1] Variance Collapse...")
    all_correct, all_typec = [], []

    for fpath in data_files:
        records = _load_json(fpath)
        c, t = _extract_a_l(records, layer_key)
        all_correct.extend(c)
        all_typec.extend(t)

    all_correct = np.array(all_correct)
    all_typec   = np.array(all_typec)

    t_stat, p_val = stats.ttest_ind(all_typec, all_correct, equal_var=False)
    d_val = _cohens_d(all_typec, all_correct)
    labels = [0]*len(all_correct) + [1]*len(all_typec)
    scores = list(all_correct) + list(all_typec)
    auc    = roc_auc_score(labels, scores)
    ci_lo, ci_hi = _bootstrap_auc_ci(labels, scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    parts = ax.violinplot(
        [all_correct, all_typec],
        showmeans=True, showextrema=True
    )
    colors = [EMERALD, ROSE]
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    for key in ("cmeans", "cmaxes", "cmins", "cbars"):
        if key in parts:
            parts[key].set_color("black")

    ax.set_xticks([1, 2])
    ax.set_xticklabels([
        f"Corrigible (Correct)\nn={len(all_correct)}",
        f"Rigid (Type C)\nn={len(all_typec)}"
    ])
    ax.set_ylabel(r"Attractor Absorption Coefficient ($A_l$)")
    ax.set_title("Variance Collapse in Epistemic Rigidity\n(Aggregated: all model-domain pairs)")

    stat_text = (f"Cohen's d = {d_val:.2f}\n"
                 f"p = {p_val:.1e}\n"
                 f"AUC = {auc:.4f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    ax.text(1.5, ax.get_ylim()[1] * 0.92, stat_text,
            ha="center", va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
            fontsize=11)

    out = os.path.join(out_dir, "fig1_variance_collapse.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")

    return {
        "correct_mean": float(all_correct.mean()),
        "correct_std":  float(all_correct.std(ddof=1)),
        "typec_mean":   float(all_typec.mean()),
        "typec_std":    float(all_typec.std(ddof=1)),
        "cohens_d":     float(d_val),
        "p_value":      float(p_val),
        "auc":          float(auc),
        "auc_ci":       [float(ci_lo), float(ci_hi)],
    }


# ===========================================================================
# Figure 2: Cross-Model × Cross-Domain AUC Grid — Exp A
# ===========================================================================

def fig_cross_model_domain(data_dir, out_dir, layer_key="Layer_16"):
    """
    [Fig 2] 2x2 AUC grid: Llama-3 / Mistral-7B × Legal / Biomedical.
    Primary evidence for generalizability claim: AUC > 0.96 across all pairs.
    """
    print("[Fig 2] Cross-Model / Cross-Domain AUC Grid...")

    model_tags  = ["Meta-Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.2"]
    domain_tags = ["Legal", "Biomedical"]
    model_labels  = ["Llama-3-8B", "Mistral-7B"]
    domain_labels = ["Legal", "Biomedical"]

    grid_auc = np.full((len(model_tags), len(domain_tags)), np.nan)
    grid_ci  = {}

    for i, m in enumerate(model_tags):
        fpath = os.path.join(data_dir, f"expA_{m}_L16_k1.json")
        if not os.path.exists(fpath):
            print(f"  [Warning] Not found: {fpath}")
            continue
        records = _load_json(fpath)
        for j, domain in enumerate(domain_tags):
            domain_records = [r for r in records if r["domain"] == domain]
            c, t = _extract_a_l(domain_records, layer_key)
            if len(c) == 0:
                continue
            labels = [0]*len(c) + [1]*len(t)
            scores = list(c) + list(t)
            auc = roc_auc_score(labels, scores)
            ci_lo, ci_hi = _bootstrap_auc_ci(labels, scores)
            grid_auc[i, j] = auc
            grid_ci[(i, j)] = (ci_lo, ci_hi)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid_auc, vmin=0.85, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="ROC AUC")

    ax.set_xticks(range(len(domain_labels)))
    ax.set_yticks(range(len(model_labels)))
    ax.set_xticklabels(domain_labels, fontsize=13)
    ax.set_yticklabels(model_labels,  fontsize=13)
    ax.set_xlabel("Domain",    fontsize=13)
    ax.set_ylabel("Model",     fontsize=13)
    ax.set_title("Cross-Model × Cross-Domain AUC\n(Epistemic Rigidity Detection, Layer 16)",
                 fontsize=13)

    for i in range(len(model_tags)):
        for j in range(len(domain_tags)):
            v = grid_auc[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=12)
            else:
                ci = grid_ci.get((i, j), (np.nan, np.nan))
                ax.text(j, i,
                        f"{v:.4f}\n[{ci[0]:.3f}, {ci[1]:.3f}]",
                        ha="center", va="center", fontsize=11, fontweight="bold")

    out = os.path.join(out_dir, "fig2_cross_model_domain_auc.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")
    return {"grid_auc": grid_auc.tolist()}


# ===========================================================================
# Figure 3: Noise Control — Exp B-2
# ===========================================================================

def fig_noise_control(data_files, out_dir, layer_key="Layer_16"):
    """
    [Fig 3] Displacement norm comparison: ||v_can|| vs ||v_noise||.
    Proves: S_l = ||v_noise|| / ||v_can|| << 1
    Rebuts: 'Model is just responding to any prefix.'
    """
    print("[Fig 3] Noise Control (S_l = ||v_noise|| / ||v_can||)...")

    r_can_all, r_noise_all, s_l_all = [], [], []

    for fpath in data_files:
        records = _load_json(fpath)
        for r in records:
            ls = r.get("layer_scans", {})
            if layer_key not in ls:
                continue
            for cond in ["Correct", "TypeC"]:
                mag = ls[layer_key][cond]["magnitudes_normalized"]
                r_can_all.append(  mag["R_can_rel"])
                r_noise_all.append(mag["R_noise_rel"])
                s_l = mag["R_noise_rel"] / (mag["R_can_rel"] + 1e-8)
                s_l_all.append(s_l)

    r_can_all   = np.array(r_can_all)
    r_noise_all = np.array(r_noise_all)
    s_l_all     = np.array(s_l_all)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(r_can_all, r_noise_all, alpha=0.5, color=BLUE, s=20)
    lim = max(r_can_all.max(), r_noise_all.max()) * 1.05
    ax1.plot([0, lim], [0, lim], "k--", lw=1, label="y = x (equal response)")
    ax1.set_xlabel(r"$\|v_{can}\|$ / $\|h_{base}\|$ (Canonical displacement)")
    ax1.set_ylabel(r"$\|v_{noise}\|$ / $\|h_{base}\|$ (Noise displacement)")
    ax1.set_title("Displacement Norm: Semantic vs. Noise")
    ax1.legend(fontsize=10)

    ax2.hist(s_l_all, bins=30, color=BLUE, edgecolor="black", alpha=0.75)
    ax2.axvline(np.mean(s_l_all), color=ROSE, lw=2,
                label=f"Mean S_l = {np.mean(s_l_all):.3f}")
    ax2.axvline(1.0, color="gray", lw=1.5, linestyle="--", label="S_l = 1.0 (null)")
    ax2.set_xlabel(r"$S_l = \|v_{noise}\| / \|v_{can}\|$")
    ax2.set_ylabel("Count")
    ax2.set_title(r"Stiffness Coefficient $S_l$ Distribution")
    ax2.legend(fontsize=10)

    fig.suptitle("Exp B-2: Noise Control Validation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "fig3_noise_control.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")
    return {"mean_S_l": float(np.mean(s_l_all)), "std_S_l": float(np.std(s_l_all, ddof=1))}


# ===========================================================================
# Figure 4: Layer-wise Ablation — Exp C
# ===========================================================================

def fig_layer_ablation(data_dir, out_dir):
    """
    [Fig 4] AUC per layer — inverse-U curve.
    Answers: 'Why Layer 16?' with empirical evidence.
    Generates per-model curves on the same axes.
    """
    print("[Fig 4] Layer-wise Ablation...")

    model_tags   = ["Meta-Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.2"]
    model_labels = ["Llama-3-8B", "Mistral-7B"]
    colors_m     = [NAVY, ROSE]

    fig, ax = plt.subplots(figsize=(10, 5))
    summary = {}

    for m, label, color in zip(model_tags, model_labels, colors_m):
        fpath = os.path.join(data_dir, f"expC_{m}_AllLayers_k1.json")
        if not os.path.exists(fpath):
            print(f"  [Warning] Not found: {fpath}")
            continue
        records = _load_json(fpath)

        layer_nums = sorted(
            set(int(k.split("_")[1]) for r in records for k in r["layer_scans"])
        )
        layer_aucs = {}
        for ln in layer_nums:
            lk = f"Layer_{ln}"
            s, lb = [], []
            for r in records:
                if lk not in r["layer_scans"]:
                    continue
                s.append(r["layer_scans"][lk]["Correct"]["coefficients"]["Attractor_Absorption_A_l"])
                lb.append(0)
                s.append(r["layer_scans"][lk]["TypeC"  ]["coefficients"]["Attractor_Absorption_A_l"])
                lb.append(1)
            if len(set(lb)) > 1:
                layer_aucs[ln] = roc_auc_score(lb, s)

        x = np.array(sorted(layer_aucs))
        y = np.array([layer_aucs[l] for l in x])

        ax.scatter(x, y, color=color, s=60, zorder=5)
        if len(x) > 3:
            x_new = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, y, k=min(3, len(x)-1))
            ax.plot(x_new, spl(x_new), color=color, lw=2, alpha=0.6, linestyle="--")
        ax.plot(x, y, color=color, lw=2, label=label, marker="o", markersize=7)

        best = max(layer_aucs, key=layer_aucs.get)
        ax.axvline(best, color=color, lw=1.2, linestyle="-.", alpha=0.6,
                   label=f"Critical Depth L{best} (AUC={layer_aucs[best]:.3f})")
        summary[label] = {"best_layer": int(best), "best_auc": float(layer_aucs[best])}

    ax.axhline(0.5, color="gray", linestyle=":", lw=1.2, label="Random (AUC=0.5)")
    ax.set_xlabel("Transformer Layer Depth")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Exp C: Layer-wise Controllability Profiling\n"
                 "(Epistemic Critical Depth — Inverse-U Curve)")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=10)

    out = os.path.join(out_dir, "fig4_layer_ablation.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")
    return summary


# ===========================================================================
# Figure 5: Temperature Sweep — Exp D
# ===========================================================================

def fig_temperature_sweep(data_dir, out_dir, layer_key="Layer_16"):
    """
    [Fig 5] A_l distribution across T=0.1 / 0.7 / 1.0.
    Proves: Rigidity is temperature-invariant (structural, not decoding artifact).
    """
    print("[Fig 5] Temperature Sweep...")

    model_tags   = ["Meta-Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.2"]
    model_labels = ["Llama-3-8B", "Mistral-7B"]
    temperatures = [0.1, 0.7, 1.0]

    fig, axes = plt.subplots(1, len(model_tags), figsize=(14, 5), sharey=True)
    summary = {}

    for ax, m, label in zip(axes, model_tags, model_labels):
        temp_aucs = {}
        for temp in temperatures:
            fpath = os.path.join(data_dir, f"expD_{m}_T{temp}_L16_k1.json")
            if not os.path.exists(fpath):
                print(f"  [Warning] Not found: {fpath}")
                continue
            records = _load_json(fpath)
            c, t = _extract_a_l(records, layer_key)
            if len(c) == 0:
                continue
            labels = [0]*len(c) + [1]*len(t)
            scores = list(c) + list(t)
            auc = roc_auc_score(labels, scores)
            temp_aucs[temp] = auc

            bp = ax.boxplot(
                [c, t],
                positions=[temp - 0.12, temp + 0.12],
                widths=0.1,
                patch_artist=True,
                medianprops=dict(color="black", lw=2),
            )
            bp["boxes"][0].set_facecolor(EMERALD)
            bp["boxes"][1].set_facecolor(ROSE)

        for temp, auc in temp_aucs.items():
            ax.text(temp, ax.get_ylim()[1] if ax.get_ylim()[1] > -999 else 1.05,
                    f"AUC={auc:.3f}", ha="center", fontsize=9, color=NAVY)

        ax.set_xticks(temperatures)
        ax.set_xticklabels([f"T={t}" for t in temperatures])
        ax.set_xlabel("Temperature")
        ax.set_title(label)
        summary[label] = temp_aucs

    axes[0].set_ylabel(r"$A_l$ (Attractor Absorption Coefficient)")
    fig.suptitle("Exp D: Temperature Sweep — Rigidity Invariance\n"
                 "Green=Correct, Red=TypeC", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "fig5_temperature_sweep.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")
    return summary


# ===========================================================================
# Figure 6: ROC curves — all model-domain pairs on one plot
# ===========================================================================

def fig_roc_all_pairs(data_dir, out_dir, layer_key="Layer_16"):
    """
    [Fig 6] Overlay ROC curves for all 4 model-domain pairs.
    Visual anchor for AUC > 0.96 across all pairs claim.
    """
    print("[Fig 6] ROC curves (all pairs)...")

    model_tags    = ["Meta-Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.2"]
    model_labels  = ["Llama-3-8B", "Mistral-7B"]
    domain_tags   = ["Legal", "Biomedical"]
    line_styles   = ["-", "--"]
    pair_colors   = [NAVY, BLUE, ROSE, EMERALD]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Random")

    ci = 0
    for i, (m, ml) in enumerate(zip(model_tags, model_labels)):
        fpath = os.path.join(data_dir, f"expA_{m}_L16_k1.json")
        if not os.path.exists(fpath):
            continue
        records = _load_json(fpath)
        for j, domain in enumerate(domain_tags):
            domain_records = [r for r in records if r["domain"] == domain]
            c, t = _extract_a_l(domain_records, layer_key)
            if len(c) == 0:
                continue
            labels = [0]*len(c) + [1]*len(t)
            scores = list(c) + list(t)
            auc  = roc_auc_score(labels, scores)
            fpr, tpr, _ = roc_curve(labels, scores)
            ax.plot(fpr, tpr,
                    color=pair_colors[ci % len(pair_colors)],
                    lw=2, linestyle=line_styles[j],
                    label=f"{ml} × {domain} (AUC={auc:.4f})")
            ci += 1

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC: Epistemic Rigidity Detection\n(All model-domain pairs, Layer 16)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)

    out = os.path.join(out_dir, "fig6_roc_all_pairs.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


# ===========================================================================
# Statistical Summary JSON
# ===========================================================================

def export_stats_summary(stats_dict, out_dir):
    path = os.path.join(out_dir, "phase6_stats_summary.json")
    with open(path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"\n[Stats] Summary saved: {path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 6 Analyzer — AIES 2026")
    parser.add_argument("--data_dir", required=True,  help="Directory with RunPod JSON results")
    parser.add_argument("--out_dir",  required=True,  help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Collect Exp A result files (main cross-model/domain results)
    exp_a_files = [
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir)
        if f.startswith("expA_") and f.endswith(".json")
    ]
    if not exp_a_files:
        print("[Warning] No expA_*.json found. Skipping figures that require Exp A data.")

    stats_all = {}

    if exp_a_files:
        stats_all["fig1_variance_collapse"]   = fig_variance_collapse(exp_a_files, args.out_dir)
        stats_all["fig2_cross_model_domain"]  = fig_cross_model_domain(args.data_dir, args.out_dir)
        stats_all["fig3_noise_control"]       = fig_noise_control(exp_a_files, args.out_dir)
        fig_roc_all_pairs(args.data_dir, args.out_dir)

    exp_c_exists = any(
        f.startswith("expC_") for f in os.listdir(args.data_dir)
    )
    if exp_c_exists:
        stats_all["fig4_layer_ablation"] = fig_layer_ablation(args.data_dir, args.out_dir)

    exp_d_exists = any(
        f.startswith("expD_") for f in os.listdir(args.data_dir)
    )
    if exp_d_exists:
        stats_all["fig5_temperature_sweep"] = fig_temperature_sweep(args.data_dir, args.out_dir)

    export_stats_summary(stats_all, args.out_dir)
    print("\n[Complete] All figures generated.")


if __name__ == "__main__":
    main()
