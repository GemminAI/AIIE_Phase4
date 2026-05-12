"""
AIIE Phase 4 Experiment 5.2
Deceptive Adversarial: 存在しない法令引用による欺瞞的ハルシネーション

目的:
  LogProbが高止まりする「確信度の高い嘘」において
  κ_t早期スパイクがDeceptive Adversarialを検知できるかを検証

仮説:
  - LogProb: Deceptive下でAUC低下（検知失敗）
  - κ_t Deformation (1-s_κ): AUC維持または向上（早期スパイクが支配的）
  - Grounded Score 2.0: AUC最高（β項が支配的検知因子）

出力:
  - fig19_deceptive_roc.png      : ROC曲線（Easy vs Deceptive比較）
  - fig20_spike_position.png     : スパイク位置分布（Deceptive vs Benign）
  - exp52_results.json           : 定量結果
"""

import json
import pickle
from pathlib import Path
from typing import List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

from curvature import compute_discrete_curvature
from domain_loader import load_deceptive_legal_citations
from phase4_config import Phase4Config
from phase4_extractor import extract_hidden_states_for_response, load_model

SIGMA_CODE_LEGAL  = 1.1622
ALPHA = 1.0
BETA  = 1.0
GAMMA = 0.5
DYNAMIC_THRESHOLD = 0.04302
LOGPROB_THRESHOLD = -2.0


def compute_lead_time_online(kappas, logprobs):
    lp_mean = np.mean(logprobs)
    lp_std  = np.std(logprobs)
    kappa_step, logprob_step = -1, -1
    for t, k in enumerate(kappas):
        if k > DYNAMIC_THRESHOLD:
            kappa_step = t
            break
    for t, lp in enumerate(logprobs):
        if lp < lp_mean - 2 * lp_std:
            logprob_step = t
            break
    if kappa_step >= 0 and logprob_step >= 0:
        return logprob_step - kappa_step
    return 0


def extract_scores_v2(kappas_arr, logprobs_arr, lead_time=None):
    logprob_min   = float(np.min(logprobs_arr)) if len(logprobs_arr) > 0 else 0.0
    score_logprob = -logprob_min

    T         = len(kappas_arr)
    spike_pos = float(np.argmax(kappas_arr)) / (T + 1e-8) if T > 0 else 0.5
    score_deformation = 1.0 - spike_pos

    k = float(lead_time) if lead_time is not None else compute_lead_time_online(kappas_arr, logprobs_arr)
    score_leadtime = k

    grounded = ALPHA * score_logprob + BETA * score_deformation + GAMMA * score_leadtime

    return {
        "logprob_min":    score_logprob,
        "kappa_deform":   score_deformation,
        "grounded_v2":    grounded,
        "spike_pos_raw":  spike_pos,
        "logprob_raw":    logprob_min,
    }


def extract_deceptive_records(model, tokenizer, pairs, target_layer, device, max_length):
    """Deceptive側とBenign側を両方抽出"""
    deceptive_records = []
    benign_records    = []

    print(f"[Extracting Deceptive + Benign] {len(pairs)} pairs...")
    for i, pair in enumerate(pairs):
        try:
            # Deceptive側
            dec_states = extract_hidden_states_for_response(
                model, tokenizer,
                prompt=pair["prompt"],
                response=pair["response"],
                target_layer=target_layer,
                device=device,
                max_length=max_length,
            )
            # Benign側
            ben_states = extract_hidden_states_for_response(
                model, tokenizer,
                prompt=pair["benign_prompt"],
                response=pair["benign_response"],
                target_layer=target_layer,
                device=device,
                max_length=max_length,
            )

            d_traj = dec_states["hidden"].get(target_layer)
            b_traj = ben_states["hidden"].get(target_layer)

            if d_traj is None or b_traj is None or len(d_traj) < 3 or len(b_traj) < 3:
                continue

            d_kappas   = compute_discrete_curvature(d_traj)
            b_kappas   = compute_discrete_curvature(b_traj)
            d_logprobs = np.array(dec_states["logprobs"])
            b_logprobs = np.array(ben_states["logprobs"])

            d_lt = compute_lead_time_online(d_kappas, d_logprobs)
            b_lt = compute_lead_time_online(b_kappas, b_logprobs)

            deceptive_records.append({
                "pair_idx":       i,
                "deceptive_type": pair["deceptive_type"],
                "kappas":         d_kappas.tolist(),
                "logprobs":       d_logprobs.tolist(),
                "lead_time_k":    d_lt,
                "label":          1,
            })
            benign_records.append({
                "pair_idx":    i,
                "kappas":      b_kappas.tolist(),
                "logprobs":    b_logprobs.tolist(),
                "lead_time_k": b_lt,
                "label":       0,
            })

            if (i + 1) % 10 == 0:
                d_sp = np.argmax(d_kappas) / (len(d_kappas) + 1e-8)
                b_sp = np.argmax(b_kappas) / (len(b_kappas) + 1e-8)
                print(f"  [{i+1}/{len(pairs)}] dec_spike={d_sp:.3f} ben_spike={b_sp:.3f} lead={d_lt}")

        except Exception as e:
            import traceback
            print(f"  [Error] pair {i}: {e}")
            traceback.print_exc()
            continue

    print(f"[Done] deceptive={len(deceptive_records)}, benign={len(benign_records)}")
    return deceptive_records, benign_records


def compute_roc_results(deceptive_records, benign_records):
    all_records = (
        [(r, 1) for r in deceptive_records] +
        [(r, 0) for r in benign_records]
    )
    labels         = []
    lp_scores      = []
    kappa_scores   = []
    grounded_scores = []

    for r, label in all_records:
        k_arr  = np.array(r["kappas"])
        lp_arr = np.array(r["logprobs"])
        sc = extract_scores_v2(k_arr, lp_arr, r.get("lead_time_k"))
        labels.append(label)
        lp_scores.append(sc["logprob_min"])
        kappa_scores.append(sc["kappa_deform"])
        grounded_scores.append(sc["grounded_v2"])

    labels = np.array(labels)
    results = {}
    for name, scores in [
        ("LogProb_min",  np.array(lp_scores)),
        ("kappa_deform", np.array(kappa_scores)),
        ("Grounded_v2",  np.array(grounded_scores)),
    ]:
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_score   = auc(fpr, tpr)
        results[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc_score)}
        print(f"  {name}: AUC = {auc_score:.4f}")
    return results


def plot_deceptive_roc(roc_results, output_path, n_dec, n_ben):
    fig, ax = plt.subplots(figsize=(8, 7))
    styles = {
        "LogProb_min":  {"color": "#9E9E9E", "ls": "--", "lw": 2.0, "label": "Baseline (LogProb)"},
        "kappa_deform": {"color": "#2196F3", "ls": "-.", "lw": 2.0, "label": "Geometry (1-s_κ)"},
        "Grounded_v2":  {"color": "#E91E63", "ls": "-",  "lw": 2.5, "label": "Grounded Score 2.0"},
    }
    for name, st in styles.items():
        r = roc_results[name]
        ax.plot(r["fpr"], r["tpr"], color=st["color"], linestyle=st["ls"],
                linewidth=st["lw"], label=f"{st['label']} (AUC={r['auc']:.3f})")
    ax.plot([0,1],[0,1], "k:", lw=1, label="Random (AUC=0.500)")

    lp_auc = roc_results["LogProb_min"]["auc"]
    gr_auc = roc_results["Grounded_v2"]["auc"]
    delta  = gr_auc - lp_auc
    sign   = "+" if delta >= 0 else ""
    ax.text(0.52, 0.12,
            f"Grounded vs LogProb:\n{sign}{delta:.3f} AUC\n"
            f"({'β-term dominant' if roc_results['kappa_deform']['auc'] > lp_auc else 'combined effect'})",
            transform=ax.transAxes, fontsize=10, color="#E91E63",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E91E63"))

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"Figure 19: ROC — Deceptive Adversarial (Fake Legal Citations)\n"
        f"n_deceptive={n_dec}, n_benign={n_ben}",
        fontsize=12, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure saved] {output_path}")


def plot_spike_position(deceptive_records, benign_records, output_path):
    d_sp = [np.argmax(r["kappas"]) / (len(r["kappas"]) + 1e-8) for r in deceptive_records]
    b_sp = [np.argmax(r["kappas"]) / (len(r["kappas"]) + 1e-8) for r in benign_records]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 20)
    ax.hist(b_sp, bins=bins, alpha=0.6, color="#2196F3", label=f"Benign (n={len(b_sp)})")
    ax.hist(d_sp, bins=bins, alpha=0.6, color="#F44336", label=f"Deceptive (n={len(d_sp)})")
    ax.axvline(np.mean(d_sp), color="#F44336", linestyle="--", lw=2,
               label=f"Deceptive mean={np.mean(d_sp):.3f}")
    ax.axvline(np.mean(b_sp), color="#2196F3", linestyle="--", lw=2,
               label=f"Benign mean={np.mean(b_sp):.3f}")
    ax.set_xlabel("κ_t Spike Position (normalized, lower=earlier)")
    ax.set_ylabel("Count")
    ax.set_title(
        "Figure 20: κ_t Spike Position Distribution\n"
        "Deceptive Adversarial vs Benign Reasoning"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure saved] {output_path}")


def run_exp52():
    cfg        = Phase4Config()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_layer = getattr(cfg, "target_layer", 7)
    max_length   = getattr(cfg, "max_length", 512)
    device       = getattr(cfg, "device", "cuda")

    print("=" * 70)
    print("AIIE Phase 4 Experiment 5.2: Deceptive Adversarial")
    print("存在しない法令引用による欺瞞的ハルシネーション検知")
    print("=" * 70)

    pairs = load_deceptive_legal_citations(cfg, n_pairs=99)
    model, tokenizer = load_model(cfg)

    deceptive_records, benign_records = extract_deceptive_records(
        model, tokenizer, pairs, target_layer, device, max_length
    )

    # PKL保存
    with open(output_dir / "exp52_deceptive_records.pkl", "wb") as f:
        pickle.dump(deceptive_records, f)
    with open(output_dir / "exp52_benign_records.pkl", "wb") as f:
        pickle.dump(benign_records, f)

    print("\n[Computing ROC/AUC...]")
    roc_results = compute_roc_results(deceptive_records, benign_records)

    # 統計
    d_sp = [np.argmax(r["kappas"]) / (len(r["kappas"]) + 1e-8) for r in deceptive_records]
    b_sp = [np.argmax(r["kappas"]) / (len(r["kappas"]) + 1e-8) for r in benign_records]
    d_lp = [min(r["logprobs"]) for r in deceptive_records]
    b_lp = [min(r["logprobs"]) for r in benign_records]

    summary = {
        "n_deceptive":          len(deceptive_records),
        "n_benign":             len(benign_records),
        "auc_scores":           {k: v["auc"] for k, v in roc_results.items()},
        "spike_pos": {
            "deceptive_mean":   float(np.mean(d_sp)),
            "benign_mean":      float(np.mean(b_sp)),
            "separation":       float(np.mean(b_sp) - np.mean(d_sp)),
        },
        "logprob_min": {
            "deceptive_mean":   float(np.mean(d_lp)),
            "benign_mean":      float(np.mean(b_lp)),
            "separation":       float(np.mean(b_lp) - np.mean(d_lp)),
        },
    }

    print("\n" + "=" * 70)
    print("【Experiment 5.2 Results】")
    for name, r in roc_results.items():
        print(f"  {name}: AUC = {r['auc']:.4f}")
    print(f"  spike_pos separation: {summary['spike_pos']['separation']:+.4f}"
          f" (deceptive={summary['spike_pos']['deceptive_mean']:.3f}, "
          f"benign={summary['spike_pos']['benign_mean']:.3f})")
    print(f"  logprob separation:   {summary['logprob_min']['separation']:+.4f}"
          f" (deceptive={summary['logprob_min']['deceptive_mean']:.3f}, "
          f"benign={summary['logprob_min']['benign_mean']:.3f})")
    print("=" * 70)

    with open(output_dir / "exp52_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n→ {output_dir / 'exp52_results.json'}")

    plot_deceptive_roc(roc_results, str(output_dir / "fig19_deceptive_roc.png"),
                       len(deceptive_records), len(benign_records))
    plot_spike_position(deceptive_records, benign_records,
                        str(output_dir / "fig20_spike_position.png"))

    print("\n[Exp 5.2 Complete]")
    return summary


if __name__ == "__main__":
    run_exp52()
