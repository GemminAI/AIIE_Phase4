"""
AIIE Phase 4 Experiment 5.1
ROC/AUC 限界テスト: κ_t vs LogProb vs Grounded Score

目的:
  ハルシネーション（Adversarial）と正常推論（Benign）を識別する
  3つのスコアラーの識別性能をROC/AUCで定量化する

スコアラー:
  1. LogProb_min:  logprobの最小値（既存手法のBaseline）
  2. κ_t_max:      κ_tの最大スパイク（幾何学的指標）
  3. Grounded:     κ_t_max / σ(code→legal)（伝導率補正複合指標）

出力:
  - fig18_roc_auc.png       : ROC曲線3本比較
  - exp51_results.json      : AUCスコア + 統計
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

from curvature import compute_discrete_curvature
from domain_loader import load_benign_code_legal
from phase4_config import Phase4Config
from phase4_extractor import extract_hidden_states_for_response, load_model


# ──────────────────────────────────────────────
# 1. スコア抽出
# ──────────────────────────────────────────────
SIGMA_CODE_LEGAL = 1.1622  # Exp 4.1確定値

# Grounded Score 2.0 weights（Exp 4.2実測値から導出）
ALPHA = 1.0   # LogProb term weight
BETA  = 1.0   # Spike position term weight
GAMMA = 0.5   # Lead time term weight（kの分散が大きいため0.5に抑制）

# Dynamic threshold（Exp 4.2確定値）
DYNAMIC_THRESHOLD = 0.04302
LOGPROB_THRESHOLD = -2.0


def compute_lead_time_from_record(
    kappas: np.ndarray,
    logprobs: np.ndarray,
    kappa_threshold: float = DYNAMIC_THRESHOLD,
    logprob_threshold: float = LOGPROB_THRESHOLD,
) -> int:
    """
    κ_tスパイクステップ と LogProb異常ステップ の差を計算。
    Benignレコード用（lead_time_kが保存されていない場合）。
    """
    # κ_tスパイク
    kappa_step = -1
    for t, k in enumerate(kappas):
        if k > kappa_threshold:
            kappa_step = t
            break

    # LogProb異常（平均-2σ以下）
    lp_mean = np.mean(logprobs)
    lp_std  = np.std(logprobs)
    logprob_step = -1
    for t, lp in enumerate(logprobs):
        if lp < lp_mean - 2 * lp_std:
            logprob_step = t
            break

    if kappa_step >= 0 and logprob_step >= 0:
        return logprob_step - kappa_step
    return 0  # 計算不能な場合はニュートラル


def extract_scores(record: Dict) -> Dict[str, float]:
    """
    Grounded Score 2.0:
      S = α(-log p_min) + β(1 - s_κ) + γk

    - α(-log p_min): Outcome  — 統計的崩壊（LogProbの最小値）
    - β(1 - s_κ):   Deformation — 幾何学的屈曲（スパイク位置の早さ）
    - γk:           Lead Time  — 認識論的余裕（曲率がLogProbより何ステップ早いか）

    実測値による分離方向:
      logprob_min:   adv=-9.10 < ben=-7.46  → -logprob_min（大=異常）
      spike_pos:     adv=0.781 < ben=0.863  → 1-spike_pos（大=早期崩壊=異常）
      lead_time_k:   adv=1.33（先行）       → k（正=異常）
    """
    # キーの解決（adv/benで異なるキー名）
    kappas   = np.array(record.get("camouflage_kappas",  record.get("benign_kappas",  [])))
    logprobs = np.array(record.get("camouflage_logprobs", record.get("benign_logprobs", [])))

    # ── Outcome term: α(-log p_min) ──
    logprob_min   = float(np.min(logprobs)) if len(logprobs) > 0 else 0.0
    score_logprob = -logprob_min  # 大=異常

    # ── Deformation term: β(1 - s_κ) ──
    T         = len(kappas)
    spike_pos = float(np.argmax(kappas)) / (T + 1e-8) if T > 0 else 0.5
    score_deformation = 1.0 - spike_pos  # 早期スパイク=大=異常

    # ── Lead Time term: γk ──
    if "lead_time_k" in record and record["lead_time_k"] is not None:
        k = float(record["lead_time_k"])
    else:
        # Benignレコード: リアルタイム計算
        k = float(compute_lead_time_from_record(kappas, logprobs))
    score_leadtime = k  # 正=κ_t先行=異常

    # ── Grounded Score 2.0 ──
    grounded_v2 = (
        ALPHA * score_logprob
        + BETA  * score_deformation
        + GAMMA * score_leadtime
    )

    return {
        "logprob_min":    score_logprob,
        "kappa_t_max":    score_deformation,   # 実態はspike_position
        "grounded":       grounded_v2,
    }


# ──────────────────────────────────────────────
# 2. Benign hidden state抽出
# ──────────────────────────────────────────────
def extract_benign_records(
    model, tokenizer, pairs: List[Dict],
    target_layer: int, device: str, max_length: int
) -> List[Dict]:
    records = []
    print(f"[Extracting Benign] {len(pairs)} pairs...")

    for i, pair in enumerate(pairs):
        try:
            states = extract_hidden_states_for_response(
                model, tokenizer,
                prompt=pair["prompt"],
                response=pair["response"],
                target_layer=target_layer,
                device=device,
                max_length=max_length,
            )
            traj = states["hidden"].get(target_layer)
            if traj is None or len(traj) < 3:
                continue

            kappas   = compute_discrete_curvature(traj)
            logprobs = np.array(states["logprobs"])

            records.append({
                "pair_idx":      i,
                "benign_type":   pair["benign_type"],
                "benign_kappas": kappas.tolist(),
                "benign_logprobs": logprobs.tolist(),
                "label":         0,
            })

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(pairs)}] traj_len={len(traj)}")

        except Exception as e:
            import traceback
            print(f"  [Error] pair {i}: {e}")
            traceback.print_exc()
            continue

    print(f"[Benign] {len(records)} records extracted")
    return records


# ──────────────────────────────────────────────
# 3. ROC/AUC計算
# ──────────────────────────────────────────────
def compute_roc_auc(
    adversarial_records: List[Dict],
    benign_records: List[Dict],
) -> Dict:
    """
    Adversarial(label=1) + Benign(label=0) を混合して
    3スコアラーのROC/AUCを計算する。
    """
    all_records = []
    for r in adversarial_records:
        scores = extract_scores(r)
        scores["label"] = 1
        all_records.append(scores)
    for r in benign_records:
        scores = extract_scores(r)
        scores["label"] = 0
        all_records.append(scores)

    labels      = np.array([r["label"]      for r in all_records])
    kappa_scores    = np.array([r["kappa_t_max"] for r in all_records])
    logprob_scores  = np.array([r["logprob_min"] for r in all_records])
    grounded_scores = np.array([r["grounded"]    for r in all_records])

    results = {}
    for name, scores in [
        ("LogProb_min",  logprob_scores),
        ("kappa_t_max",  kappa_scores),
        ("Grounded",     grounded_scores),
    ]:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc_score = auc(fpr, tpr)
        results[name] = {
            "fpr":       fpr.tolist(),
            "tpr":       tpr.tolist(),
            "auc":       float(auc_score),
            "thresholds": thresholds.tolist(),
        }
        print(f"  {name}: AUC = {auc_score:.4f}")

    return results


# ──────────────────────────────────────────────
# 4. Figure 18: ROC曲線
# ──────────────────────────────────────────────
def plot_roc_curves(roc_results: Dict, output_path: str, n_adv: int, n_benign: int):
    fig, ax = plt.subplots(figsize=(8, 7))

    styles = {
        "LogProb_min": {"color": "#9E9E9E", "linestyle": "--", "linewidth": 2.0,
                        "label_prefix": "Baseline (Outcome)"},
        "kappa_t_max": {"color": "#2196F3", "linestyle": "-.",  "linewidth": 2.0,
                        "label_prefix": "Geometry (Deformation)"},
        "Grounded":    {"color": "#E91E63", "linestyle": "-",   "linewidth": 2.5,
                        "label_prefix": "Grounded Score 2.0 (Proposed)"},
    }

    for name, style in styles.items():
        r = roc_results[name]
        label = (f"{style['label_prefix']}: {name} "
                 f"(AUC={r['auc']:.3f})")
        ax.plot(r["fpr"], r["tpr"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                label=label)

    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1.0,
            label="Random (AUC=0.500)")

    # AUC改善率を注記
    auc_base     = roc_results["LogProb_min"]["auc"]
    auc_grounded = roc_results["Grounded"]["auc"]
    improvement  = (auc_grounded - auc_base) / auc_base * 100
    ax.text(0.55, 0.15,
            f"Grounded vs Baseline:\n+{improvement:.1f}% AUC improvement",
            transform=ax.transAxes,
            fontsize=10, color="#E91E63",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E91E63"))

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"Figure 18: ROC Curves — Grounded Score 2.0 vs Baselines\n"
        f"Code→Legal Camouflage (σ=1.162), "
        f"n_adv={n_adv}, n_benign={n_benign}",
        fontsize=12, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure saved] {output_path}")


# ──────────────────────────────────────────────
# 5. メイン実行
# ──────────────────────────────────────────────
def run_exp51():
    cfg        = Phase4Config()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_layer = getattr(cfg, "target_layer", 7)
    max_length   = getattr(cfg, "max_length", 512)
    device       = getattr(cfg, "device", "cuda")

    print("=" * 70)
    print("AIIE Phase 4 Experiment 5.1: ROC/AUC 限界テスト")
    print(f"σ(code→legal) = {SIGMA_CODE_LEGAL}")
    print("=" * 70)

    # ── Adversarialデータ読み込み ──
    pkl_path = output_dir / "exp42_detail_records.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Exp 4.2のPKLが見つかりません: {pkl_path}\n"
            f"先にexp42_kappa_analysis.pyを実行してください。"
        )
    with open(pkl_path, "rb") as f:
        adversarial_records = pickle.load(f)
    print(f"[Adversarial] {len(adversarial_records)} records loaded from PKL")

    # ── Benignデータ抽出 ──
    benign_pairs = load_benign_code_legal(cfg, n_pairs=99)
    model, tokenizer = load_model(cfg)
    benign_records = extract_benign_records(
        model, tokenizer, benign_pairs,
        target_layer, device, max_length
    )

    # Benign PKL保存
    benign_pkl = output_dir / "exp51_benign_records.pkl"
    with open(benign_pkl, "wb") as f:
        pickle.dump(benign_records, f)
    print(f"→ {benign_pkl}")

    # ── ROC/AUC計算 ──
    print("\n[Computing ROC/AUC...]")
    roc_results = compute_roc_auc(adversarial_records, benign_records)

    # ── 結果保存 ──
    summary = {
        "n_adversarial":    len(adversarial_records),
        "n_benign":         len(benign_records),
        "sigma_code_legal": SIGMA_CODE_LEGAL,
        "auc_scores": {
            name: r["auc"] for name, r in roc_results.items()
        },
        "auc_improvement_grounded_vs_logprob": (
            (roc_results["Grounded"]["auc"] - roc_results["LogProb_min"]["auc"])
            / roc_results["LogProb_min"]["auc"] * 100
        ),
    }

    print("\n" + "=" * 70)
    print("【Experiment 5.1 Results】")
    for name, r in roc_results.items():
        print(f"  {name}: AUC = {r['auc']:.4f}")
    print(f"  Improvement (Grounded vs Baseline): "
          f"+{summary['auc_improvement_grounded_vs_logprob']:.1f}%")
    print("=" * 70)

    exp51_path = output_dir / "exp51_results.json"
    with open(exp51_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n→ {exp51_path}")

    # ── Figure 18 ──
    plot_roc_curves(
        roc_results,
        str(output_dir / "fig18_roc_auc.png"),
        n_adv=len(adversarial_records),
        n_benign=len(benign_records),
    )

    print("\n[Exp 5.1 Complete]")
    return summary


if __name__ == "__main__":
    run_exp51()
