"""
AIIE Phase 4 Experiment 4.2
Topological Camouflage 限界テスト: κ_t vs LogProb 感度分析

目的:
  code→legal カモフラージュ経路(σ=1.1622)において
  κ_t がハルシネーションを LogProb より早く検知できるかを検証

出力:
  - fig15_kappa_vs_logprob.png : κ_t軌道 vs LogProb比較
  - fig16_lead_time.png        : リードタイムk分布
  - fig17_threshold_calibration.png : RFC-0041動的閾値最適化
  - exp42_results.json         : 定量結果
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from curvature import compute_discrete_curvature, compute_curvature_features
from domain_loader import load_adversarial_code_legal
from phase4_config import Phase4Config
from phase4_extractor import extract_hidden_states_for_response, load_model


# ──────────────────────────────────────────────
# 1. κ_t スパイク検出
# ──────────────────────────────────────────────
def detect_kappa_spike(kappas: np.ndarray, threshold: float) -> int:
    """
    κ_t が threshold を超えた最初のステップを返す。
    超えない場合は -1。
    """
    for t, k in enumerate(kappas):
        if k > threshold:
            return t
    return -1


# ──────────────────────────────────────────────
# 2. RFC-0041 動的閾値計算
# ──────────────────────────────────────────────
def compute_dynamic_threshold(base_threshold: float, sigma: float) -> float:
    """
    THRESHOLD_PAUSE = Base_Threshold / σ(A→B)
    
    σ > 1（高伝導）→ 閾値を引き下げ → より敏感にPAUSEを発動
    """
    return base_threshold / sigma


# ──────────────────────────────────────────────
# 3. リードタイム計算
# ──────────────────────────────────────────────
def compute_lead_time(
    kappas: np.ndarray,
    logprobs: np.ndarray,
    kappa_threshold: float,
    logprob_threshold: float,
) -> Dict:
    """
    κ_t スパイク発生ステップ と LogProb異常ステップ の差（リードタイム k）を計算
    """
    kappa_step = detect_kappa_spike(kappas, kappa_threshold)
    
    # LogProbの異常: 平均から2σ以上の低下
    lp_mean = np.mean(logprobs)
    lp_std = np.std(logprobs)
    logprob_step = -1
    for t, lp in enumerate(logprobs):
        if lp < lp_mean - 2 * lp_std:
            logprob_step = t
            break
    
    lead_time = None
    if kappa_step >= 0 and logprob_step >= 0:
        lead_time = logprob_step - kappa_step  # 正 = κ_tが先に検知
    
    return {
        "kappa_spike_step": kappa_step,
        "logprob_anomaly_step": logprob_step,
        "lead_time_k": lead_time,
    }


def _pad(arr: np.ndarray, length: int) -> np.ndarray:
    return np.pad(arr, (0, length - len(arr)), constant_values=np.nan)


# ──────────────────────────────────────────────
# 4. Figure 15: κ_t軌道 vs LogProb
# ──────────────────────────────────────────────
def plot_kappa_vs_logprob(
    results: List[Dict],
    sigma: float,
    base_threshold: float,
    output_path: str,
):
    dynamic_threshold = compute_dynamic_threshold(base_threshold, sigma)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(
        f"Figure 15: κ_t vs LogProb — Code→Legal Camouflage (σ={sigma:.3f})",
        fontsize=13, fontweight="bold"
    )
    
    # 上段: κ_t軌道（全ペアの平均）
    ax1 = axes[0]
    all_kappas_correct = []
    all_kappas_camouflage = []
    
    for r in results:
        if r.get("correct_kappas") is not None and r.get("camouflage_kappas") is not None:
            all_kappas_correct.append(np.array(r["correct_kappas"]))
            all_kappas_camouflage.append(np.array(r["camouflage_kappas"]))
    
    if all_kappas_correct:
        max_len = max(len(k) for k in all_kappas_correct + all_kappas_camouflage)
        kc_matrix = np.array([_pad(k, max_len) for k in all_kappas_correct])
        km_matrix = np.array([_pad(k, max_len) for k in all_kappas_camouflage])
        
        mean_correct = np.nanmean(kc_matrix, axis=0)
        mean_camouflage = np.nanmean(km_matrix, axis=0)
        
        x = np.arange(max_len)
        ax1.plot(x, mean_correct, color="#2196F3", label="Correct reasoning", linewidth=1.5)
        ax1.plot(x, mean_camouflage, color="#F44336", label="Camouflage (hallucination)", linewidth=1.5)
        ax1.axhline(base_threshold, color="gray", linestyle="--", alpha=0.7, label=f"Base θ={base_threshold:.3f}")
        ax1.axhline(dynamic_threshold, color="#FF9800", linestyle="--", linewidth=2,
                    label=f"Dynamic θ={dynamic_threshold:.3f} (÷σ={sigma:.2f})")
        ax1.set_ylabel("κ_t (curvature)")
        ax1.set_title("Hidden State Curvature Trajectory (L7 average)")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
    
    # 下段: LogProb比較
    ax2 = axes[1]
    all_lp_correct = [np.array(r["correct_logprobs"]) for r in results if r.get("correct_logprobs") is not None]
    all_lp_camouflage = [np.array(r["camouflage_logprobs"]) for r in results if r.get("camouflage_logprobs") is not None]
    
    if all_lp_correct:
        max_len2 = max(len(l) for l in all_lp_correct + all_lp_camouflage)
        lpc_matrix = np.array([_pad(l, max_len2) for l in all_lp_correct])
        lpm_matrix = np.array([_pad(l, max_len2) for l in all_lp_camouflage])
        
        ax2.plot(np.nanmean(lpc_matrix, axis=0), color="#2196F3", label="Correct", linewidth=1.5)
        ax2.plot(np.nanmean(lpm_matrix, axis=0), color="#F44336", label="Camouflage", linewidth=1.5)
        ax2.set_ylabel("Log Probability")
        ax2.set_xlabel("Token step")
        ax2.set_title("LogProb Trajectory (camouflage maintains high confidence = detection failure)")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure saved] {output_path}")


# ──────────────────────────────────────────────
# 5. Figure 16: リードタイム分布
# ──────────────────────────────────────────────
def plot_lead_time_distribution(lead_times: List[int], output_path: str):
    valid = [lt for lt in lead_times if lt is not None]
    if not valid:
        print("[Warning] No valid lead times to plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid, bins=20, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="k=0 (simultaneous)")
    ax.axvline(np.mean(valid), color="#FF9800", linestyle="-", linewidth=2,
               label=f"Mean k={np.mean(valid):.1f}")
    ax.set_xlabel("Lead time k (steps κ_t precedes LogProb anomaly)")
    ax.set_ylabel("Count")
    ax.set_title("Figure 16: κ_t Lead Time Distribution\n(positive = κ_t detects earlier)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure saved] {output_path}")


# ──────────────────────────────────────────────
# 6. Figure 17: RFC-0041 動的閾値キャリブレーション
# ──────────────────────────────────────────────
def plot_threshold_calibration(
    sigma_values: List[float],
    base_threshold: float,
    output_path: str,
):
    sigmas = np.linspace(0.5, 1.5, 200)
    thresholds = base_threshold / sigmas
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sigmas, thresholds, color="#673AB7", linewidth=2.5)
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="σ=1.0 (no contagion)")
    
    colors = ["#F44336", "#FF9800", "#2196F3", "#4CAF50"]
    labels = ["code→legal\n(σ=1.162)", "medical→code\n(σ=1.023)",
              "medical→legal\n(σ=1.005)", "legal→medical\n(σ=0.990)"]
    
    for sv, color, label in zip(sigma_values, colors, labels):
        th = base_threshold / sv
        ax.scatter([sv], [th], color=color, s=120, zorder=5)
        ax.annotate(label, (sv, th), textcoords="offset points",
                    xytext=(8, 4), fontsize=8, color=color)
    
    ax.set_xlabel("σ(A→B) — Semantic Conductivity")
    ax.set_ylabel("THRESHOLD_PAUSE (dynamic)")
    ax.set_title(
        f"Figure 17: RFC-0041 Dynamic Threshold Calibration\n"
        f"THRESHOLD_PAUSE = {base_threshold:.3f} / σ(A→B)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure saved] {output_path}")


# ──────────────────────────────────────────────
# 7. メイン実行
# ──────────────────────────────────────────────
def run_exp42():
    cfg = Phase4Config()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    SIGMA_CODE_LEGAL = 1.1622  # Exp 4.1確定値
    BASE_THRESHOLD = 0.05      # κ_tベース閾値（Phase 1準拠）
    DYNAMIC_THRESHOLD = compute_dynamic_threshold(BASE_THRESHOLD, SIGMA_CODE_LEGAL)
    
    target_layer = getattr(cfg, "target_layer", 7)
    max_length = getattr(cfg, "max_length", 512)
    device = getattr(cfg, "device", "cuda")
    
    print("=" * 70)
    print("AIIE Phase 4 Experiment 4.2: Topological Camouflage 限界テスト")
    print(f"σ(code→legal) = {SIGMA_CODE_LEGAL}")
    print(f"Base threshold = {BASE_THRESHOLD:.4f}")
    print(f"Dynamic threshold = {DYNAMIC_THRESHOLD:.4f} (÷{SIGMA_CODE_LEGAL})")
    print("=" * 70)
    
    # データ生成
    pairs = load_adversarial_code_legal(cfg, n_pairs=99)
    
    # モデルロード
    model, tokenizer = load_model(cfg)
    
    # hidden state抽出 + κ_t計算
    results = []
    print(f"\n[Extracting] {len(pairs)} adversarial pairs...")
    
    for i, pair in enumerate(pairs):
        try:
            original_prompt = pair.get("original_prompt", pair["prompt"])
            
            # 正解推論
            correct_states = extract_hidden_states_for_response(
                model,
                tokenizer,
                prompt=original_prompt,
                response=pair["correct"],
                target_layer=target_layer,
                device=device,
                max_length=max_length,
            )
            # カモフラージュ推論
            camouflage_states = extract_hidden_states_for_response(
                model,
                tokenizer,
                prompt=pair["prompt"],
                response=pair["hallucinated"],
                target_layer=target_layer,
                device=device,
                max_length=max_length,
            )
            
            c_traj = correct_states["hidden"].get(target_layer)
            m_traj = camouflage_states["hidden"].get(target_layer)
            correct_logprobs = np.array(correct_states["logprobs"])
            camouflage_logprobs = np.array(camouflage_states["logprobs"])
            
            if c_traj is None or m_traj is None or len(c_traj) < 3 or len(m_traj) < 3:
                continue
            
            c_kappas = compute_discrete_curvature(c_traj)
            m_kappas = compute_discrete_curvature(m_traj)
            
            lead_info = compute_lead_time(
                m_kappas, camouflage_logprobs,
                kappa_threshold=DYNAMIC_THRESHOLD,
                logprob_threshold=-2.0,
            )
            
            results.append({
                "pair_idx": i,
                "camouflage_type": pair["camouflage_type"],
                "correct_kappas": c_kappas.tolist(),
                "camouflage_kappas": m_kappas.tolist(),
                "correct_logprobs": correct_logprobs.tolist(),
                "camouflage_logprobs": camouflage_logprobs.tolist(),
                "correct_features": compute_curvature_features(c_traj),
                "camouflage_features": compute_curvature_features(m_traj),
                **lead_info,
            })
            
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(pairs)}] lead_time={lead_info['lead_time_k']}")
        
        except Exception as e:
            print(f"  [Error] pair {i}: {e}")
            continue
    
    print(f"\n[Done] {len(results)} pairs processed")
    
    # 統計集計
    lead_times = [r["lead_time_k"] for r in results if r["lead_time_k"] is not None]
    kappa_detected = sum(1 for r in results if r["kappa_spike_step"] >= 0)
    logprob_detected = sum(1 for r in results if r["logprob_anomaly_step"] >= 0)
    
    summary = {
        "n_pairs": len(results),
        "sigma_code_legal": SIGMA_CODE_LEGAL,
        "base_threshold": BASE_THRESHOLD,
        "dynamic_threshold": DYNAMIC_THRESHOLD,
        "kappa_detection_rate": kappa_detected / len(results) if results else 0,
        "logprob_detection_rate": logprob_detected / len(results) if results else 0,
        "mean_lead_time_k": float(np.mean(lead_times)) if lead_times else None,
        "std_lead_time_k": float(np.std(lead_times)) if lead_times else None,
        "positive_lead_rate": sum(1 for lt in lead_times if lt > 0) / len(lead_times) if lead_times else 0,
    }
    
    print("\n" + "=" * 70)
    print("【Experiment 4.2 Results】")
    print(f"  κ_t detection rate:    {summary['kappa_detection_rate']:.1%}")
    print(f"  LogProb detection rate:{summary['logprob_detection_rate']:.1%}")
    print(f"  Mean lead time k:      {summary['mean_lead_time_k']}")
    print(f"  κ_t先行率 (k>0):       {summary['positive_lead_rate']:.1%}")
    print("=" * 70)
    
    # JSON保存
    exp42_path = output_dir / "exp42_results.json"
    with open(exp42_path, "w") as f:
        json.dump({"summary": summary, "n_detail_records": len(results)}, f, indent=2)
    print(f"\n→ {exp42_path}")
    
    # Figure生成
    plot_kappa_vs_logprob(
        results, SIGMA_CODE_LEGAL, BASE_THRESHOLD,
        str(output_dir / "fig15_kappa_vs_logprob.png")
    )
    plot_lead_time_distribution(
        lead_times,
        str(output_dir / "fig16_lead_time.png")
    )
    plot_threshold_calibration(
        sigma_values=[1.1622, 1.0227, 1.0047, 0.9905],
        base_threshold=BASE_THRESHOLD,
        output_path=str(output_dir / "fig17_threshold_calibration.png")
    )
    
    print("\n[Exp 4.2 Complete]")
    return summary


if __name__ == "__main__":
    run_exp42()
