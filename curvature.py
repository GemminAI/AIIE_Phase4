"""
experiment/curvature.py

【離散軌道曲率 κ_t の定義】

本 κ_t は AIIE RFC-0039 §2 / RFC-0040 §3.1 に準拠した計算仕様である。
AIIE RFC-0016（35TAG 空間における κ_vec = SIV の2階位置差分）とは
適用空間・計算式ともに独立した別定義であり、混同してはならない。

  v_t   = h_t - h_{t-1}                     (速度ベクトル)
  v̂_t  = v_t / ‖v_t‖                       (単位速度ベクトル: 方向のみ)

  κ_t = ‖v̂_{t+1} - v̂_t‖ / (‖v_t‖ + ε)   (RFC-0039 §2 / RFC-0040 §3.1)

  ε = 1e-8（ゼロ除算防止）

【1st-order Drift δ_t（ベースライン指標）の定義】

  δ_t = ‖v_t‖ = ‖h_t - h_{t-1}‖            (RFC-0040 §5.1 Baseline 1)

  κ_t との対比により「2階微分の優位性（なぜ曲率か）」を実証する。

【NOTE】
  κ_t, δ_t ともに高次元空間（~4096 次元）でそのまま計算する。
  PCA による次元削減は可視化にのみ使用し、計量計算には使用しない。
  論文 §3.3 に明記すること。
"""

from typing import Dict, List, Optional
import numpy as np


def compute_discrete_curvature(
    trajectory: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    1つの hidden state 軌道から離散曲率時系列 κ_t を計算する。

    Args:
        trajectory: shape (T, D) — T 時刻 × D 次元の hidden state 列
        eps:        数値安定化パラメータ

    Returns:
        kappas: shape (T,) — 先頭・末尾の 1 点は 0（速度計算不可）
    """
    T, D = trajectory.shape
    kappas = np.zeros(T)

    for t in range(1, T - 1):
        v1 = trajectory[t]   - trajectory[t - 1]   # (D,)
        v2 = trajectory[t+1] - trajectory[t]        # (D,)

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # 停滞点（速度 ≒ 0）はスキップ
        if norm_v1 < eps or norm_v2 < eps:
            kappas[t] = 0.0
            continue

        hat_v1 = v1 / norm_v1
        hat_v2 = v2 / norm_v2

        # 方向変化量を局所スケールで正規化
        kappas[t] = np.linalg.norm(hat_v2 - hat_v1) / (norm_v1 + eps)

    return kappas


def compute_first_order_drift(
    trajectory: np.ndarray,
) -> np.ndarray:
    """
    1st-order Drift δ_t: 各ステップの移動距離 ‖v_t‖（RFC-0040 §5.1 Baseline 1）

    κ_t（2階）との AUC 比較により「なぜ2階微分が必要か」を論証する。

    Args:
        trajectory: shape (T, D)

    Returns:
        drifts: shape (T,) — t=0 は 0（前ステップなし）
    """
    T = len(trajectory)
    drifts = np.zeros(T)
    for t in range(1, T):
        drifts[t] = np.linalg.norm(trajectory[t] - trajectory[t - 1])
    return drifts


def compute_curvature_features(
    trajectory: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    軌道から統計量を抽出する。論文 Table 2 / Figure 3 向け。

    Returns:
        {
          "kappa_mean":   軌道全体の平均曲率（主要指標）
          "kappa_max":    最大曲率（スパイク強度）
          "kappa_std":    曲率の標準偏差（変動性）
          "kappa_early":  前半 50% の平均曲率（早期検出指標）
          "kappa_late":   後半 50% の平均曲率
          "drift_mean":   軌道全体の平均 1st-order Drift（ベースライン1用）
          "drift_std":    1st-order Drift の標準偏差
        }
    """
    kappas = compute_discrete_curvature(trajectory, eps=eps)
    drifts = compute_first_order_drift(trajectory)
    T = len(kappas)
    mid = T // 2

    nonzero_k = kappas[kappas > 0]
    if len(nonzero_k) == 0:
        nonzero_k = np.array([0.0])

    nonzero_d = drifts[drifts > 0]
    if len(nonzero_d) == 0:
        nonzero_d = np.array([0.0])

    return {
        "kappa_mean":  float(np.mean(nonzero_k)),
        "kappa_max":   float(np.max(nonzero_k)),
        "kappa_std":   float(np.std(nonzero_k)),
        "kappa_early": float(np.mean(kappas[:mid][kappas[:mid] > 0]) if np.any(kappas[:mid] > 0) else 0.0),
        "kappa_late":  float(np.mean(kappas[mid:][kappas[mid:] > 0]) if np.any(kappas[mid:] > 0) else 0.0),
        "drift_mean":  float(np.mean(nonzero_d)),
        "drift_std":   float(np.std(nonzero_d)),
    }


def compute_all_curvatures(
    extraction_results: List[Dict],
    target_layers: List[int],
    eps: float = 1e-8,
) -> Dict[str, List]:
    """
    extract_batch() の結果を受け取り、全ペア・全層の曲率・drift・logprob を集計する。

    Returns:
        {
          "correct_features": List[Dict[layer_idx, Dict[str, float]]],
          "failed_features":  List[Dict[layer_idx, Dict[str, float]]],
          "correct_kappas":   List[Dict[layer_idx, np.ndarray]],
          "failed_kappas":    List[Dict[layer_idx, np.ndarray]],
          "correct_drifts":   List[Dict[layer_idx, np.ndarray]],  ← NEW: 1st-order drift時系列
          "failed_drifts":    List[Dict[layer_idx, np.ndarray]],  ← NEW
          "correct_logprobs": List[np.ndarray],
          "failed_logprobs":  List[np.ndarray],
        }
    """
    correct_features = []
    failed_features  = []
    correct_kappas   = []
    failed_kappas    = []
    correct_drifts   = []
    failed_drifts    = []
    correct_logprobs = []
    failed_logprobs  = []

    for result in extraction_results:
        c_hidden = result["correct"]["hidden"]
        f_hidden = result["failed"]["hidden"]
        c_lp     = result["correct"]["logprobs"]
        f_lp     = result["failed"]["logprobs"]

        c_feat = {}
        f_feat = {}
        c_kap  = {}
        f_kap  = {}
        c_dft  = {}
        f_dft  = {}

        for layer_idx in target_layers:
            c_traj = c_hidden.get(layer_idx)
            f_traj = f_hidden.get(layer_idx)

            if c_traj is None or f_traj is None or len(c_traj) < 3 or len(f_traj) < 3:
                continue

            c_feat[layer_idx] = compute_curvature_features(c_traj, eps=eps)
            f_feat[layer_idx] = compute_curvature_features(f_traj, eps=eps)
            c_kap[layer_idx]  = compute_discrete_curvature(c_traj, eps=eps)
            f_kap[layer_idx]  = compute_discrete_curvature(f_traj, eps=eps)
            c_dft[layer_idx]  = compute_first_order_drift(c_traj)
            f_dft[layer_idx]  = compute_first_order_drift(f_traj)

        correct_features.append(c_feat)
        failed_features.append(f_feat)
        correct_kappas.append(c_kap)
        failed_kappas.append(f_kap)
        correct_drifts.append(c_dft)
        failed_drifts.append(f_dft)
        correct_logprobs.append(c_lp)
        failed_logprobs.append(f_lp)

    return {
        "correct_features": correct_features,
        "failed_features":  failed_features,
        "correct_kappas":   correct_kappas,
        "failed_kappas":    failed_kappas,
        "correct_drifts":   correct_drifts,
        "failed_drifts":    failed_drifts,
        "correct_logprobs": correct_logprobs,
        "failed_logprobs":  failed_logprobs,
    }
