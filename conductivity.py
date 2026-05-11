"""
conductivity.py — AIIE Phase 4: Semantic Conductivity σ(A→B)
Gemmina Intelligence LLC / Pure Information Laboratory
2026-05-11
"""

import numpy as np
from itertools import permutations
from scipy.spatial.distance import cdist

try:
    import ot
except ImportError:
    raise ImportError("POT が必要です: pip install POT>=0.9.0")


DOMAINS = ["medical", "legal", "code", "conversation"]


# ================================================================
# GW距離（Phase 3 manifold_aligner.py を再実装・統合）
# ================================================================

def build_distance_matrix(states: np.ndarray, metric: str = "cosine") -> np.ndarray:
    C = cdist(states, states, metric=metric).astype(np.float64)
    max_val = C.max()
    return C / max_val if max_val > 1e-10 else C


def compute_gw(
    states_A: np.ndarray,
    states_B: np.ndarray,
    epsilon:  float = 0.1,
    n_iter:   int   = 100,
    metric:   str   = "cosine",
) -> float:
    C_A  = build_distance_matrix(states_A, metric)
    C_B  = build_distance_matrix(states_B, metric)
    mu_A = np.ones(len(states_A)) / len(states_A)
    mu_B = np.ones(len(states_B)) / len(states_B)

    _, log = ot.gromov.entropic_gromov_wasserstein(
        C_A, C_B, mu_A, mu_B,
        loss_fun="square_loss",
        epsilon=epsilon,
        max_iter=n_iter,
        log=True,
        verbose=False,
    )
    return float(log["gw_dist"])


# ================================================================
# hidden states の抽出（pkl results → np.ndarray）
# ================================================================

def extract_states(
    results: list,
    condition: str,  # "correct" | "failed"
    layer:    int,
    n_samples: int  = 200,
    seed:      int  = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = []

    for r in results:
        hidden = r[condition]["hidden"]
        if layer not in hidden:
            continue
        traj = hidden[layer]
        if isinstance(traj, np.ndarray) and traj.ndim == 2 and traj.shape[0] > 0:
            states.append(np.mean(traj, axis=0))

    if not states:
        raise ValueError(f"condition={condition}, layer={layer}: 0 samples found")

    states = np.array(states, dtype=np.float64)
    if len(states) > n_samples:
        idx = rng.choice(len(states), n_samples, replace=False)
        states = states[idx]

    return states


# ================================================================
# Semantic Conductivity σ(A→B)
# ================================================================

def compute_sigma(
    states_A_corr: np.ndarray,
    states_A_fail: np.ndarray,
    states_B_corr: np.ndarray,
    cfg,
) -> dict:
    """
    σ(A→B) = GW(A_corr, B_corr) / GW(A_fail, B_corr)

    Returns:
        {
            "gw_cc":          float,  # GW(A_corr, B_corr)
            "gw_fc":          float,  # GW(A_fail, B_corr)
            "sigma":          float,
            "delta":          float,  # gw_cc - gw_fc （正 → camouflage zone侵入）
            "gamma":          float,  # camouflage intensity = 1 - 1/σ
            "high_contagion": bool,   # σ > 1
        }
    """
    gw_cc = compute_gw(states_A_corr, states_B_corr, cfg.epsilon, cfg.n_iter)
    gw_fc = compute_gw(states_A_fail, states_B_corr, cfg.epsilon, cfg.n_iter)

    sigma = gw_cc / (gw_fc + 1e-12)
    delta = gw_cc - gw_fc
    gamma = 1.0 - 1.0 / sigma if sigma > 1e-6 else 0.0

    return {
        "gw_cc":          gw_cc,
        "gw_fc":          gw_fc,
        "sigma":          sigma,
        "delta":          delta,
        "gamma":          gamma,
        "high_contagion": sigma > cfg.sigma_critical,
    }


# ================================================================
# 4×4 Σ行列の構築
# ================================================================

def build_sigma_matrix(domain_results: dict, cfg) -> tuple:
    """
    全ペアの σ を計算して 4×4 行列を返す。

    Returns:
        sigma_matrix: np.ndarray (4, 4) — 対角は NaN
        detail:       dict — {"{A}→{B}": result_dict}
    """
    n = len(DOMAINS)
    sigma_matrix = np.full((n, n), np.nan)
    detail       = {}

    for i, A in enumerate(DOMAINS):
        # 各ドメインの states を事前抽出
        sA_corr = extract_states(
            domain_results[A], "correct", cfg.target_layer, cfg.n_gw, cfg.seed
        )
        sA_fail = extract_states(
            domain_results[A], "failed", cfg.target_layer, cfg.n_gw, cfg.seed
        )

        for j, B in enumerate(DOMAINS):
            if i == j:
                continue

            sB_corr = extract_states(
                domain_results[B], "correct", cfg.target_layer, cfg.n_gw, cfg.seed
            )

            print(f"  σ({A}→{B}) ...", flush=True)
            r = compute_sigma(sA_corr, sA_fail, sB_corr, cfg)
            sigma_matrix[i][j] = r["sigma"]
            detail[f"{A}→{B}"] = r

            status = "HIGH CONTAGION 🔴" if r["high_contagion"] else "isolated 🟢"
            print(f"    σ={r['sigma']:.4f}  γ={r['gamma']:.4f}  {status}")

    return sigma_matrix, detail


# ================================================================
# 結果テーブルの文字列生成
# ================================================================

def format_sigma_table(sigma_matrix: np.ndarray) -> str:
    header = "σ(A→B)      | " + " | ".join(f"{d[:6]:>8}" for d in DOMAINS)
    sep    = "-" * len(header)
    lines  = [header, sep]

    for i, A in enumerate(DOMAINS):
        row = f"{A[:10]:<12}| "
        for j in range(len(DOMAINS)):
            if i == j:
                row += f"{'  —  ':>8} | "
            else:
                val = sigma_matrix[i][j]
                mark = "🔴" if val > 1.0 else "🟢"
                row += f"{val:>6.3f}{mark} | "
        lines.append(row)

    lines.append(sep)
    n_high = int(np.sum(sigma_matrix[~np.isnan(sigma_matrix)] > 1.0))
    n_total = int(np.sum(~np.isnan(sigma_matrix)))
    lines.append(f"\nTopological Camouflage 検出: {n_high}/{n_total} ペア")
    return "\n".join(lines)
