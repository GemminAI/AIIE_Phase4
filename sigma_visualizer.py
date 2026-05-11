"""
sigma_visualizer.py — AIIE Phase 4: Figure 12/13/14 生成
Gemmina Intelligence LLC / Pure Information Laboratory
2026-05-11

Figure 12: σ ヒートマップ（4×4、対角NaN）
Figure 13: 有向伝導グラフ（networkx）
Figure 14: 方向性非対称散布図 σ(A→B) vs σ(B→A)
"""

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("[warn] networkx not found. Figure 13 will be skipped.")

DOMAINS = ["medical", "legal", "code", "conversation"]

DOMAIN_COLORS = {
    "medical":      "#E53935",  # 赤: 最高リスクドメイン
    "legal":        "#1E88E5",  # 青
    "code":         "#43A047",  # 緑
    "conversation": "#FB8C00",  # オレンジ: 拡散源想定
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "figure.dpi":       180,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


# ================================================================
# Figure 12: σ ヒートマップ
# ================================================================

def plot_sigma_heatmap(sigma_matrix: np.ndarray, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # 対角を NaN にして白表示
    mask = np.eye(len(DOMAINS), dtype=bool)

    # カラーマップ: 1.0 を境界にして赤/緑を分岐
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "conductivity",
        [(0.0, "#4CAF50"),   # 低σ: 緑（安全）
         (0.5, "#FFF9C4"),   # σ=1付近: 黄
         (1.0, "#E53935")],  # 高σ: 赤（危険）
    )

    vmax = max(2.0, np.nanmax(sigma_matrix) * 1.1)

    sns.heatmap(
        sigma_matrix,
        annot=True, fmt=".3f",
        xticklabels=DOMAINS, yticklabels=DOMAINS,
        mask=mask,
        cmap=cmap, vmin=0.0, vmax=vmax,
        center=1.0,
        linewidths=0.5, linecolor="white",
        ax=ax,
        annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={"label": "σ(A→B)", "shrink": 0.8},
    )

    ax.set_xlabel("Target domain B", labelpad=8)
    ax.set_ylabel("Source domain A", labelpad=8)
    ax.set_title(
        "Figure 12 — Semantic Conductivity Matrix σ(A→B)\n"
        "Red: Topological Camouflage (σ>1)  |  Green: Effective Isolation (σ≤1)\n"
        "AIIE Phase 4 | Gemmina Intelligence LLC",
        fontsize=9, pad=10,
    )

    plt.tight_layout()
    path = Path(output_dir) / "fig12_sigma_heatmap.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[Figure saved] {path}")
    return str(path)


# ================================================================
# Figure 13: 有向伝導グラフ
# ================================================================

def plot_contagion_graph(sigma_matrix: np.ndarray, output_dir: str) -> str:
    if not HAS_NX:
        print("[skip] networkx not available for Figure 13")
        return ""

    G = nx.DiGraph()
    G.add_nodes_from(DOMAINS)

    # エッジ: σ > 1 のみ表示（それ以外は薄いグレー）
    edge_list_high = []
    edge_list_low  = []
    edge_weights   = {}

    for i, A in enumerate(DOMAINS):
        for j, B in enumerate(DOMAINS):
            if i == j or np.isnan(sigma_matrix[i][j]):
                continue
            s = sigma_matrix[i][j]
            G.add_edge(A, B, weight=s)
            edge_weights[(A, B)] = s
            if s > 1.0:
                edge_list_high.append((A, B))
            else:
                edge_list_low.append((A, B))

    pos = nx.circular_layout(G)

    fig, ax = plt.subplots(figsize=(7, 6))

    # ノード
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=[DOMAIN_COLORS[d] for d in G.nodes()],
        node_size=2200, alpha=0.92,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9.5, font_color="white", font_weight="bold")

    # エッジ: 高σ（赤・太）
    nx.draw_networkx_edges(
        G, pos, edgelist=edge_list_high, ax=ax,
        edge_color="#E53935", width=3.0, alpha=0.85,
        arrows=True, arrowsize=20,
        connectionstyle="arc3,rad=0.18",
    )
    # エッジ: 低σ（緑・細）
    nx.draw_networkx_edges(
        G, pos, edgelist=edge_list_low, ax=ax,
        edge_color="#4CAF50", width=1.2, alpha=0.5,
        arrows=True, arrowsize=12,
        connectionstyle="arc3,rad=0.18",
    )

    # エッジラベル（σ値）
    edge_labels = {(A, B): f"{s:.2f}" for (A, B), s in edge_weights.items() if s > 1.0}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=8, font_color="#C62828",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
    )

    ax.set_title(
        "Figure 13 — Directed Semantic Contagion Graph\n"
        "Red arrows: Topological Camouflage (σ>1)  |  Green: Isolated (σ≤1)\n"
        "AIIE Phase 4 | Gemmina Intelligence LLC",
        fontsize=9,
    )
    ax.axis("off")

    plt.tight_layout()
    path = Path(output_dir) / "fig13_contagion_graph.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[Figure saved] {path}")
    return str(path)


# ================================================================
# Figure 14: 非対称性散布図 σ(A→B) vs σ(B→A)
# ================================================================

def plot_asymmetry(sigma_matrix: np.ndarray, output_dir: str) -> str:
    x_vals, y_vals, labels, colors = [], [], [], []

    for i, A in enumerate(DOMAINS):
        for j, B in enumerate(DOMAINS):
            if j <= i:
                continue
            s_ab = sigma_matrix[i][j]
            s_ba = sigma_matrix[j][i]
            if np.isnan(s_ab) or np.isnan(s_ba):
                continue
            x_vals.append(s_ab)
            y_vals.append(s_ba)
            labels.append(f"{A[:3]}↔{B[:3]}")
            # 両方向でσ>1なら赤、片方のみ橙、両方≤1なら緑
            if s_ab > 1.0 and s_ba > 1.0:
                colors.append("#E53935")
            elif s_ab > 1.0 or s_ba > 1.0:
                colors.append("#FB8C00")
            else:
                colors.append("#4CAF50")

    fig, ax = plt.subplots(figsize=(6.5, 6))

    ax.scatter(x_vals, y_vals, c=colors, s=160, alpha=0.88, zorder=3)

    # σ=1 の十字線
    lim = max(max(x_vals + y_vals) * 1.1, 1.5)
    ax.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axvline(1.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)

    # 対称線 y=x
    ax.plot([0, lim], [0, lim], color="#9E9E9E", linewidth=0.8,
            linestyle=":", alpha=0.7, label="y = x (symmetric)")

    for xi, yi, lab in zip(x_vals, y_vals, labels):
        ax.annotate(lab, (xi, yi), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)

    # 象限ラベル
    ax.text(1.05, 0.5, "A contaminates B\nB isolated",
            fontsize=8, color="#FB8C00", va="center")
    ax.text(0.5, 1.05, "B contaminates A\nA isolated",
            fontsize=8, color="#FB8C00", ha="center")
    ax.text(1.05, 1.05, "Mutual\ncontagion",
            fontsize=8, color="#E53935")
    ax.text(0.4, 0.4,  "Mutual\nisolation",
            fontsize=8, color="#4CAF50")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("σ(A→B)")
    ax.set_ylabel("σ(B→A)")
    ax.set_title(
        "Figure 14 — Directional Asymmetry of Semantic Conductivity\n"
        "AIIE Phase 4 | Gemmina Intelligence LLC",
        fontsize=9,
    )
    ax.legend(fontsize=8, framealpha=0.3)

    plt.tight_layout()
    path = Path(output_dir) / "fig14_asymmetry.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[Figure saved] {path}")
    return str(path)


def generate_all_figures(sigma_matrix: np.ndarray, output_dir: str) -> list:
    paths = []
    paths.append(plot_sigma_heatmap(sigma_matrix, output_dir))
    paths.append(plot_contagion_graph(sigma_matrix, output_dir))
    paths.append(plot_asymmetry(sigma_matrix, output_dir))
    return [p for p in paths if p]
