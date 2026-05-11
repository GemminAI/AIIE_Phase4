"""
phase4_main.py — AIIE Phase 4: Cultural Transport Dynamics
エントリーポイント

実行方法:
    cd /workspace
    python -m aiie_phase4.phase4_main 2>&1 | tee results_phase4/phase4_run.log

Gemmina Intelligence LLC / Pure Information Laboratory
2026-05-11
"""

import json
import time
from datetime import datetime
from pathlib import Path

from .phase4_config    import Phase4Config
from .domain_loader    import load_all_domains
from .phase4_extractor import run_extraction
from .conductivity     import build_sigma_matrix, format_sigma_table
from .sigma_visualizer import generate_all_figures


def main():
    print("=" * 70)
    print("AIIE Phase 4: Cultural Transport Dynamics")
    print("Gemmina Intelligence LLC / Pure Information Laboratory")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    cfg = Phase4Config()
    cfg.validate()
    t_start = time.time()

    # ---------------------------------------------------------------
    # Step 1: データロード（4ドメイン）
    # ---------------------------------------------------------------
    print("\n[Step 1] Loading domain datasets ...")
    domain_data = load_all_domains(cfg)
    for d, pairs in domain_data.items():
        print(f"  {d}: {len(pairs)} pairs")

    # ---------------------------------------------------------------
    # Step 2: Hidden State 抽出（GPU）
    # ---------------------------------------------------------------
    print("\n[Step 2] Extracting hidden states (Llama-3-8B, L7) ...")
    domain_results = run_extraction(cfg, domain_data)
    for d, results in domain_results.items():
        print(f"  {d}: {len(results)} extracted")

    # ---------------------------------------------------------------
    # Step 3: σ 行列計算（CPU Only）
    # ---------------------------------------------------------------
    print("\n[Step 3] Computing Semantic Conductivity matrix ...")
    print(f"  n_gw={cfg.n_gw}, epsilon={cfg.epsilon}, n_iter={cfg.n_iter}")
    sigma_matrix, sigma_detail = build_sigma_matrix(domain_results, cfg)

    # テーブル表示
    print("\n" + "=" * 70)
    print("【Semantic Conductivity Matrix σ(A→B)】")
    print("=" * 70)
    print(format_sigma_table(sigma_matrix))

    # JSON保存
    output_dir = Path(cfg.output_dir)
    json_path  = output_dir / "sigma_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {k: {kk: float(vv) for kk, vv in v.items() if kk != "high_contagion"}
             | {"high_contagion": bool(v["high_contagion"])}
             for k, v in sigma_detail.items()},
            f, indent=2
        )
    print(f"\n→ {json_path}")

    # ---------------------------------------------------------------
    # Step 4: Figure 生成
    # ---------------------------------------------------------------
    print("\n[Step 4] Generating figures ...")
    fig_paths = generate_all_figures(sigma_matrix, str(output_dir))
    for p in fig_paths:
        print(f"  {p}")

    # ---------------------------------------------------------------
    # Step 5: 最終サマリー
    # ---------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("Phase 4 Experiment 4.1 完了")
    print("=" * 70)

    high_pairs = [(k, v["sigma"]) for k, v in sigma_detail.items() if v["high_contagion"]]
    print(f"\nTopological Camouflage 検出: {len(high_pairs)}/12 ペア")
    if high_pairs:
        print("\n高伝導経路（σ > 1.0）:")
        for pair, sigma in sorted(high_pairs, key=lambda x: -x[1]):
            print(f"  {pair}: σ = {sigma:.4f}")

    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print(f"Output: {output_dir.resolve()}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return sigma_matrix, sigma_detail


if __name__ == "__main__":
    main()
