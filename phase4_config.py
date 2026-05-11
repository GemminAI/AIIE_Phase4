"""
phase4_config.py — AIIE Phase 4: Cultural Transport Dynamics
Gemmina Intelligence LLC / Pure Information Laboratory
2026-05-11
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Phase4Config:

    # ---------------------------------------------------------------
    # モデル設定（Phase 1と統一）
    # ---------------------------------------------------------------
    model_name:   str = "meta-llama/Meta-Llama-3-8B-Instruct"
    target_layer: int = 7          # Peak curvature layer（Phase 1確定値）
    device:       str = "cuda"
    max_new_tokens: int = 256

    # ---------------------------------------------------------------
    # 実験規模
    # ---------------------------------------------------------------
    n_pairs:     int = 500         # ドメインごとのペア数
    n_gw:        int = 200         # GW計算用サブサンプル数
    seed:        int = 42

    # ---------------------------------------------------------------
    # GW計算パラメータ（Phase 3と統一）
    # ---------------------------------------------------------------
    epsilon:         float = 0.1
    n_iter:          int   = 100
    distance_metric: str   = "cosine"

    # ---------------------------------------------------------------
    # データセット設定
    # ---------------------------------------------------------------
    # Medical: MedQA (HuggingFace)
    medqa_dataset:   str = "GBaker/MedQA-USMLE-4-options"
    medqa_split:     str = "test"

    # Legal: LegalBench / contract_nli (HuggingFace)
    legal_dataset:   str = "nguha/legalbench"
    legal_config:    str = "contract_nli"
    legal_split:     str = "test"

    # Code: HumanEval (HuggingFace)
    code_dataset:    str = "openai/openai_humaneval"
    code_split:      str = "test"

    # Conversation: ShareGPT (ローカルJSONファイル)
    sharegpt_path:   str = "./ShareGPT_V3_unfiltered_cleaned_split.json"

    # ---------------------------------------------------------------
    # 出力パス
    # ---------------------------------------------------------------
    output_dir:     str = "./results_phase4"
    pkl_dir:        str = "./results_phase4/pkl"

    # ---------------------------------------------------------------
    # σ 閾値
    # ---------------------------------------------------------------
    sigma_critical: float = 1.0   # σ > 1.0 → Topological Camouflage

    # ---------------------------------------------------------------
    # 実験4.2: κ_t伝播
    # ---------------------------------------------------------------
    contamination_lengths: list = field(
        default_factory=lambda: [0, 50, 100, 200, 500]
    )
    contamination_pairs: list = field(
        default_factory=lambda: [
            ("conversation", "medical"),
            ("legal",        "medical"),
            ("code",         "medical"),
        ]
    )
    n_propagation_pairs: int = 50  # 伝播実験のペア数

    def validate(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.pkl_dir).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Phase4Config validated. output_dir={self.output_dir}")
