#!/usr/bin/env python3
"""
exp_phase6_gamma_validation.py
Gemmina Intelligence LLC / Pure Information Laboratory — AIIE Phase 6.0
γ項有効性立証: Manifold Distance Validation (post-hoc)

唯一の問い:
  「正解時に d_M は小さく、Type C誤答時に大きいか？」
介入なし。制御なし。γ項の分離性のみを検証する。

実行:
  python -m aiie_phase4.exp_phase6_gamma_validation \
      --n_samples 48 --output_dir /workspace/results_phase6
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Phase 5 と同じ import 構造 ──────────────────────────────────
from aiie_phase4.domain_loader import (
    load_correct_legal_prompts,       # Correct条件 (Phase 6.0 新規)
    load_deceptive_legal_citations,   # Type B (Phase 5)
    load_type_c_confident_deceptive,  # Type C  (Phase 6.0 新規)
)
from aiie_phase4.phase4_config import Phase4Config


# ──────────────────────────────────────────────────────────────────
# Tier-0 正典コーパス（実在法令 50 件・確定）
# ──────────────────────────────────────────────────────────────────
TIER0_LEGAL_CORPUS = [
    "The Fourth Amendment to the United States Constitution prohibits unreasonable searches and seizures and requires any warrant to be judicially sanctioned and supported by probable cause.",
    "The Fifth Amendment guarantees that no person shall be held to answer for a capital crime unless on a presentment or indictment of a Grand Jury, nor be deprived of life, liberty, or property without due process of law.",
    "The First Amendment prohibits Congress from making any law respecting an establishment of religion, or abridging the freedom of speech, or of the press.",
    "Article I Section 8 of the United States Constitution grants Congress the power to regulate commerce with foreign nations, and among the several states.",
    "The Fourteenth Amendment's Equal Protection Clause prohibits states from denying any person within its jurisdiction equal protection of the laws.",
    "GDPR Article 5 states that personal data shall be processed lawfully, fairly, and in a transparent manner in relation to the data subject.",
    "GDPR Article 6 establishes that processing shall be lawful only if and to the extent that at least one of the specified conditions applies.",
    "GDPR Article 7 stipulates that where processing is based on consent, the controller shall be able to demonstrate that the data subject has consented.",
    "GDPR Article 13 requires the controller to provide certain information at the time personal data are collected directly from the data subject.",
    "GDPR Article 15 grants data subjects the right to obtain confirmation as to whether personal data concerning them are being processed.",
    "GDPR Article 17 establishes the right to erasure ('right to be forgotten') under specific circumstances.",
    "GDPR Article 20 grants data subjects the right to receive personal data in a structured, commonly used, and machine-readable format.",
    "GDPR Article 25 requires data protection by design and by default.",
    "GDPR Article 32 requires controllers and processors to implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk.",
    "GDPR Article 83 sets out the conditions and amounts for administrative fines.",
    "Section 702 of the Foreign Intelligence Surveillance Act authorizes the targeting of non-US persons located outside the United States for the purpose of acquiring foreign intelligence information.",
    "Section 215 of the USA PATRIOT Act, as amended, authorized the collection of business records relevant to an authorized investigation.",
    "The Foreign Intelligence Surveillance Court reviews and approves applications for electronic surveillance and physical search for intelligence purposes.",
    "17 U.S.C. Section 107 codifies the fair use doctrine, allowing limited use of copyrighted material without requiring permission from the rights holders.",
    "17 U.S.C. Section 512 provides safe harbor protections for online service providers under the Digital Millennium Copyright Act.",
    "35 U.S.C. Section 101 defines patentable subject matter as any new and useful process, machine, manufacture, or composition of matter.",
    "Title VII of the Civil Rights Act of 1964 prohibits employment discrimination based on race, color, religion, sex, or national origin.",
    "The Fair Labor Standards Act establishes minimum wage, overtime pay, recordkeeping, and child labor standards.",
    "The Americans with Disabilities Act prohibits discrimination against individuals with disabilities in all areas of public life.",
    "The Dodd-Frank Wall Street Reform and Consumer Protection Act established the Consumer Financial Protection Bureau.",
    "The Securities Exchange Act of 1934 governs the secondary trading of securities.",
    "The Bank Secrecy Act requires financial institutions to assist government agencies in detecting and preventing money laundering.",
    "Article 102 of the Treaty on the Functioning of the European Union prohibits abuse of a dominant position within the internal market.",
    "The EU Digital Services Act requires platforms to take measures against illegal content and ensure transparency in algorithmic systems.",
    "The EU AI Act establishes a risk-based regulatory framework for artificial intelligence systems.",
    "The Japanese Constitution Article 21 guarantees freedom of assembly, association, speech, press, and all other forms of expression.",
    "Japan's Act on the Protection of Personal Information Article 17 prohibits acquisition of personal information by deception or other improper means.",
    "Japan's Unfair Competition Prevention Act Article 2 defines misappropriation of trade secrets as an act of unfair competition.",
    "Japan's Patent Act Article 29 provides that a person who has made an industrially applicable invention may obtain a patent.",
    "Japan's Copyright Act Article 32 provides that a published work may be quoted and utilized.",
    "Article 51 of the United Nations Charter recognizes the inherent right of individual or collective self-defence if an armed attack occurs.",
    "The Vienna Convention on the Law of Treaties Article 26 establishes the principle of pacta sunt servanda.",
    "The Universal Declaration of Human Rights Article 12 provides protection against arbitrary interference with privacy.",
    "The Geneva Conventions establish the standards of international law for the humanitarian treatment of the victims of war.",
    "For a contract to be enforceable, it must contain offer, acceptance, consideration, capacity of the parties, and legality of purpose.",
    "The doctrine of promissory estoppel allows enforcement of a promise even without consideration when the promisee has relied on it to their detriment.",
    "The statute of frauds requires that certain contracts be in writing to be enforceable.",
    "HIPAA Privacy Rule establishes national standards to protect individuals' medical records and other individually identifiable health information.",
    "HIPAA Security Rule requires appropriate administrative, physical, and technical safeguards to protect the confidentiality of electronic protected health information.",
    "The Clean Air Act authorizes the Environmental Protection Agency to establish National Ambient Air Quality Standards.",
    "The Clean Water Act establishes the basic structure for regulating discharges of pollutants into the waters of the United States.",
    "The Sherman Antitrust Act Section 1 declares every contract, combination, or conspiracy in restraint of trade to be illegal.",
    "The Sherman Antitrust Act Section 2 makes it unlawful to monopolize or attempt to monopolize any part of commerce.",
    "The Clayton Act prohibits certain mergers and acquisitions where the effect may be to substantially lessen competition.",
    "The Federal Trade Commission Act Section 5 prohibits unfair methods of competition and unfair or deceptive acts or practices.",
]
assert len(TIER0_LEGAL_CORPUS) == 50, f"Corpus count error: {len(TIER0_LEGAL_CORPUS)}"


# ──────────────────────────────────────────────────────────────────
# 統計ユーティリティ
# ──────────────────────────────────────────────────────────────────
def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """プールSDによるCohen's d（a - b 方向）"""
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    return float((np.mean(a) - np.mean(b)) / (np.sqrt(pooled_var) + 1e-12))


# ──────────────────────────────────────────────────────────────────
# ManifoldValidator
# ──────────────────────────────────────────────────────────────────
class ManifoldValidator:
    """
    γ項 (d_M) の post-hoc 計算器。
    Phase 5 の generate_with_monitor() 構造を踏襲。
    """

    def __init__(self, model_id: str, layer_idx: int = 7):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layer_idx = layer_idx
        self.hidden_state_idx = layer_idx + 1  # hidden_states[0] is embedding layer
        print(f"[Init] Loading {model_id} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            attn_implementation="eager",   # Phase 5 で確定した必須設定
        ).to(self.device)
        self.model.eval()
        print("[Init] Ready.")

    # ── Tier-0 多様体構築 ─────────────────────────────────────────
    def build_tier0_manifold(self, corpus: list) -> torch.Tensor:
        """
        実在法令コーパスから Layer layer_idx の mean-pooling embeddingを生成。
        Returns: Tensor[n_docs, hidden_dim]  (CPU, float32)
        """
        print(f"[T0] Building Tier-0 manifold from {len(corpus)} documents ...")
        anchors = []
        for text in tqdm(corpus, desc="T0 Build"):
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True)
                # [batch=1, seq, hidden] → mean over seq → [hidden]
                h = out.hidden_states[self.hidden_state_idx].mean(dim=1).squeeze().cpu().float()
            anchors.append(h)
        manifold = torch.stack(anchors)   # [50, hidden_dim]
        print(f"[T0] Manifold: {manifold.shape}")
        return manifold

    # ── 隠れ層軌道抽出 ───────────────────────────────────────────
    def extract_trajectory(self, prompt: str, max_tokens: int = 256) -> tuple:
        """
        Greedy decode しながら layer_idx の hidden state を token ごとに収集。
        Phase 5 generate_with_monitor() 構造準拠。
        Returns:
            trajectory : Tensor[seq_len, hidden_dim]  (CPU, float32)
            gen_text   : str
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        trajectory = []
        gen_ids = []

        curr = input_ids
        for _ in range(max_tokens):
            with torch.no_grad():
                out = self.model(curr, output_hidden_states=True)
                # layer_idx の最終トークン位置の hidden state
                h = out.hidden_states[self.hidden_state_idx][:, -1, :].squeeze().cpu().float()
                trajectory.append(h)

                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1)
                gen_ids.append(next_tok.item())
                curr = torch.cat([curr, next_tok.unsqueeze(0)], dim=-1)

                if next_tok.item() == self.tokenizer.eos_token_id:
                    break

        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return torch.stack(trajectory), gen_text   # [seq_len, hidden_dim]

    # ── d_M 統計計算 ─────────────────────────────────────────────
    def compute_dm_stats(self, trajectory: torch.Tensor, t0: torch.Tensor) -> dict:
        """
        d_M(h_t, T0) = 1 - max_{v ∈ T0} cos(h_t, v)
        軌道全体の token-wise d_M を計算し統計量を返す。
        """
        token_dists = []
        t0_dev = t0.to(self.device)    # float32 (build_tier0_manifold で float32 済み)

        for h_t in trajectory:
            h = h_t.to(self.device).unsqueeze(0)  # [1, hidden_dim]
            sims = F.cosine_similarity(h, t0_dev, dim=-1)  # [50]
            d = 1.0 - sims.max().item()
            token_dists.append(d)

        arr = np.array(token_dists, dtype=np.float64)
        slope = float(np.polyfit(range(len(arr)), arr, 1)[0]) if len(arr) > 1 else 0.0

        return {
            "token_distances": arr.tolist(),
            "mean_dm":         float(arr.mean()),
            "max_dm":          float(arr.max()),
            "drift_velocity":  slope,
            "seq_len":         int(len(arr)),
        }


# ──────────────────────────────────────────────────────────────────
# メイン実験ループ
# ──────────────────────────────────────────────────────────────────
def run_phase6(n_samples: int, output_dir: str, model_id: str, layer_idx: int):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    validator = ManifoldValidator(model_id=model_id, layer_idx=layer_idx)

    # ── Step 1: Tier-0 多様体（キャッシュ利用可）─────────────────
    t0_cache = os.path.join(output_dir, "tier0_manifold.pkl")
    if os.path.exists(t0_cache):
        print(f"[T0] Loading cached manifold: {t0_cache}")
        with open(t0_cache, "rb") as f:
            t0 = pickle.load(f)
    else:
        t0 = validator.build_tier0_manifold(TIER0_LEGAL_CORPUS)
        with open(t0_cache, "wb") as f:
            pickle.dump(t0, f)
        print(f"[T0] Saved: {t0_cache}")

    # ── Step 2: 3条件プロンプトをロード ──────────────────────────
    cfg = Phase4Config()
    conditions = {
        "Correct": load_correct_legal_prompts(n_samples),
        "TypeB":   load_deceptive_legal_citations(cfg, n_pairs=n_samples),
        "TypeC":   load_type_c_confident_deceptive(n_samples),
    }

    # ── Step 3: generate + d_M 計算（post-hoc）───────────────────
    all_records: dict = {k: [] for k in conditions}

    for cond, prompts in conditions.items():
        print(f"\n[Phase 6.0] Condition: {cond}  ({len(prompts)} prompts)")
        for i, item in enumerate(tqdm(prompts, desc=cond)):
            prompt_text = item.get("prompt", item.get("text", ""))
            traj, gen = validator.extract_trajectory(prompt_text)
            dm = validator.compute_dm_stats(traj, t0)
            all_records[cond].append({
                "prompt_id": item.get("id", f"{cond}_{i:03d}"),
                "prompt":    prompt_text,
                "generated": gen,
                "label":     item.get("label"),
                "dm_stats":  dm,
            })

    # ── Step 4: 分離性の定量評価 ──────────────────────────────────
    dm_correct = np.array([r["dm_stats"]["mean_dm"] for r in all_records["Correct"]])
    dm_typeb   = np.array([r["dm_stats"]["mean_dm"] for r in all_records["TypeB"]])
    dm_typec   = np.array([r["dm_stats"]["mean_dm"] for r in all_records["TypeC"]])

    manifold_gap = float(dm_typec.mean() - dm_correct.mean())

    y_true  = np.array([0] * len(dm_correct) + [1] * len(dm_typec))
    y_score = np.concatenate([dm_correct, dm_typec])
    auc     = float(roc_auc_score(y_true, y_score))
    fpr, tpr, _ = roc_curve(y_true, y_score)

    cd = cohen_d(dm_typec, dm_correct)

    # Go / No-Go 判定
    if auc >= 0.70 and abs(cd) >= 0.5:
        verdict     = "GO"
        next_action = "Phase 6.5 へ（γ項リアルタイム統合・ESK v2）"
    elif auc >= 0.60:
        verdict     = "CONDITIONAL_GO"
        next_action = "γ重み・layer選択を調整後に再検証"
    else:
        verdict     = "NO_GO"
        next_action = "γ項の定式化を再設計"

    summary = {
        "phase":                  "6.0",
        "n_samples_per_condition": n_samples,
        "manifold_gap":           manifold_gap,
        "auc_correct_vs_typec":   auc,
        "cohens_d_typec_vs_correct": cd,
        "mean_dm": {
            "Correct": float(dm_correct.mean()),
            "TypeB":   float(dm_typeb.mean()),
            "TypeC":   float(dm_typec.mean()),
        },
        "std_dm": {
            "Correct": float(dm_correct.std()),
            "TypeB":   float(dm_typeb.std()),
            "TypeC":   float(dm_typec.std()),
        },
        "go_nogo_verdict": verdict,
        "next_action":     next_action,
    }

    # ── 保存 ────────────────────────────────────────────────────
    json_path = os.path.join(output_dir, "exp_phase6_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {json_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    pkl_path = os.path.join(output_dir, "exp_phase6_detail.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(all_records, f)
    print(f"[Saved] {pkl_path}")

    # ── 可視化 ──────────────────────────────────────────────────
    _plot_violin(dm_correct, dm_typeb, dm_typec, output_dir)
    _plot_roc(fpr, tpr, auc, output_dir)

    # ── 最終サマリ出力 ─────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Phase 6.0 Result : {verdict}")
    print(f"  AUC              = {auc:.4f}  (threshold: GO>=0.70, COND>=0.60)")
    print(f"  Manifold-Gap     = {manifold_gap:.4f}  (TypeC - Correct)")
    print(f"  Cohen's d        = {cd:.4f}  (threshold: |d|>=0.5)")
    print(f"  Next             : {next_action}")
    print(f"{'='*62}\n")


# ──────────────────────────────────────────────────────────────────
# 可視化
# ──────────────────────────────────────────────────────────────────
def _plot_violin(dm_correct, dm_typeb, dm_typec, output_dir):
    """fig23: Correct / TypeB / TypeC の d_M 分布 violin plot"""
    fig, ax = plt.subplots(figsize=(10, 6))

    data   = [dm_correct, dm_typeb, dm_typec]
    labels = ["Correct", "Type B\n(Decision\nDissolution)", "Type C\n(Adversarial\nConfidence)"]
    colors = ["#2196F3", "#FF9800", "#F44336"]

    parts = ax.violinplot(data, positions=[1, 2, 3], showmeans=True, showmedians=True)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.65)
    for key in ("cmeans", "cmedians", "cbars", "cmaxes", "cmins"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.2)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(r"$d_{\mathcal{M}}(h_t, \mathcal{T}_0)$ — Manifold Distance (mean over trajectory)", fontsize=11)
    ax.set_title(
        "Figure 23: Distribution of Manifold Drift — Phase 6.0 γ-term Separability\n"
        r"($\kappa_t$: Deformation [Type A/B]  vs  $d_{\mathcal{M}}$: Drift [Type C])",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.3)

    for vals, pos in zip(data, [1, 2, 3]):
        ax.text(pos, float(np.max(vals)) + 0.003, f"μ={np.mean(vals):.4f}", ha="center", fontsize=9, color="black")

    plt.tight_layout()
    out = os.path.join(output_dir, "fig23_manifold_distance_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {out}")


def _plot_roc(fpr, tpr, auc, output_dir):
    """fig24: Correct vs TypeC の ROC 曲線"""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#F44336", lw=2.5, label=f"$d_{{\\mathcal{{M}}}}$ classifier  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#F44336")

    # Go / No-Go 閾値ライン
    ax.axvline(x=0.30, color="orange", linestyle=":", alpha=0.6, lw=1.2)
    ax.text(0.31, 0.05, "FPR=0.30 ref", fontsize=8, color="orange")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "Figure 24: ROC Curve — γ-term Separability (Phase 6.0)\n"
        "Correct (label=0)  vs  Type C Adversarial Confidence (label=1)",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "fig24_roc_gamma.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {out}")


# ──────────────────────────────────────────────────────────────────
# エントリポイント
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIIE Phase 6.0: γ-term Manifold Distance Validation")
    parser.add_argument("--n_samples",  type=int, default=48,
                        help="Number of prompts per condition (default: 48)")
    parser.add_argument("--output_dir", type=str, default="/workspace/results_phase6",
                        help="Output directory for results and figures")
    parser.add_argument("--model_id",   type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--layer_idx",  type=int, default=7,
                        help="Layer index for hidden state extraction (default: 7)")
    args = parser.parse_args()

    run_phase6(
        n_samples  = args.n_samples,
        output_dir = args.output_dir,
        model_id   = args.model_id,
        layer_idx  = args.layer_idx,
    )
