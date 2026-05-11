"""
domain_loader.py — AIIE Phase 4: 4ドメインのデータロードと
                   correct/failed ペア生成
Gemmina Intelligence LLC / Pure Information Laboratory
2026-05-11

各ドメインの出力フォーマット（Phase 1 pkl と互換）:
    {
        "prompt":        str,   # モデルへの入力プロンプト
        "correct":       str,   # 正解応答テキスト
        "hallucinated":  str,   # ハルシネーション応答テキスト
        "domain":        str,   # ドメイン名
        "meta":          dict,  # ドメイン固有メタデータ
    }
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict


# ================================================================
# 1. Medical — MedQA (USMLE 4択)
# ================================================================

def load_medical(cfg, seed: int = 42) -> List[Dict]:
    """
    MedQA から N=cfg.n_pairs のペアを生成。

    ハルシネーション定義:
        正解選択肢（answer_idx）から最も遠い誤答を強制応答として使用。
        距離は選択肢インデックスの差（循環）で近似。
    """
    from datasets import load_dataset
    rng = random.Random(seed)

    ds = load_dataset(cfg.medqa_dataset, split=cfg.medqa_split)
    pairs = []

    for item in ds:
        options  = item["options"]          # {"A": "...", "B": "...", ...}
        answer   = item["answer_idx"]       # 正解キー: "A"/"B"/"C"/"D"
        question = item["question"]

        keys = list(options.keys())
        if answer not in keys or len(keys) < 2:
            continue

        # 正解から最も遠い誤答を選択（循環距離最大）
        correct_idx  = keys.index(answer)
        wrong_idx    = (correct_idx + len(keys) // 2) % len(keys)
        wrong_key    = keys[wrong_idx]

        prompt = (
            f"Answer the following multiple-choice question.\n"
            f"Question: {question}\n"
            f"Options:\n"
            + "\n".join(f"  {k}: {options[k]}" for k in keys)
            + f"\nAnswer:"
        )

        pairs.append({
            "prompt":       prompt,
            "correct":      f"{answer}: {options[answer]}",
            "hallucinated": f"{wrong_key}: {options[wrong_key]}",
            "domain":       "medical",
            "meta":         {"question": question, "correct_key": answer},
        })

        if len(pairs) >= cfg.n_pairs:
            break

    rng.shuffle(pairs)
    print(f"[Medical] {len(pairs)} pairs loaded")
    return pairs[:cfg.n_pairs]


# ================================================================
# 2. Legal — LegalBench / contract_nli
# ================================================================

def load_legal(cfg, seed: int = 42) -> List[Dict]:
    """
    LegalBench (contract_nli) から N ペアを生成。

    ハルシネーション定義:
        正解ラベル "entailment" / "not_entailment" を反転。
        プロンプトに「Answer: <label>」を強制注入。
    """
    from datasets import load_dataset
    rng = random.Random(seed)

    ds = load_dataset(cfg.legal_dataset, cfg.legal_config, split=cfg.legal_split)
    pairs = []

    label_map = {
        "Yes": "No",
        "No":  "Yes",
        "entailment":     "not_entailment",
        "not_entailment": "entailment",
    }

    for item in ds:
        text  = item.get("text", item.get("sentence", ""))
        label = str(item.get("label", item.get("answer", "")))

        if not text or label not in label_map:
            continue

        wrong_label = label_map[label]

        prompt = (
            f"Determine whether the following legal clause entails the hypothesis.\n"
            f"{text}\n"
            f"Answer (Yes/No):"
        )

        pairs.append({
            "prompt":       prompt,
            "correct":      label,
            "hallucinated": wrong_label,
            "domain":       "legal",
            "meta":         {"original_label": label},
        })

        if len(pairs) >= cfg.n_pairs:
            break

    rng.shuffle(pairs)
    print(f"[Legal] {len(pairs)} pairs loaded")
    return pairs[:cfg.n_pairs]


# ================================================================
# 3. Code — HumanEval
# ================================================================

def load_code(cfg, seed: int = 42) -> List[Dict]:
    """
    HumanEval から N ペアを生成。

    ハルシネーション定義:
        canonical_solution に典型的バグパターンを注入。
        優先度: off-by-one → None return → 符号反転

    注意: HumanEval は 164 問しかないため n_pairs は 164 上限。
    """
    from datasets import load_dataset
    rng = random.Random(seed)

    ds = load_dataset(cfg.code_dataset, split=cfg.code_split)
    pairs = []

    for item in ds:
        prompt   = item["prompt"]
        solution = item["canonical_solution"]

        buggy = _inject_bug(solution, rng)
        if buggy is None:
            continue

        pairs.append({
            "prompt":       prompt,
            "correct":      solution,
            "hallucinated": buggy,
            "domain":       "code",
            "meta":         {"task_id": item["task_id"]},
        })

    rng.shuffle(pairs)
    print(f"[Code] {len(pairs)} pairs loaded (HumanEval max=164)")
    return pairs


def _inject_bug(solution: str, rng: random.Random) -> str:
    """
    3種のバグパターンを試みて、最初に成功したものを返す。
    いずれも失敗なら None。
    """
    # パターン1: return None（最初の return 文を置換）
    if "return " in solution and "return None" not in solution:
        lines = solution.split("\n")
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("return ") and stripped != "return None":
                indent = len(line) - len(stripped)
                lines[i] = " " * indent + "return None"
                return "\n".join(lines)

    # パターン2: off-by-one（range の終端を -1）
    if re.search(r'range\(\s*\w+\s*\)', solution):
        buggy = re.sub(
            r'range\(\s*(\w+)\s*\)',
            lambda m: f"range({m.group(1)} - 1)",
            solution, count=1,
        )
        if buggy != solution:
            return buggy

    # パターン3: 比較演算子反転（< → <=, > → >=）
    for op, wrong in [(" < ", " <= "), (" > ", " >= ")]:
        if op in solution:
            return solution.replace(op, wrong, 1)

    return None


# ================================================================
# 4. Conversation — ShareGPT
# ================================================================

class DomainDataSampler:
    """
    ShareGPT から事実ベースの脆弱ペアを抽出。
    （ユーザー提供コードをそのまま統合）
    """
    def __init__(self):
        self.fact_prefixes   = ("What year", "Who", "How many", "When", "Where")
        self.entity_pattern  = re.compile(r'([A-Z][a-z]+|\d{4}|[\d,]+)')

    def is_geometrically_vulnerable(self, conv: Dict) -> bool:
        if not conv.get("conversations") or len(conv["conversations"]) < 2:
            return False
        if "from" not in conv["conversations"][0]:
            return False
        if conv["conversations"][0]["from"] != "human":
            return False
        q = conv["conversations"][0]["value"].strip()
        a = conv["conversations"][1]["value"].strip()
        return (
            q.startswith(self.fact_prefixes)
            and bool(self.entity_pattern.search(a))
            and len(a.split()) <= 100
        )

    def generate_hallucination(self, answer: str, rng: random.Random) -> str:
        """
        数値を±1〜10ずらすか、固有名詞末尾に 'Jr.' を付加して摂動。
        本番実装ではLLMに置換させることを推奨（config参照）。
        """
        # 数値摂動
        def perturb_number(m):
            n = int(m.group().replace(",", ""))
            delta = rng.choice([-10, -5, -1, 1, 5, 10])
            return str(abs(n + delta))

        perturbed = re.sub(r'\b\d{4}\b', perturb_number, answer)
        if perturbed != answer:
            return perturbed

        perturbed = re.sub(r'\b\d+\b', perturb_number, answer, count=1)
        if perturbed != answer:
            return perturbed

        # 固有名詞摂動（最初の大文字語の後ろに "II" を追加）
        perturbed = re.sub(
            r'\b([A-Z][a-z]+)\b',
            lambda m: m.group() + " II",
            answer, count=1,
        )
        return perturbed if perturbed != answer else "[UNKNOWN] " + answer

    def process_sharegpt(
        self, input_path: str, n_limit: int = 500, seed: int = 42
    ) -> List[Dict]:
        rng = random.Random(seed)
        with open(input_path, "r") as f:
            raw_data = json.load(f)

        vulnerable = [s for s in raw_data if self.is_geometrically_vulnerable(s)]
        selected   = rng.sample(vulnerable, min(len(vulnerable), n_limit))

        print(
            f"[Conversation] Total: {len(raw_data)} | "
            f"Filtered: {len(vulnerable)} | Sampled: {len(selected)}"
        )
        return selected


def load_conversation(cfg, seed: int = 42) -> List[Dict]:
    """ShareGPT から N ペアを生成"""
    rng = random.Random(seed)
    sampler = DomainDataSampler()

    sharegpt_path = Path(cfg.sharegpt_path)
    if not sharegpt_path.exists():
        raise FileNotFoundError(
            f"ShareGPT が見つかりません: {sharegpt_path}\n"
            f"https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered"
            f" からダウンロードしてください"
        )

    raw_samples = sampler.process_sharegpt(str(sharegpt_path), cfg.n_pairs, seed)

    pairs = []
    for s in raw_samples:
        q = s["conversations"][0]["value"].strip()
        a = s["conversations"][1]["value"].strip()
        h = sampler.generate_hallucination(a, rng)

        pairs.append({
            "prompt":       f"Question: {q}\nAnswer:",
            "correct":      a,
            "hallucinated": h,
            "domain":       "conversation",
            "meta":         {"source": "sharegpt"},
        })

    print(f"[Conversation] {len(pairs)} pairs built")
    return pairs


# ================================================================
# 統合ローダー
# ================================================================

DOMAIN_LOADERS = {
    "medical":      load_medical,
    "legal":        load_legal,
    "code":         load_code,
    "conversation": load_conversation,
}


def load_all_domains(cfg) -> Dict[str, List[Dict]]:
    """全4ドメインを一括ロード"""
    all_data = {}
    for domain, loader in DOMAIN_LOADERS.items():
        print(f"\n--- Loading: {domain} ---")
        all_data[domain] = loader(cfg, seed=cfg.seed)
    return all_data
