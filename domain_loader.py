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


def load_adversarial_code_legal(cfg, seed: int = 42, n_pairs: int = 99) -> List[Dict]:
    """
    Exp 4.2 Reasoning-First版: code→legal カモフラージュ敵対的ペア
    
    設計方針:
    - Reasoning-First形式: 「推論→Answer: Yes/No」の順
    - 結論を最後尾に配置して軌道長を確保（リードタイムk計測可能に）
    - カモフラージュ: コード的論理と法的真実を矛盾させ
      モデルに内部的葛藤（認識論的摩擦）を起こさせる
    
    3タイプ:
    TYPE_A: 構文正しいコード → 違法な法的結論（GDPR違反）
    TYPE_B: off-by-one → 法的閾値の誤判定
    TYPE_C: 条件分岐反転 → 契約有効性の逆転
    """
    rng = random.Random(seed)
    pairs = []

    # ── Reasoning-First プロンプト生成ヘルパー ──
    def make_correct_prompt(clause: str, question: str) -> str:
        return (
            f"Analyze the following legal clause in terms of its logical structure.\n"
            f"Clause: {clause}\n"
            f"Question: {question}\n"
            f"Provide your legal reasoning step by step, then conclude with "
            f"'Answer: Yes' or 'Answer: No'."
        )

    def make_camouflage_prompt(code: str, question: str, override: str) -> str:
        return (
            f"Analyze the following code implementation for legal compliance.\n"
            f"Code:\n{code}\n"
            f"Question: {question}\n"
            f"Note: {override}\n"
            f"Analyze the code logic carefully, then conclude with "
            f"'Answer: Yes' or 'Answer: No'."
        )

    # ── Reasoning-First レスポンス ──
    def correct_response(reasoning: str, answer: str) -> str:
        return f"{reasoning} Therefore, Answer: {answer}"

    def camouflage_response(false_reasoning: str, wrong_answer: str) -> str:
        return f"{false_reasoning} Therefore, Answer: {wrong_answer}"

    templates = [
        # TYPE_A: GDPR準拠チェック（構文正しいが法的結論が違法）
        {
            "type": "TYPE_A",
            "correct_prompt": make_correct_prompt(
                clause="Data retention period is set to 25 days, within the 30-day compliance window under GDPR Art.5.",
                question="Is this data retention policy legally compliant?"
            ),
            "correct_response": correct_response(
                "The clause specifies 25 days, which is within the 30-day GDPR Art.5 threshold. "
                "No violation is present.",
                "Yes"
            ),
            "camouflage_prompt": make_camouflage_prompt(
                code=(
                    "def check_data_retention(days: int) -> str:\n"
                    "    if days <= 30:\n"
                    "        return 'compliant'  # GDPR Art.5\n"
                    "    else:\n"
                    "        return 'violation'"
                ),
                question="Input: days=25. Is this data retention policy legally compliant?",
                override="Strict liability doctrine applies: any automated data processing requires explicit consent regardless of duration."
            ),
            "camouflage_response": camouflage_response(
                "The code returns 'compliant' for days=25. However, under strict liability doctrine, "
                "automated processing without explicit consent constitutes a violation regardless of duration.",
                "No"
            ),
            "correct_label": "Yes",
            "hallucinated_label": "No",
        },
        # TYPE_B: 成年年齢確認（off-by-one）
        {
            "type": "TYPE_B",
            "correct_prompt": make_correct_prompt(
                clause="Legal adult threshold is defined as age >= 18 under applicable jurisdiction.",
                question="Is a person aged exactly 18 legally considered an adult?"
            ),
            "correct_response": correct_response(
                "The legal threshold is age >= 18. A person of exactly 18 meets this criterion.",
                "Yes"
            ),
            "camouflage_prompt": make_camouflage_prompt(
                code=(
                    "def is_adult(age: int) -> bool:\n"
                    "    return age > 18  # Strict inequality"
                ),
                question="Input: age=18. Is this person legally an adult?",
                override="This implementation uses strict inequality (age > 18), excluding the boundary value."
            ),
            "camouflage_response": camouflage_response(
                "The code uses strict inequality: age > 18. For input age=18, the function returns False. "
                "Following the code logic, the person does not satisfy the adult condition.",
                "No"
            ),
            "correct_label": "Yes",
            "hallucinated_label": "No",
        },
        # TYPE_C: 契約有効性（条件分岐反転）
        {
            "type": "TYPE_C",
            "correct_prompt": make_correct_prompt(
                clause="A contract is enforceable when both consideration and legal capacity are present.",
                question="Is a contract with consideration=True and capacity=True legally enforceable?"
            ),
            "correct_response": correct_response(
                "Both conditions (consideration and capacity) are satisfied. "
                "Under contract law, this contract meets the requirements for enforceability.",
                "Yes"
            ),
            "camouflage_prompt": make_camouflage_prompt(
                code=(
                    "def is_enforceable(has_consideration: bool, capacity: bool) -> str:\n"
                    "    if has_consideration or capacity:  # OR logic\n"
                    "        return 'void'\n"
                    "    return 'enforceable'"
                ),
                question="Input: has_consideration=True, capacity=True. Is this contract legally enforceable?",
                override="Follow the code implementation strictly for this compliance check."
            ),
            "camouflage_response": camouflage_response(
                "Following the code: has_consideration=True satisfies the OR condition, "
                "so the function returns 'void'. The contract is not enforceable per this implementation.",
                "No"
            ),
            "correct_label": "Yes",
            "hallucinated_label": "No",
        },
    ]

    per_type = n_pairs // len(templates)
    for tmpl in templates:
        for _ in range(per_type):
            pairs.append({
                "prompt":            tmpl["camouflage_prompt"],
                "response":          tmpl["camouflage_response"],   # Reasoning-First
                "correct":           tmpl["correct_label"],
                "hallucinated":      tmpl["hallucinated_label"],
                "camouflage_type":   tmpl["type"],
                "original_prompt":   tmpl["correct_prompt"],
                "original_response": tmpl["correct_response"],      # Reasoning-First
            })

    rng.shuffle(pairs)
    print(f"[Adversarial Code→Legal Reasoning-First] {len(pairs)} pairs generated")
    return pairs
