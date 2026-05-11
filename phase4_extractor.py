"""
phase4_extractor.py — AIIE Phase 4: Hidden State 抽出
Phase 1 の hidden_state_extractor.py を4ドメイン対応に拡張。
Gemmina Intelligence LLC / Pure Information Laboratory
2026-05-11

出力 pkl フォーマット（Phase 1/3 と完全互換）:
    [
        {
            "correct": {
                "hidden":   {7: np.ndarray(T, D)},
                "logprobs": [float, ...]
            },
            "failed": {
                "hidden":   {7: np.ndarray(T, D)},
                "logprobs": [float, ...]
            },
            "domain":   str,
            "prompt":   str,
        },
        ...
    ]
"""

import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(cfg):
    """Llama-3-8B-Instruct をロード"""
    print(f"[Model] Loading {cfg.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="./hf_cache",
    )
    model.eval()
    print(f"[Model] Loaded. Layers: {model.config.num_hidden_layers}")
    return model, tokenizer


def extract_hidden_states_for_response(
    model,
    tokenizer,
    prompt: str,
    response: str,
    target_layer: int,
    device: str = "cuda",
    max_length: int = 512,
) -> Dict:
    """
    Teacher Forcing: prompt + response を連結してフォワードパスし、
    response トークン部分の hidden states を取得する。

    Returns:
        {
            "hidden":   {target_layer: np.ndarray(T_resp, D)},
            "logprobs": [float, ...]  (各応答トークンの対数確率)
        }
    """
    full_text = prompt + " " + response

    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    prompt_len = prompt_inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
        )

    # hidden states: tuple of (batch, seq, dim) per layer
    # response トークン部分のみ抽出
    resp_hidden = outputs.hidden_states[target_layer + 1]  # +1: embedding層をスキップ
    resp_hidden = resp_hidden[0, prompt_len:, :].cpu().float().numpy()

    # log probabilities
    logits    = outputs.logits[0, prompt_len - 1:-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    resp_ids  = inputs["input_ids"][0, prompt_len:]
    token_logprobs = [
        log_probs[i, resp_ids[i]].item()
        for i in range(min(len(resp_ids), log_probs.shape[0]))
    ]

    return {
        "hidden":   {target_layer: resp_hidden},
        "logprobs": token_logprobs,
    }


def extract_domain(
    model,
    tokenizer,
    domain_pairs: List[Dict],
    cfg,
    domain_name: str,
) -> List[Dict]:
    """
    1ドメイン分の hidden states を抽出して pkl に保存。

    Returns:
        Phase 1/3 と互換の results リスト
    """
    results = []
    errors  = 0

    for pair in tqdm(domain_pairs, desc=f"Extracting [{domain_name}]"):
        try:
            correct_states = extract_hidden_states_for_response(
                model, tokenizer,
                prompt   = pair["prompt"],
                response = pair["correct"],
                target_layer = cfg.target_layer,
                device   = cfg.device,
            )
            failed_states = extract_hidden_states_for_response(
                model, tokenizer,
                prompt   = pair["prompt"],
                response = pair["hallucinated"],
                target_layer = cfg.target_layer,
                device   = cfg.device,
            )

            # hidden states が空でないことを確認
            if (correct_states["hidden"][cfg.target_layer].shape[0] == 0 or
                    failed_states["hidden"][cfg.target_layer].shape[0] == 0):
                errors += 1
                continue

            results.append({
                "correct": correct_states,
                "failed":  failed_states,
                "domain":  domain_name,
                "prompt":  pair["prompt"],
            })

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [warn] {domain_name} extraction error: {e}")

    print(f"[{domain_name}] {len(results)} extracted, {errors} errors")

    # pkl 保存
    pkl_path = Path(cfg.pkl_dir) / f"{domain_name}_extraction_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[{domain_name}] Saved → {pkl_path}")

    return results


def run_extraction(cfg, domain_data: Dict[str, List]) -> Dict[str, List]:
    """全4ドメインの抽出を実行"""
    model, tokenizer = load_model(cfg)

    all_results = {}
    for domain_name, pairs in domain_data.items():
        pkl_path = Path(cfg.pkl_dir) / f"{domain_name}_extraction_results.pkl"

        # 既存 pkl があればスキップ
        if pkl_path.exists():
            print(f"[{domain_name}] pkl found, loading ...")
            with open(pkl_path, "rb") as f:
                all_results[domain_name] = pickle.load(f)
            print(f"  → {len(all_results[domain_name])} samples")
            continue

        all_results[domain_name] = extract_domain(
            model, tokenizer, pairs, cfg, domain_name
        )

    return all_results
