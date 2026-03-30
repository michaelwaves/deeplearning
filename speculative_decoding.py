import time
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DRAFT_MODEL_NAME = "gpt2"
TARGET_MODEL_NAME = "gpt2-large"
DEVICE = "cuda"
DTYPE = torch.float16
K = 5  # draft tokens per speculation step
MAX_NEW_TOKENS = 100
TEMPERATURE = 1.0
TOP_P = 0.9


def load_models():
    tok = GPT2Tokenizer.from_pretrained(TARGET_MODEL_NAME)
    tok.pad_token = tok.eos_token
    draft = GPT2LMHeadModel.from_pretrained(
        DRAFT_MODEL_NAME, torch_dtype=DTYPE).to(DEVICE).eval()
    target = GPT2LMHeadModel.from_pretrained(
        TARGET_MODEL_NAME, torch_dtype=DTYPE).to(DEVICE).eval()
    print("Models loaded.")
    return tok, draft, target


def sample_token(logits: torch.Tensor, temperature, top_p: float):
    logits = logits/temperature

    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_logits[cumulative_probs -
                  F.softmax(sorted_logits, dim=-1) > top_p] = -float("inf")
    logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def get_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids)
    return out.logits[:, -1, :]


@torch.no_grad()
def get_all_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids)
    return out.logits


@torch.no_grad()
def speculative_decode(
    draft_model, target_model, tokenizer, prompt_ids,
    max_new_tokens: int = MAX_NEW_TOKENS, k: int = K,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P

) -> tuple[torch.Tensor, dict]:
    ids = prompt_ids.clone().to(DEVICE)
    total_draft_tokens = 0
    total_accepted = 0
    target_model_calls = 0
    generated = 0

    while generated < max_new_tokens:
        remaining = max_new_tokens-generated
        k_this_step = min(k, remaining)
        draft_tokens = []
        draft_probs = []

        cur_ids = ids
        for _ in range(k_this_step):
            logits = get_logits(draft_model, cur_ids)
            probs = F.softmax(logits/temperature, dim=-1)
            token = sample_token(logits, temperature, top_p)
            draft_tokens.append(token)
            draft_probs.append(probs[0, token[0, 0]])
            cur_ids = torch.cat([cur_ids, token], dim=1)

        ids_with_draft = cur_ids
        target_logits = get_all_logits(target_model, ids_with_draft)
        target_model_calls += 1

        prompt_len = ids.shape[1]
        verify_start = prompt_len-1

        accepted = 0
        for i in range(k_this_step):
            t_logits = target_logits[0, verify_start+i, :]
            t_probs = F.softmax(t_logits/temperature, dim=-1)
            token_id = draft_tokens[i][0, 0]
            p_target = t_probs[token_id].item()
            p_draft = draft_probs[i].item()

            accept_prob = min(1.0, p_target / (p_draft + 1e-9))
            u = torch.rand(1).item()
            if u < accept_prob:
                ids = torch.cat([ids, draft_tokens[i]], dim=1)
                accepted += 1
                generated += 1
                if generated >= max_new_tokens:
                    break
            else:
                adjusted = F.relu(t_probs - torch.tensor(p_draft, device=DEVICE) *
                                  F.one_hot(token_id, num_classes=t_probs.shape[0]).float().to(DEVICE))
                if adjusted.sum() < 1e-9:
                    adjusted = t_probs
                adjusted = adjusted / adjusted.sum()
                corrected = torch.multinomial(
                    adjusted, num_samples=1).unsqueeze(0)
                ids = torch.cat([ids, corrected], dim=1)
                generated += 1
                break
        total_draft_tokens += k_this_step
        total_accepted += accepted

        # If we accepted all k draft tokens, sample one bonus token from target
        if accepted == k_this_step and generated < max_new_tokens:
            bonus_logits = target_logits[0, verify_start + k_this_step, :]
            bonus_token = sample_token(
                bonus_logits, temperature, top_p).unsqueeze(0)
            ids = torch.cat([ids, bonus_token], dim=1)
            generated += 1

    stats = {
        "acceptance_rate": total_accepted / max(total_draft_tokens, 1),
        "target_model_calls": target_model_calls,
        "total_new_tokens": generated,
    }
    return ids, stats


@torch.no_grad()
def standard_decode(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> tuple[torch.Tensor, dict]:
    """Baseline: autoregressive decoding with the target model only."""
    ids = prompt_ids.clone().to(DEVICE)
    calls = 0
    for _ in range(max_new_tokens):
        logits = get_logits(model, ids)
        token = sample_token(logits, temperature, top_p)
        ids = torch.cat([ids, token], dim=1)
        calls += 1
    return ids, {"target_model_calls": calls}


if __name__ == "__main__":
    tokenizer, draft_model, target_model = load_models()

    prompt = "The theory of relativity states that"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    spec_ids, spec_stats = speculative_decode(
        draft_model, target_model, tokenizer, input_ids)
    torch.cuda.synchronize()
    spec_time = time.perf_counter() - t0

    spec_text = tokenizer.decode(spec_ids[0], skip_special_tokens=True)

    print("\n[Speculative Decoding]")
    print(f"  Output            : {spec_text}")
    print(f"  Time              : {spec_time:.2f}s")
    print(f"  Target fwd passes : {spec_stats['target_model_calls']}")
    print(f"  Acceptance rate   : {spec_stats['acceptance_rate']:.1%}")
    print(
        f"  Tokens/sec        : {spec_stats['total_new_tokens'] / spec_time:.1f}")

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    std_ids, std_stats = standard_decode(target_model, input_ids)
    torch.cuda.synchronize()
    std_time = time.perf_counter()-t0

    std_text = tokenizer.decode(std_ids[0], skip_special_tokens=True)
    print("\n[Standard Decoding  (target model only)]")
    print(f"  Output            : {std_text}")
    print(f"  Time              : {std_time:.2f}s")
    print(f"  Target fwd passes : {std_stats['target_model_calls']}")
    print(f"  Tokens/sec        : {MAX_NEW_TOKENS / std_time:.1f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    speedup = std_time / spec_time
    print("\n" + "=" * 60)
    print(f"  Speedup           : {speedup:.2f}×")
    print("=" * 60)
