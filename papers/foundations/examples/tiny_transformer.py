"""Small CPU-first decoder-only Transformer for shape and cache study.

This is a randomly initialized teaching model, not a trained language model.
It uses B=1, T=4, D=8, H=2, dh=4, Dff=32, vocab_size=16, L=2, learned absolute
positions, Pre-LN, LayerNorm, GELU, and dropout=0.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=8, n_heads=2):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, past_kv=None, past_len=0):
        batch, query_len, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(batch, query_len, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))  # [B,H,Tq,dh], new K/V [B,H,Tq,dh]
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        key_len = k.shape[2]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        query_positions = past_len + torch.arange(query_len, device=x.device)
        key_positions = torch.arange(key_len, device=x.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))
        weights = scores.softmax(dim=-1)  # normalize over visible history positions Tk
        context = weights @ v
        context = context.transpose(1, 2).contiguous().view(batch, query_len, width)
        return self.out(context), (k, v)


class DecoderBlock(nn.Module):
    def __init__(self, d_model=8, n_heads=2, d_ff=32):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, past_kv=None, past_len=0):
        attn_out, new_kv = self.attn(self.ln_attn(x), past_kv, past_len)
        x = x + attn_out
        x = x + self.ffn(self.ln_ffn(x))
        return x, new_kv


class TinyDecoderLM(nn.Module):
    def __init__(self, vocab_size=16, d_model=8, n_heads=2, d_ff=32,
                 n_layers=2, max_context=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_context = max_context
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_context, d_model)
        self.blocks = nn.ModuleList(
            DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, cache=None, use_cache=False):
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must have nonempty shape [B,Tq]")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must contain integer IDs")
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.vocab_size):
            raise ValueError("token ID outside vocabulary")
        if (use_cache or cache is not None) and self.training:
            raise ValueError("KV cache comparison and generation require eval() mode")
        if cache is not None and not use_cache:
            raise ValueError("pass use_cache=True when consuming a KV cache")

        if cache is None:
            past_len = 0
        else:
            if len(cache) != len(self.blocks):
                raise ValueError("cache must contain one K/V pair per layer")
            past_len = cache[0][0].shape[2]
            expected_batch = token_ids.shape[0]
            for key, value in cache:
                if key.shape != value.shape or key.shape[:2] != (expected_batch, self.n_heads):
                    raise ValueError("cache batch/head/feature shapes do not match")
                if key.shape[2] != past_len or key.shape[3] != self.d_head:
                    raise ValueError("cache lengths or head dimensions do not match")
        query_len = token_ids.shape[1]
        if past_len + query_len > self.max_context:
            raise ValueError("sequence exceeds learned position table")

        positions = past_len + torch.arange(query_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        new_cache = []
        for index, block in enumerate(self.blocks):
            past = None if cache is None else cache[index]
            x, layer_cache = block(x, past, past_len)
            new_cache.append(layer_cache)
        logits = self.lm_head(self.final_ln(x))  # [B,Tq,vocab_size]
        return logits, new_cache if use_cache else None


@torch.no_grad()
def greedy_generate(model, prompt, max_new_tokens=4):
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if max_new_tokens == 0:
        return prompt.clone()
    model.eval()
    logits, cache = model(prompt, use_cache=True)
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    result = torch.cat((prompt, next_id), dim=1)
    for _ in range(1, max_new_tokens):
        logits, cache = model(next_id, cache=cache, use_cache=True)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        result = torch.cat((result, next_id), dim=1)
    return result


def main():
    torch.manual_seed(7)
    model = TinyDecoderLM()
    tokens = torch.tensor([[2, 5, 7, 1, 9]], dtype=torch.long)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    assert inputs.shape == targets.shape == (1, 4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    model.train()
    first_loss = None
    for step in range(80):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    final_loss = loss.item()
    assert final_loss < first_loss, (first_loss, final_loss)
    print(f"loss: {first_loss:.4f} -> {final_loss:.4f}")

    model.eval()
    with torch.no_grad():
        full_logits, full_cache = model(inputs, use_cache=True)
        token_cache = None
        token_logits = []
        for index in range(inputs.shape[1]):
            part_logits, token_cache = model(
                inputs[:, index:index + 1], cache=token_cache, use_cache=True
            )
            token_logits.append(part_logits)
        torch.testing.assert_close(torch.cat(token_logits, dim=1), full_logits,
                                   rtol=1e-5, atol=1e-6)

        first, chunk_cache = model(inputs[:, :2], use_cache=True)
        second, chunk_cache = model(inputs[:, 2:], cache=chunk_cache, use_cache=True)
        torch.testing.assert_close(torch.cat((first, second), dim=1), full_logits,
                                   rtol=1e-5, atol=1e-6)
        assert len(full_cache) == len(model.blocks) == 2
        assert all(k.shape == v.shape == (1, 2, 4, 4) for k, v in full_cache)
        model.train()
        try:
            model(inputs[:, :1], cache=full_cache)
        except ValueError:
            pass
        else:
            raise AssertionError("consuming a cache in training mode must fail")
        model.eval()

        changed = inputs.clone()
        changed[:, 3] = (changed[:, 3] + 1) % model.vocab_size
        changed_logits, _ = model(changed)
        torch.testing.assert_close(full_logits[:, :3], changed_logits[:, :3],
                                   rtol=1e-5, atol=1e-6)

        # This probability distribution is over vocab_size, unlike attention over Tk.
        next_token_probabilities = full_logits[:, -1, :].softmax(dim=-1)
        assert next_token_probabilities.shape == (1, model.vocab_size)
        prompt = inputs[:, :2]
        generated = greedy_generate(model, prompt, max_new_tokens=4)
        assert torch.equal(greedy_generate(model, prompt, max_new_tokens=0), prompt)
        try:
            greedy_generate(model, prompt, max_new_tokens=-1)
        except ValueError:
            pass
        else:
            raise AssertionError("negative generation length must fail")

        custom = TinyDecoderLM(vocab_size=16, d_model=12, n_heads=3, d_ff=24,
                               n_layers=1, max_context=8).eval()
        _, custom_cache = custom(torch.tensor([[2, 5]]), use_cache=True)
        assert custom_cache[0][0].shape == (1, 3, 2, 4)
        _, custom_cache = custom(torch.tensor([[7]]), cache=custom_cache, use_cache=True)
        assert custom_cache[0][0].shape == (1, 3, 3, 4)

    print("logits:", tuple(full_logits.shape))
    print("attention/cache per layer:", [tuple(k.shape) for k, _ in full_cache])
    print("next-token probabilities:", tuple(next_token_probabilities.shape))
    print("greedy continuation:", generated.tolist())
    print("PASS: training gradients, causal prefix, full/token/chunk cache equality")


if __name__ == "__main__":
    main()
