"""NumPy semantic checks for the foundations examples; CPU math only."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    centered = x - np.mean(x)
    inv_std = 1.0 / np.sqrt(np.mean(centered * centered) + eps)
    return gamma * centered * inv_std + beta


def layer_norm_jacobian(x, gamma, eps=1e-5):
    width = x.size
    centered = x - np.mean(x)
    inv_std = 1.0 / np.sqrt(np.mean(centered * centered) + eps)
    identity_minus_mean = np.eye(width) - np.ones((width, width)) / width
    return gamma[:, None] * (
        identity_minus_mean * inv_std
        - np.outer(centered, centered) * (inv_std ** 3 / width)
    )


def validate_post_ln_gradient():
    x = np.array([0.2, -0.7, 1.1, 0.4], dtype=np.float64)
    matrix = np.array([
        [0.3, -0.2, 0.1, 0.0],
        [0.1, 0.4, -0.3, 0.2],
        [-0.2, 0.1, 0.2, 0.3],
        [0.0, -0.1, 0.2, 0.5],
    ])
    bias = np.array([0.1, -0.2, 0.05, 0.3])
    gamma = np.array([1.2, 0.8, 1.1, 0.9])
    beta = np.array([0.1, -0.1, 0.2, 0.0])
    upstream = np.array([0.7, -0.4, 0.2, 1.1])
    eps = 1e-5

    def objective(value):
        residual_sum = value + matrix @ value + bias
        return float(upstream @ layer_norm(residual_sum, gamma, beta, eps))

    residual_sum = x + matrix @ x + bias
    j_ln = layer_norm_jacobian(residual_sum, gamma, eps)
    analytic = (np.eye(x.size) + matrix.T) @ j_ln.T @ upstream
    step = 1e-6
    finite_diff = np.array([
        (objective(x + np.eye(x.size)[i] * step)
         - objective(x - np.eye(x.size)[i] * step)) / (2 * step)
        for i in range(x.size)
    ])
    error = float(np.max(np.abs(analytic - finite_diff)))
    assert error < 1e-7, error
    return error


def validate_logit_gradient():
    logits = np.array([0.2, -0.4, 1.1, 0.7], dtype=np.float64)
    target = 2

    def loss(value):
        return float(np.log(np.exp(value - value.max()).sum()) + value.max() - value[target])

    probabilities = softmax(logits)
    analytic = probabilities.copy()
    analytic[target] -= 1.0
    step = 1e-6
    finite_diff = np.array([
        (loss(logits + np.eye(logits.size)[i] * step)
         - loss(logits - np.eye(logits.size)[i] * step)) / (2 * step)
        for i in range(logits.size)
    ])
    error = float(np.max(np.abs(analytic - finite_diff)))
    assert error < 1e-7, error
    return error


def linear(x, weight, bias):
    # weight is [out,in], matching torch.nn.Linear and Y = X W^T + b.
    return x @ weight.T + bias


class NumpyTinyDecoder:
    def __init__(self, seed=17):
        self.batch, self.d_model, self.n_heads = 1, 8, 2
        self.d_head, self.d_ff, self.vocab, self.n_layers = 4, 32, 16, 2
        self.max_context = 64
        rng = np.random.default_rng(seed)

        def weight(out_dim, in_dim):
            return rng.normal(0.0, 0.08, (out_dim, in_dim))

        def bias(width):
            return np.zeros(width)

        self.token_embedding = rng.normal(0.0, 0.08, (self.vocab, self.d_model))
        self.position_embedding = rng.normal(0.0, 0.08, (self.max_context, self.d_model))
        self.final_norm = (np.ones(self.d_model), np.zeros(self.d_model))
        self.lm_head = (weight(self.vocab, self.d_model), bias(self.vocab))
        self.layers = []
        for _ in range(self.n_layers):
            self.layers.append({
                "ln1": (np.ones(self.d_model), np.zeros(self.d_model)),
                "qkv_w": weight(3 * self.d_model, self.d_model),
                "qkv_b": bias(3 * self.d_model),
                "out_w": weight(self.d_model, self.d_model),
                "out_b": bias(self.d_model),
                "ln2": (np.ones(self.d_model), np.zeros(self.d_model)),
                "ff1_w": weight(self.d_ff, self.d_model),
                "ff1_b": bias(self.d_ff),
                "ff2_w": weight(self.d_model, self.d_ff),
                "ff2_b": bias(self.d_model),
            })

    def forward(self, token_ids, cache=None):
        ids = np.asarray(token_ids)
        if ids.ndim != 2 or ids.shape[1] == 0 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("token IDs must be a nonempty integer [B,Tq] array")
        if np.any(ids < 0) or np.any(ids >= self.vocab):
            raise ValueError("token ID outside vocabulary")
        if ids.shape[0] != self.batch:
            raise ValueError("this toy reference fixes B=1")
        if cache is not None and len(cache) != self.n_layers:
            raise ValueError("cache must contain separate K/V arrays for each layer")
        past_len = 0 if cache is None else cache[0][0].shape[2]
        query_len = ids.shape[1]
        if past_len + query_len > self.max_context:
            raise ValueError("learned position table exceeded")

        positions = past_len + np.arange(query_len)
        x = self.token_embedding[ids] + self.position_embedding[positions][None, :, :]
        new_cache = []
        for layer_index, w in enumerate(self.layers):
            normalized = np.stack([
                layer_norm(row, *w["ln1"]) for row in x.reshape(-1, self.d_model)
            ]).reshape(x.shape)
            qkv = linear(normalized, w["qkv_w"], w["qkv_b"])
            q, k, v = np.split(qkv, 3, axis=-1)

            def heads(tensor):
                return tensor.reshape(self.batch, query_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

            q, k, v = map(heads, (q, k, v))
            if cache is not None:
                old_k, old_v = cache[layer_index]
                if old_k.shape != old_v.shape or old_k.shape[:2] != (self.batch, self.n_heads):
                    raise ValueError("invalid per-layer cache shape")
                k = np.concatenate((old_k, k), axis=2)
                v = np.concatenate((old_v, v), axis=2)
            key_len = k.shape[2]
            scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.d_head)
            query_positions = past_len + np.arange(query_len)
            key_positions = np.arange(key_len)
            allowed = key_positions[None, :] <= query_positions[:, None]
            scores = np.where(allowed[None, None, :, :], scores, -np.inf)
            weights = softmax(scores, axis=-1)  # attention softmax over Tk
            context = weights @ v
            context = context.transpose(0, 2, 1, 3).reshape(self.batch, query_len, self.d_model)
            x = x + linear(context, w["out_w"], w["out_b"])

            normalized = np.stack([
                layer_norm(row, *w["ln2"]) for row in x.reshape(-1, self.d_model)
            ]).reshape(x.shape)
            hidden = linear(normalized, w["ff1_w"], w["ff1_b"])
            hidden = np.vectorize(math.erf)(hidden / math.sqrt(2.0))
            hidden = 0.5 * linear(normalized, w["ff1_w"], w["ff1_b"]) * (1.0 + hidden)
            x = x + linear(hidden, w["ff2_w"], w["ff2_b"])
            new_cache.append((k, v))

        normalized = np.stack([
            layer_norm(row, *self.final_norm) for row in x.reshape(-1, self.d_model)
        ]).reshape(x.shape)
        logits = linear(normalized, *self.lm_head)  # logits softmax, if needed, is over V
        return logits, new_cache


def check_source_sync():
    pattern = re.compile(r"<!-- source-check: ([^>]+) -->\s*~~~python\n(.*?)\n~~~", re.S)
    checked = 0
    for markdown in ROOT.glob("*.md"):
        content = markdown.read_text(encoding="utf-8")
        for relative_path, snippet in pattern.findall(content):
            source_path = (markdown.parent / relative_path).resolve()
            source = source_path.read_text(encoding="utf-8")
            if snippet + "\n" != source:
                raise AssertionError(f"source-check mismatch: {markdown.name} -> {relative_path}")
            compile(source, str(source_path), "exec")
            checked += 1
    if checked == 0:
        raise AssertionError("no source-check blocks found")
    return checked


def main():
    source_blocks = check_source_sync()
    model = NumpyTinyDecoder()
    tokens = np.array([[2, 5, 7, 1, 9]], dtype=np.int64)
    inputs = tokens[:, :-1]
    full_logits, full_cache = model.forward(inputs)

    token_cache = None
    token_logits = []
    for index in range(inputs.shape[1]):
        logits, token_cache = model.forward(inputs[:, index:index + 1], token_cache)
        token_logits.append(logits)
    token_logits = np.concatenate(token_logits, axis=1)
    np.testing.assert_allclose(token_logits, full_logits, rtol=1e-10, atol=1e-10)

    first, chunk_cache = model.forward(inputs[:, :2])
    second, chunk_cache = model.forward(inputs[:, 2:], chunk_cache)
    np.testing.assert_allclose(np.concatenate((first, second), axis=1), full_logits,
                               rtol=1e-10, atol=1e-10)
    assert len(full_cache) == model.n_layers
    assert all(k.shape == v.shape == (1, 2, 4, 4) for k, v in full_cache)
    assert all(full_cache[i][0] is not full_cache[j][0]
               for i in range(model.n_layers) for j in range(i + 1, model.n_layers))

    changed = inputs.copy()
    changed[:, 3] = (changed[:, 3] + 1) % model.vocab
    changed_logits, _ = model.forward(changed)
    np.testing.assert_allclose(changed_logits[:, :3], full_logits[:, :3],
                               rtol=1e-10, atol=1e-10)

    post_ln_error = validate_post_ln_gradient()
    logit_error = validate_logit_gradient()
    print(f"PASS: {source_blocks} source block(s) match and compile")
    print("PASS: NumPy full/token/chunk cache equality, causal prefix, per-layer cache shapes")
    print(f"PASS: Post-LN chain-rule vs finite difference max error {post_ln_error:.3e}")
    print(f"PASS: cross-entropy logit gradient vs finite difference max error {logit_error:.3e}")
    print("CPU NumPy mathematical checks only; this does not establish PyTorch runtime or GPU behavior.")


if __name__ == "__main__":
    main()
