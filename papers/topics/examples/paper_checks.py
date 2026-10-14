"""Small NumPy checks mapped to the 14 papers/topics articles.

These are arithmetic/semantic checks for the in-page derivations, not model
training, framework, GPU-kernel, accuracy, or performance tests.
The optional --torch flag runs a focused CPU check for the DPO code path.
"""
from math import exp, isclose, log
import sys

import numpy as np


def check_47_mamba_zoh_step_and_scan():
    a, b, h_prev, x, delta = -1.0, 1.0, 3.0, 2.0, log(2.0)
    a_bar = exp(delta * a)
    b_bar_zoh = (exp(delta * a) - 1.0) / a * b
    h_zoh = a_bar * h_prev + b_bar_zoh * x
    h_author_step = a_bar * h_prev + delta * b * x
    assert isclose(h_zoh, 2.5, abs_tol=1e-12)
    assert isclose(h_author_step, 1.5 + 2.0 * log(2.0), abs_tol=1e-12)

    def compose(right, left):
        ar, br = right
        al, bl = left
        return ar * al, ar * bl + br

    u = [(0.5, 1.0), (0.25, -2.0), (0.8, 0.5)]
    lhs = compose(compose(u[2], u[1]), u[0])
    rhs = compose(u[2], compose(u[1], u[0]))
    assert np.allclose(lhs, rhs)


def check_48_masked_ce_gradient_and_accumulation():
    logits = np.array([[[log(3.0), 0.0], [0.0, 1.0]]])
    labels = np.array([[0, -100]])
    mask = np.array([[1, 0]])
    hidden = np.array([[[2.0], [7.0]]])
    valid = mask.astype(bool)
    safe_labels = np.where(valid, labels, 0)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    rows = np.indices(labels.shape)
    nll = -logp[rows[0], rows[1], safe_labels] * valid
    assert valid.sum() == 1 and isclose(float(nll.sum()), log(4.0 / 3.0), abs_tol=1e-12)
    grad = np.exp(logp)
    grad[rows[0], rows[1], safe_labels] -= 1.0
    grad *= valid[..., None]
    dW = hidden.reshape(-1, 1).T @ grad.reshape(-1, 2) / valid.sum()
    assert np.allclose(dW, [[-0.5, 0.5]])

    summed_a, tokens_a = np.array([2.0, 0.0]), 2
    summed_b, tokens_b = np.array([0.0, 3.0]), 1
    accumulated = (summed_a + summed_b) / (tokens_a + tokens_b)
    assert np.allclose(accumulated, [2.0 / 3.0, 1.0])
    try:
        if mask.sum() == 0:
            raise ValueError("no valid target tokens")
    except ValueError:
        raise AssertionError("the check batch must contain a valid token")


def check_49_lora_merge_and_initial_gradients():
    x = np.array([[1.0, 2.0]])
    W0 = np.array([[0.2, 0.1], [-0.3, 0.4]])
    A = np.array([[1.0, -1.0]])
    B = np.zeros((2, 1))
    scale = 1.0
    y = x @ W0.T + scale * (x @ A.T) @ B.T
    assert np.allclose(y, x @ W0.T)
    upstream = np.array([[1.0, -2.0]])
    grad_B = scale * upstream.T @ (x @ A.T)
    grad_A = scale * (upstream @ B).T @ x
    assert np.allclose(grad_B, [[-1.0], [2.0]])
    assert np.allclose(grad_A, 0.0)
    B = np.array([[0.1], [-0.2]])
    y_bypass = x @ W0.T + scale * (x @ A.T) @ B.T
    y_merged = x @ (W0 + scale * B @ A).T
    assert np.allclose(y_bypass, y_merged)


def check_50_dpo_masks_and_group_advantage():
    labels = np.array([[1, -100]])
    mask = np.array([[1, 0]], dtype=bool)
    safe = np.where(mask, labels, 0)
    logits = np.array([[[0.0, log(3.0)], [log(2.0), 0.0]]])
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    gathered = logp[0, np.arange(2), safe[0]]
    sequence_logp = (gathered * mask[0]).sum()
    assert isclose(float(sequence_logp), log(0.75), abs_tol=1e-12)

    rewards = np.array([1.0, 0.0, -1.0])
    advantages = (rewards - rewards.mean()) / rewards.std()
    assert np.allclose(advantages, [1.22474487139, 0.0, -1.22474487139])
    ratio, eps, adv = 1.3, 0.2, advantages[0]
    clipped_obj = min(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)
    assert isclose(clipped_obj, 1.2 * adv, abs_tol=1e-12)


def check_50_dpo_torch_core():
    import torch

    def sequence_logp(logits, labels, answer_mask):
        target = labels[:, 1:]
        mask = answer_mask[:, 1:].bool()
        safe_target = target.masked_fill(~mask, 0)
        token_logp = logits[:, :-1].log_softmax(-1).gather(
            -1, safe_target[..., None]
        ).squeeze(-1)
        return (token_logp * mask).sum(-1)

    labels_w = torch.tensor([[-100, 2, 1, -100]])
    labels_l = torch.tensor([[-100, 1, 3, -100]])
    answer_mask = torch.tensor([[0, 1, 1, 0]])
    policy_logits_w = torch.tensor(
        [[[0.0, 0.2, 1.3, -0.4], [0.1, 1.1, -0.2, 0.0],
          [0.0, -0.3, 0.5, 1.0], [0.3, 0.2, 0.1, 0.0]]], requires_grad=True)
    policy_logits_l = torch.tensor(
        [[[0.1, 1.2, 0.0, -0.2], [0.3, -0.2, 0.8, 0.5],
          [0.0, 0.6, 0.2, 0.9], [0.3, 0.2, 0.1, 0.0]]], requires_grad=True)
    ref_logits_w = torch.tensor(
        [[[0.0, 0.1, 1.0, -0.2], [0.2, 0.9, -0.1, 0.0],
          [0.0, -0.1, 0.4, 0.8], [0.3, 0.2, 0.1, 0.0]]], requires_grad=True)
    ref_logits_l = torch.tensor(
        [[[0.2, 0.9, 0.1, -0.1], [0.2, -0.1, 0.6, 0.4],
          [0.1, 0.4, 0.0, 0.7], [0.3, 0.2, 0.1, 0.0]]], requires_grad=True)

    policy_w = sequence_logp(policy_logits_w, labels_w, answer_mask)
    policy_l = sequence_logp(policy_logits_l, labels_l, answer_mask)
    with torch.no_grad():
        ref_w = sequence_logp(ref_logits_w, labels_w, answer_mask)
        ref_l = sequence_logp(ref_logits_l, labels_l, answer_mask)
    assert not torch.allclose(policy_w, policy_l)
    loss = -torch.nn.functional.logsigmoid(
        0.1 * ((policy_w - ref_w) - (policy_l - ref_l))
    ).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert policy_logits_w.grad is not None and torch.count_nonzero(policy_logits_w.grad)
    assert policy_logits_l.grad is not None and torch.count_nonzero(policy_logits_l.grad)
    assert ref_logits_w.grad is None and ref_logits_l.grad is None


def check_51_top_p_boundary():
    probs = np.array([0.60, 0.25, 0.10, 0.05])
    order = np.argsort(probs)[::-1]
    sorted_p = probs[order]
    keep = np.cumsum(sorted_p) - sorted_p < 0.8
    selected = sorted_p[keep] / sorted_p[keep].sum()
    assert np.array_equal(order[keep], [0, 1])
    assert np.allclose(selected, [0.60 / 0.85, 0.25 / 0.85])


def logsumexp(values):
    values = np.asarray(values, dtype=float)
    m = values.max()
    return m + np.log(np.exp(values - m).sum())


def check_52_rag_sequence_vs_token_marginalization():
    doc_prob = np.array([0.7, 0.3])
    token_prob = np.array([[0.2, 0.9], [0.8, 0.1]])
    sequence_prob = np.sum(doc_prob * np.prod(token_prob, axis=1))
    token_marginal_prob = np.prod(np.sum(doc_prob[:, None] * token_prob, axis=0))
    assert isclose(sequence_prob, 0.15, abs_tol=1e-12)
    assert isclose(token_marginal_prob, 0.2508, abs_tol=1e-12)
    assert isclose(np.exp(logsumexp(np.log(doc_prob) + np.log(token_prob).sum(1))), 0.15)


def check_53_gptq_constrained_compensation():
    H = np.array([[4.0, 2.0], [2.0, 2.0]])
    Hinv = np.linalg.inv(H)
    w = np.array([1.3, 0.7])
    q0 = 1.0
    error = w[0] - q0
    delta = -error * Hinv[:, 0] / Hinv[0, 0]
    compensated = w + delta
    loss = delta @ H @ delta
    assert np.allclose(Hinv, [[0.5, -0.5], [-0.5, 1.0]])
    assert np.allclose(compensated, [1.0, 1.0])
    assert isclose(loss, 0.18, abs_tol=1e-12)
    assert isclose(error**2 / Hinv[0, 0], loss, abs_tol=1e-12)


def symmetric_quantize(w, qmax=7):
    scale = np.max(np.abs(w)) / qmax
    return np.clip(np.round(w / scale), -qmax, qmax) * scale


def check_54_awq_equivalence_and_quantized_outputs():
    x = np.array([[0.2, 1.0], [0.4, -0.5]])
    W = np.array([[0.3, 0.7], [0.2, -0.4]])
    s = np.array([2.0, 1.0])
    xp, Wp = x / s, W * s[None, :]
    y = x @ W.T
    assert np.allclose(xp @ Wp.T, y)
    err_base = np.linalg.norm(x @ symmetric_quantize(W).T - y)
    err_scaled = np.linalg.norm(xp @ symmetric_quantize(Wp).T - y)
    assert np.isfinite(err_base) and np.isfinite(err_scaled)


def check_55_smoothquant_float_equivalence_and_int8_path():
    X = np.array([[1.0, 0.25], [-0.5, 2.0]])
    W = np.array([[0.5, -1.0], [2.0, 0.25]])
    smooth = np.array([2.0, 1.0])
    Xp, Wp = X / smooth, smooth[:, None] * W
    Y = X @ W
    assert np.allclose(Xp @ Wp, Y)
    sx, sw = np.max(np.abs(Xp)) / 127, np.max(np.abs(Wp)) / 127
    Xq = np.clip(np.round(Xp / sx), -127, 127).astype(np.int8).astype(np.int32)
    Wq = np.clip(np.round(Wp / sw), -127, 127).astype(np.int8).astype(np.int32)
    acc32 = Xq @ Wq
    Yhat = acc32.astype(np.float64) * sx * sw
    assert acc32.dtype == np.int32 and np.all(np.isfinite(Yhat))
    assert np.linalg.norm(Yhat - Y) < 0.05


def clip_loss(image, text, temperature):
    image = image / np.linalg.norm(image, axis=-1, keepdims=True)
    text = text / np.linalg.norm(text, axis=-1, keepdims=True)
    scores = image @ text.T / temperature

    def row_ce(matrix):
        return np.mean([logsumexp(row) - row[i] for i, row in enumerate(matrix)])

    return 0.5 * (row_ce(scores) + row_ce(scores.T))


def check_56_clip_symmetric_contrastive_loss():
    paired = np.eye(2)
    mismatched = paired[::-1]
    assert clip_loss(paired, paired, 0.1) < clip_loss(paired, mismatched, 0.1)


def check_57_diffusion_and_conditional_flow_step():
    x0, noise, alpha_bar = 2.0, -1.0, 0.64
    xt = np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * noise
    x0_hat = (xt - np.sqrt(1 - alpha_bar) * noise) / np.sqrt(alpha_bar)
    assert isclose(xt, 1.0) and isclose(x0_hat, x0)
    z, data, t = -1.0, 3.0, 0.25
    xt_flow = (1 - t) * z + t * data
    v_cond = data - z
    assert isclose(xt_flow, 0.0) and isclose(xt_flow + 0.25 * v_cond, 1.0)


def check_58_welford_and_hierarchical_topk():
    def merge(a, b):
        na, ma, m2a = a
        nb, mb, m2b = b
        n = na + nb
        d = mb - ma
        return n, ma + d * nb / n, m2a + m2b + d * d * na * nb / n

    assert merge((2, 2.0, 2.0), (1, 5.0, 0.0)) == (3, 3.0, 8.0)
    shards, k = [[9, 1, 8, 2], [7, 6, 3, 5]], 2
    local = [sorted(s, reverse=True)[:k] for s in shards]
    candidates = [v for part in local for v in part]
    full = [v for part in shards for v in part]
    assert sorted(candidates, reverse=True)[:k] == sorted(full, reverse=True)[:k]


def check_59_distillation_kl_and_low_rank():
    def softmax(x, temperature):
        z = x / temperature
        e = np.exp(z - z.max())
        return e / e.sum()

    teacher = softmax(np.array([2.0, 0.0]), 2.0)
    student = softmax(np.array([1.0, 0.0]), 2.0)
    kl = np.sum(teacher * (np.log(teacher) - np.log(student)))
    assert kl > 0.0
    W = np.array([[3.0, 0.0], [0.0, 1.0]])
    U, singular, Vt = np.linalg.svd(W, full_matrices=False)
    W_rank1 = U[:, :1] @ np.diag(singular[:1]) @ Vt[:1]
    assert isclose(np.linalg.norm(W - W_rank1), 1.0, abs_tol=1e-12)


def check_60_rope_relative_position_identity():
    def rotate(v, position, omega):
        angle = position * omega
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        return R @ v

    q, k = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    p, s, omega = 2, 5, np.pi / 4
    lhs = rotate(q, p, omega) @ rotate(k, s, omega)
    rhs = q @ rotate(k, s - p, omega)
    assert np.isclose(lhs, rhs)


CHECKS = [
    check_47_mamba_zoh_step_and_scan,
    check_48_masked_ce_gradient_and_accumulation,
    check_49_lora_merge_and_initial_gradients,
    check_50_dpo_masks_and_group_advantage,
    check_51_top_p_boundary,
    check_52_rag_sequence_vs_token_marginalization,
    check_53_gptq_constrained_compensation,
    check_54_awq_equivalence_and_quantized_outputs,
    check_55_smoothquant_float_equivalence_and_int8_path,
    check_56_clip_symmetric_contrastive_loss,
    check_57_diffusion_and_conditional_flow_step,
    check_58_welford_and_hierarchical_topk,
    check_59_distillation_kl_and_low_rank,
    check_60_rope_relative_position_identity,
]


if __name__ == "__main__":
    for check in CHECKS:
        check()
        print(f"PASS: {check.__name__}")
    if "--torch" in sys.argv:
        check_50_dpo_torch_core()
        print("PASS: check_50_dpo_torch_core")
