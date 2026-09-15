"""Deterministic sampling primitives and cache-state models, without GPU dependencies."""
import math


def nucleus(logits, threshold, temperature=1.0):
    if not logits or not 0 < threshold <= 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("nonempty logits, 0 < p <= 1, positive temperature required")
    if not all(math.isfinite(x) for x in logits):
        raise ValueError("finite logits required")
    ids = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    maximum = logits[ids[0]]
    mass = [math.exp((logits[i]-maximum)/temperature) for i in ids]
    total = sum(mass)
    selected, weights, cumulative = [], [], 0.0
    for token, value in zip(ids, mass):
        selected.append(token)
        weights.append(value)
        cumulative += value / total
        if threshold < 1 and cumulative >= threshold:
            break
    normalizer = sum(weights)
    return selected, [x/normalizer for x in weights]


def inverse_cdf(ids, weights, uniform):
    if not ids or len(ids) != len(weights) or not 0 <= uniform < 1:
        raise ValueError("valid support and uniform in [0,1) required")
    if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights),1.0):
        raise ValueError("probabilities must be normalized")
    cumulative = 0.0
    for token, weight in zip(ids, weights):
        cumulative += weight
        if uniform < cumulative:
            return token
    return next(token for token, weight in reversed(list(zip(ids, weights))) if weight > 0)


def acceptance_and_residual(target, draft):
    if len(target) != len(draft) or not target or any(x<0 or not math.isfinite(x) for x in target+draft):
        raise ValueError("matching finite nonnegative distributions required")
    if not math.isclose(sum(target),1.0) or not math.isclose(sum(draft),1.0):
        raise ValueError("normalized distributions required")
    overlap = sum(min(p,q) for p,q in zip(target,draft))
    residual = [max(p-q,0) for p,q in zip(target,draft)]
    norm = sum(residual)
    return overlap, ([x/norm for x in residual] if norm else None)


def commit_speculation(cache, prefix_length, draft_count, accepted):
    if not 0 <= accepted <= draft_count or not 0 <= prefix_length <= len(cache)-draft_count:
        raise ValueError("invalid speculative cache span")
    # Logical truncation: rejected values may remain physically allocated.
    return cache[:prefix_length+accepted]


if __name__ == "__main__":
    ids, weights = nucleus([math.log(.5),math.log(.3),math.log(.2)], .7)
    assert ids == [0,1] and abs(weights[0]-.625)<1e-12
    assert nucleus([2,1,0], 1)[0] == [0,1,2]
    assert nucleus([0,0,0], .1)[0] == [0]
    assert inverse_cdf([0,1],[.625,.375], .624) == 0
    assert inverse_cdf([0,1],[.625,.375], .625) == 1
    # Stratified uniform inputs verify the CDF partition without stochastic flakiness.
    counts = [0,0]
    for j in range(800):
        counts[inverse_cdf([0,1],[.625,.375], (j+.5)/800)] += 1
    assert counts == [500,300]
    p,q = [.6,.3,.1],[.2,.5,.3]
    accept, residual = acceptance_and_residual(p,q)
    assert abs(accept-.6)<1e-12
    assert all(abs(min(a,b)+(1-accept)*r-a)<1e-12 for a,b,r in zip(p,q,residual))
    same_accept, same_residual = acceptance_and_residual(p,p)
    assert math.isclose(same_accept,1.0) and same_residual is None
    assert commit_speculation(list(range(10)), 5, 3, 1) == list(range(6))
    print("PASS: nucleus crossing token, p=1/ties, inverse CDF boundary, rejection correction, KV rollback")
