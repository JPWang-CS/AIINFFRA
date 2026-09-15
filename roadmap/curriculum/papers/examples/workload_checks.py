"""CPU thought experiments for training and inference workloads."""
import heapq
import numpy as np


def compute_optimum(compute, kappa, a, b, alpha, beta):
    if min(compute, kappa, a, b, alpha, beta) <= 0:
        raise ValueError("positive model constants required")
    product = compute / kappa
    n = ((alpha * a / (beta * b)) * product ** beta) ** (1 / (alpha + beta))
    return n, product / n


def worker_makespan(durations, workers, synchronous=False):
    durations = [float(t) for t in durations]
    if workers < 1 or any(not np.isfinite(t) or t < 0 for t in durations):
        raise ValueError("positive workers and finite nonnegative durations required")
    if synchronous:
        return sum(max(durations[i:i + workers]) for i in range(0, len(durations), workers))
    available = [0.] * workers
    heapq.heapify(available)
    for duration in durations:
        start = heapq.heappop(available)
        heapq.heappush(available, start + duration)
    return max(available)


def clipped_objective(new_logp, old_logp, advantage, valid, clip=0.2):
    new, old, adv, mask = [np.asarray(x) for x in (new_logp, old_logp, advantage, valid)]
    if not (new.shape == old.shape == adv.shape == mask.shape) or clip < 0:
        raise ValueError("aligned token arrays and nonnegative clip required")
    mask = mask.astype(bool)
    if not mask.any():
        raise ValueError("no valid action tokens")
    ratio = np.exp(new[mask] - old[mask])
    unclipped = ratio * adv[mask]
    clipped = np.clip(ratio, 1 - clip, 1 + clip) * adv[mask]
    return float(np.minimum(unclipped, clipped).mean())


def prefix_storage(prefix_tokens, suffix_lengths, bytes_per_token):
    lengths = [int(x) for x in suffix_lengths]
    if prefix_tokens < 0 or bytes_per_token < 0 or any(x < 0 for x in lengths):
        raise ValueError("nonnegative sizes required")
    duplicated = (len(lengths) * prefix_tokens + sum(lengths)) * bytes_per_token
    shared = (prefix_tokens + sum(lengths)) * bytes_per_token if lengths else 0
    return duplicated, shared


def check():
    n, d = compute_optimum(240000., 6., 4., 1., 1., 1.)
    np.testing.assert_allclose([n, d], [400., 100.])
    candidates = np.geomspace(50., 2000., 20000)
    losses = 4 / candidates + 1 / (40000 / candidates)
    assert abs(candidates[np.argmin(losses)] / n - 1) < .001
    t = [1, 10, 1, 1, 10, 1]
    sync = worker_makespan(t, 2, True)
    async_time = worker_makespan(t, 2, False)
    assert sync == 21 and async_time == 13
    assert async_time >= max(sum(t) / 2, max(t))
    value = clipped_objective(np.log([.8, .1, .9]), np.log([.4, .2, .9]),
                              [1., -1., 100.], [True, True, False])
    np.testing.assert_allclose(value, (1.2 - .8) / 2)
    duplicate, shared = prefix_storage(1000, [10, 20, 30, 40], 16)
    assert duplicate == 65600 and shared == 17600
    # Even an early linear layer's cache changes when its weights change.
    hidden = np.array([[1., 2.]])
    old_w = np.eye(2); new_w = old_w.copy(); new_w[0, 0] += .1
    assert not np.array_equal(hidden @ old_w, hidden @ new_w)
    print("PASS: compute allocation; scheduling long tail; clipped token objective; "
          "prefix storage; weight-version cache counterexample")


if __name__ == "__main__":
    check()
