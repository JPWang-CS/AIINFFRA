"""Request timestamp and Amdahl examples; no GPU measurement."""
import math


def request_metrics(arrival_ms, token_times_ms):
    times = list(token_times_ms)
    if not times or not all(math.isfinite(x) for x in [arrival_ms, *times]):
        raise ValueError("at least one token and finite timestamps required")
    if times[0] < arrival_ms or any(a > b for a, b in zip(times, times[1:])):
        raise ValueError("timestamps must be chronological")
    ttft = times[0] - arrival_ms
    e2e = times[-1] - arrival_ms
    tpot = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else None
    return ttft, tpot, e2e


def amdahl(fraction, speedup):
    if not 0 <= fraction <= 1 or not math.isfinite(speedup) or speedup <= 0:
        raise ValueError("invalid serial fraction or local speedup")
    return 1.0 / (1.0 - fraction + fraction / speedup)


if __name__ == "__main__":
    assert request_metrics(0, [120, 160, 220]) == (120, 50, 220)
    assert request_metrics(10, [25]) == (15, None, 15)
    assert math.isclose(amdahl(0.2, 2), 1 / 0.9)
    assert math.isclose(amdahl(0.1, 2), 1 / 0.95)
    for arrival, times in [(0, []), (5, [4]), (0, [3, 2]), (0, [float("nan")])]:
        try:
            request_metrics(arrival, times)
        except ValueError:
            continue
        raise AssertionError("invalid timestamps accepted")
    # Component P95s cannot be added to reconstruct request P95.
    ttfts = [100] * 50 + [1] * 50
    decode_tails = [1] * 50 + [100] * 50
    p95 = lambda xs: sorted(xs)[math.ceil(0.95 * len(xs)) - 1]
    assert p95(ttfts) + p95(decode_tails) == 200
    assert p95([a + b for a, b in zip(ttfts, decode_tails)]) == 101
    print("PASS: timestamp identity, single-token boundary, P95 counterexample, Amdahl")
