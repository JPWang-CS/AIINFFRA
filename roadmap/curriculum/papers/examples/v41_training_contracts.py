"""CPU arithmetic for the reported schedule and packed-sample attention mask."""
import math

BATCH_TOKENS = 100_600_000
WARMUP_TOKENS = 2_000 * BATCH_TOKENS
PEAK_LR = 2.6e-4
FLOOR_LR = 2.6e-5


def learning_rate(tokens_seen):
    if type(tokens_seen) is not int or not 0 <= tokens_seen <= 45_000_000_000_000:
        raise ValueError("tokens_seen must be an integer within the reported 0..45T run")
    if tokens_seen < WARMUP_TOKENS:
        return PEAK_LR * tokens_seen / WARMUP_TOKENS
    if tokens_seen <= 28_000_000_000_000:
        return PEAK_LR
    if tokens_seen < 40_000_000_000_000:
        phase = (tokens_seen - 28_000_000_000_000) / 12_000_000_000_000
        return FLOOR_LR + 0.5 * (PEAK_LR - FLOOR_LR) * (1 + math.cos(math.pi * phase))
    return FLOOR_LR


def packed_causal_visible(sample_ids, query, key):
    if not (type(query) is int and type(key) is int
            and 0 <= query < len(sample_ids) and 0 <= key < len(sample_ids)):
        raise ValueError("query and key must index the packed sequence")
    return key <= query and sample_ids[key] == sample_ids[query]


def main():
    assert WARMUP_TOKENS == 201_200_000_000
    assert learning_rate(0) == 0
    assert learning_rate(WARMUP_TOKENS) == PEAK_LR
    assert learning_rate(28_000_000_000_000) == PEAK_LR
    assert math.isclose(learning_rate(34_000_000_000_000), 1.43e-4, abs_tol=1e-16)
    assert learning_rate(40_000_000_000_000) == FLOOR_LR
    assert learning_rate(45_000_000_000_000) == FLOOR_LR
    for bad in (-1, 45_000_000_000_001, 1.5, True):
        try:
            learning_rate(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid schedule input accepted")
    samples = [0, 0, 1, 1, 1]
    visible = [[k for k in range(5) if packed_causal_visible(samples, q, k)]
               for q in range(5)]
    assert visible == [[0], [0, 1], [2], [2, 3], [2, 3, 4]]
    assert 0 <= 2 and not packed_causal_visible(samples, 2, 0)
    print("PASS: warmup/cosine/floor boundaries, midpoint LR, packed-sample isolation")
    print("Continuous-token arithmetic proxy; no training run or quality measurement")


if __name__ == "__main__":
    main()
