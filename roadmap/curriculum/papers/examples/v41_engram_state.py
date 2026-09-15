"""CPU teaching model of Engram address state, lookup, gating, and prefetch."""
import numpy as np


DEAD = -1


def make_compressed_map(token_spellings):
    """Demonstration only; the real mapping is built from the model tokenizer."""
    canonical = {}
    token_map = []
    for spelling in token_spellings:
        key = spelling.strip().casefold()
        if key not in canonical:
            canonical[key] = len(canonical)
        token_map.append(canonical[key])
    return token_map


class NgramAddressState:
    def __init__(self, token_map, pad_id, multipliers, primes):
        self.token_map = token_map
        self.pad_id = int(pad_id)
        self.multipliers = np.asarray(multipliers, dtype=np.int64)
        self.primes = np.asarray(primes, dtype=np.int64)
        if self.multipliers.ndim != 1 or self.primes.ndim != 2:
            raise ValueError("one multiplier per lookback and one prime per order/head required")
        if self.multipliers.size != self.primes.shape[0] + 1 or np.any(self.primes <= 1):
            raise ValueError("orders 2..N require one extra lookback and positive bucket moduli")
        offsets = np.zeros_like(self.primes)
        cursor = 0
        for order in range(self.primes.shape[0]):
            for head in range(self.primes.shape[1]):
                offsets[order, head] = cursor
                cursor += self.primes[order, head]
        self.offsets = offsets
        self.table_rows = cursor
        self.cache = []

    def append_chunk(self, token_ids, token_mask=None):
        if token_mask is None:
            token_mask = [True] * len(token_ids)
        if len(token_ids) != len(token_mask):
            raise ValueError("token IDs and token mask must align")
        emitted = []
        for token_id, valid_text in zip(token_ids, token_mask):
            self.cache.append(self.token_map[int(token_id)] if valid_text else DEAD)
            emitted.append(self.hash_at(len(self.cache) - 1))
        return emitted

    def hash_at(self, position):
        orders, heads = self.primes.shape
        lookback = []
        blocked = False
        for shift in range(orders + 1):
            source_pos = position - shift
            source = DEAD if source_pos < 0 else self.cache[source_pos]
            blocked = blocked or source_pos < 0 or source == DEAD
            lookback.append(self.pad_id if blocked else source)
        ids = np.empty((orders, heads), dtype=np.int64)
        for order in range(orders):
            rolling = lookback[0] * self.multipliers[0]
            for shift in range(1, order + 2):
                rolling ^= lookback[shift] * self.multipliers[shift]
            for head in range(heads):
                ids[order, head] = self.offsets[order, head] + rolling % self.primes[order, head]
        return ids


def lookup_rows(ids, table):
    return table[np.asarray(ids, dtype=np.int64)]


def normalized_gate(residual, key, gamma_q, gamma_k, eps=1e-6, clamp=1e-6, token_mask=None):
    """HF Minimal gate arithmetic for one stream, evaluated with synthetic values."""
    residual = np.asarray(residual, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    if residual.shape != key.shape or residual.ndim < 1 or residual.shape[-1] == 0:
        raise ValueError("residual and key must have the same nonempty feature axis")
    gamma_q = np.asarray(gamma_q, dtype=np.float64)
    gamma_k = np.asarray(gamma_k, dtype=np.float64)
    if gamma_q.shape != (residual.shape[-1],) or gamma_k.shape != gamma_q.shape:
        raise ValueError("per-channel query/key weights must match hidden width")
    if eps <= 0 or clamp <= 0:
        raise ValueError("eps and signed-root clamp must be positive")
    rstd = 1.0 / np.sqrt(np.mean(residual**2, axis=-1) + eps)
    rstd *= 1.0 / np.sqrt(np.mean(key**2, axis=-1) + eps)
    dot = np.sum(residual * gamma_q * gamma_k * key, axis=-1) * rstd / np.sqrt(residual.shape[-1])
    signed_root = np.copysign(np.sqrt(np.maximum(np.abs(dot), clamp)), dot)
    gate = 1.0 / (1.0 + np.exp(-signed_root))
    if token_mask is not None:
        gate = np.where(token_mask, gate, 0.0)
    return gate


def consume_memory(residual, key, value, gamma_q, gamma_k, token_mask=None):
    gate = normalized_gate(residual, key, gamma_q, gamma_k, token_mask=token_mask)
    return residual + gate[..., None] * value


def prefetch_wait_ms(request_ready_ms, consumer_ready_ms):
    return max(0.0, request_ready_ms - consumer_ready_ms)


def main():
    spellings = [" The", "the", "THE", "river", "image"]
    token_map = make_compressed_map(spellings)
    assert token_map[0] == token_map[1] == token_map[2]

    multipliers = np.array([3, 7, 11, 15], dtype=np.int64)
    primes = np.array([[17, 19], [23, 29], [31, 37]], dtype=np.int64)
    state = NgramAddressState(token_map, pad_id=1, multipliers=multipliers, primes=primes)
    prefill = state.append_chunk([0, 3])
    decode = state.append_chunk([1, 4], token_mask=[True, False])
    assert prefill[-1].shape == decode[0].shape == decode[1].shape == (3, 2)
    assert state.cache[-1] == DEAD
    # An image/dead boundary prevents the following text hash from crossing it.
    after_image = state.append_chunk([3])[0]
    assert after_image.shape == (3, 2)
    assert after_image.min() >= 0 and after_image.max() < state.table_rows

    table = np.arange(state.table_rows * 4, dtype=np.float32).reshape(state.table_rows, 4)
    rows = lookup_rows(decode[0], table)
    assert rows.shape == (3, 2, 4)
    value = rows.mean(axis=(0, 1))
    residual_a = np.array([1.0, 0.0, 0.0, 0.0])
    residual_b = np.array([0.0, 1.0, 0.0, 0.0])
    keys = np.stack([rows[0, 0], rows[1, 1]])
    gamma_q = np.array([1.0, 0.8, 1.1, 0.7])
    gamma_k = np.array([0.9, 1.2, 0.6, 1.0])
    gates = np.array([
        normalized_gate(residual_a, keys[0], gamma_q, gamma_k),
        normalized_gate(residual_b, keys[1], gamma_q, gamma_k),
    ])
    assert gates[0] != gates[1]
    assert normalized_gate(-residual_a, keys[0], gamma_q, gamma_k) < 0.5
    assert normalized_gate(np.zeros(4), keys[0], gamma_q, gamma_k) > 0.5
    assert normalized_gate(residual_a, keys[0], gamma_q, gamma_k, token_mask=False) == 0.0
    updated_a = residual_a + gates[0] * value
    updated_b = residual_b + gates[1] * value
    assert not np.array_equal(updated_a, updated_b)
    assert prefetch_wait_ms(3.0, 5.0) == 0.0
    assert prefetch_wait_ms(7.0, 5.0) == 2.0
    print("PASS: tokenizer-map example, chunked hash state, image boundary, lookup, HF Minimal gate arithmetic, prefetch wait")
    print("Synthetic tokenizer map and weights; no checkpoint, RDMA, or accelerator used")


if __name__ == "__main__":
    main()
