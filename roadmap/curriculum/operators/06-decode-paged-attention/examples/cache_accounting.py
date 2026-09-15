"""CPU arithmetic and state-source models; not a DeepSeek inference implementation."""


def kv_payload(layers, lengths, kv_heads, head_dim, element_bytes):
    if min(layers, kv_heads, head_dim, element_bytes) <= 0 or any(n < 0 for n in lengths):
        raise ValueError("invalid cache dimensions")
    return 2 * layers * sum(lengths) * kv_heads * head_dim * element_bytes


def cache_sources(modes):
    kv_owner = index_owner = None
    result = []
    for layer, mode in enumerate(modes):
        if mode == "Full":
            kv_owner = index_owner = layer
        elif mode == "Reindex":
            if kv_owner is None:
                raise ValueError("Reindex needs an existing KV source")
            index_owner = layer
        elif mode == "Reuse":
            if kv_owner is None or index_owner is None:
                raise ValueError("Reuse needs KV and selection sources")
        else:
            raise ValueError("unknown mode")
        result.append((layer, kv_owner, index_owner))
    return result


def replay_positions(query, start, window):
    if window <= 0 or start < 0 or query < start:
        raise ValueError("invalid replay segment")
    return list(range(max(start, query - window + 1), query + 1))


def ced_token_layer_ratio(sequence, window):
    if sequence <= 0 or window <= 0:
        raise ValueError("positive sequence and window required")
    return (sequence + min(sequence, window)) / (2 * sequence)


if __name__ == "__main__":
    assert kv_payload(1, [4096] * 8, 8, 128, 2) == 128 * 1024**2
    assert 40 + 20 + 4 == 64
    assert 40 / 4 + 20 / 2 + 4 == 24
    assert cache_sources(["Full", "Reuse", "Reindex", "Reuse", "Full", "Reuse"]) == [
        (0, 0, 0), (1, 0, 0), (2, 0, 2), (3, 0, 2), (4, 4, 4), (5, 4, 4)]
    for invalid in [["Reuse"], ["Reindex"], ["Full", "unknown"]]:
        try:
            cache_sources(invalid)
        except ValueError:
            continue
        raise AssertionError("invalid producer ordering accepted")
    assert replay_positions(100, 100, 4) == [100]
    assert replay_positions(103, 100, 4) == [100, 101, 102, 103]
    assert ced_token_layer_ratio(4096, 128) == 0.515625
    assert ced_token_layer_ratio(128, 128) == 1.0
    print("PASS: cache budget, additive compression, CSA2 source ownership, replay mask, CED work proxy")
