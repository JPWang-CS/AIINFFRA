"""CPU-only shape, BF16 weight, and KV-cache ledger for Embedded-7B.

The constants below come from the checked-in openPangu-Embedded-7B-V1.1
config.json. This is arithmetic verification, not a model load or NPU/GPU run.
"""

HIDDEN = 4096
INTERMEDIATE = 12800
LAYERS = 34
QUERY_HEADS = 32
KV_HEADS = 8
VOCAB = 153376
BYTES_PER_BF16 = 2
BLOCK_SIZE = 128


def require_integer(name, value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def calculate():
    assert HIDDEN % QUERY_HEADS == 0
    assert QUERY_HEADS % KV_HEADS == 0
    head_dim = HIDDEN // QUERY_HEADS
    kv_width = KV_HEADS * head_dim

    # Q, K, V, O projection weights and their configured biases.
    attention_weights = 2 * HIDDEN * HIDDEN + 2 * HIDDEN * kv_width
    attention_bias = 2 * HIDDEN + 2 * kv_width
    mlp_weights = 3 * HIDDEN * INTERMEDIATE  # gate + up + down, no bias
    norm_weights = 2 * HIDDEN
    layer_params = attention_weights + attention_bias + mlp_weights + norm_weights

    embedding_params = VOCAB * HIDDEN
    lm_head_params = VOCAB * HIDDEN  # tie_word_embeddings=false
    final_norm_params = HIDDEN
    total_params = LAYERS * layer_params + embedding_params + lm_head_params + final_norm_params

    # One K and one V vector per KV head, per layer, per token.
    kv_bytes_per_token = LAYERS * 2 * KV_HEADS * head_dim * BYTES_PER_BF16

    def kv_bytes(tokens, batch=1):
        require_integer("tokens", tokens, 0)
        require_integer("batch", batch, 1)
        return tokens * batch * kv_bytes_per_token

    def paged_kv_bytes(tokens, batch=1, block_size=BLOCK_SIZE):
        require_integer("tokens", tokens, 0)
        require_integer("batch", batch, 1)
        require_integer("block_size", block_size, 1)
        blocks = (tokens + block_size - 1) // block_size
        return blocks * block_size * batch * kv_bytes_per_token

    return {
        "head_dim": head_dim,
        "kv_width": kv_width,
        "attention_weights": attention_weights,
        "attention_bias": attention_bias,
        "mlp_weights": mlp_weights,
        "layer_params": layer_params,
        "embedding_params": embedding_params,
        "lm_head_params": lm_head_params,
        "final_norm_params": final_norm_params,
        "total_params": total_params,
        "weight_bytes": total_params * BYTES_PER_BF16,
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_bytes": kv_bytes,
        "paged_kv_bytes": paged_kv_bytes,
    }


def gibibytes(byte_count):
    return byte_count / (1024**3)


def main():
    ledger = calculate()

    def assert_rejected(call):
        try:
            call()
        except ValueError:
            return
        raise AssertionError("invalid ledger input was accepted")

    assert ledger["head_dim"] == 128
    assert ledger["kv_width"] == 1024
    assert ledger["layer_params"] == 199_247_872
    assert ledger["total_params"] == 8_030_887_936
    assert ledger["kv_bytes_per_token"] == 139_264
    assert ledger["kv_bytes"](32768) == 4.25 * (1024**3)
    assert ledger["kv_bytes"](0) == 0
    assert ledger["paged_kv_bytes"](0) == 0
    assert ledger["paged_kv_bytes"](128) == 128 * ledger["kv_bytes_per_token"]
    assert ledger["paged_kv_bytes"](129) == 256 * ledger["kv_bytes_per_token"]
    assert ledger["paged_kv_bytes"](256) == 256 * ledger["kv_bytes_per_token"]
    huge_tokens = 10**100
    huge_slots = ((huge_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    assert ledger["paged_kv_bytes"](huge_tokens) == huge_slots * ledger["kv_bytes_per_token"]

    for invalid_call in (
        lambda: ledger["kv_bytes"](-1),
        lambda: ledger["kv_bytes"](1.5),
        lambda: ledger["kv_bytes"](True),
        lambda: ledger["kv_bytes"](1, batch=0),
        lambda: ledger["kv_bytes"](1, batch=1.0),
        lambda: ledger["kv_bytes"](1, batch=True),
        lambda: ledger["paged_kv_bytes"](-1),
        lambda: ledger["paged_kv_bytes"](True),
        lambda: ledger["paged_kv_bytes"](1.5),
        lambda: ledger["paged_kv_bytes"](1, batch=0),
        lambda: ledger["paged_kv_bytes"](1, batch=-1),
        lambda: ledger["paged_kv_bytes"](1, batch=1.5),
        lambda: ledger["paged_kv_bytes"](1, batch=False),
        lambda: ledger["paged_kv_bytes"](1, block_size=0),
        lambda: ledger["paged_kv_bytes"](1, block_size=-1),
        lambda: ledger["paged_kv_bytes"](1, block_size=1.5),
        lambda: ledger["paged_kv_bytes"](1, block_size=True),
    ):
        assert_rejected(invalid_call)

    print("openPangu-Embedded-7B-V1.1 | config-derived CPU ledger")
    print(f"head_dim={ledger['head_dim']}; K/V width per projection={ledger['kv_width']}")
    print(f"attention projection weights/layer={ledger['attention_weights']:,} params")
    print(f"attention projection biases/layer={ledger['attention_bias']:,} params")
    print(f"MLP weights/layer={ledger['mlp_weights']:,} params")
    print(f"one decoder layer={ledger['layer_params']:,} params")
    print(f"all model weights={ledger['total_params']:,} params; BF16 payload={gibibytes(ledger['weight_bytes']):.6f} GiB")
    print(f"KV/token across all layers={ledger['kv_bytes_per_token']:,} bytes")
    print(f"KV at 32,768 tokens, batch 1={gibibytes(ledger['kv_bytes'](32768)):.2f} GiB")
    print(f"paged KV at 129 tokens, block {BLOCK_SIZE}={ledger['paged_kv_bytes'](129):,} bytes")
    print("PASS: arithmetic, page-boundary, large-integer, and invalid-input assertions")
    print("No model weights, framework, accelerator, or performance claim used")


if __name__ == "__main__":
    main()
