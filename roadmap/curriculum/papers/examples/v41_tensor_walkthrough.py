"""CPU-only tensor-order and shape checks; no model weights or accelerator."""
import numpy as np


def patch_embed(patches, weight, bias):
    return patches.reshape(len(patches), -1) @ weight.T + bias


def split_vision_qkv(projected, weight, bias, heads):
    n, dv = projected.shape
    if dv % heads or weight.shape != (3 * dv, dv) or bias.shape != (3 * dv,):
        raise ValueError("incompatible vision projection shapes")
    qkv = projected @ weight.T + bias
    return tuple(x.reshape(n, heads, dv // heads)
                 for x in np.split(qkv, 3, axis=-1))


def aligner_unshuffle(features, n_h, n_w, ratio):
    if (features.ndim != 2 or features.shape[0] != n_h * n_w
            or ratio <= 0 or n_h % ratio or n_w % ratio):
        raise ValueError("flat feature grid must be divisible by ratio")
    dv = features.shape[-1]
    grid = features.reshape(n_h, n_w, dv)
    return (grid.reshape(n_h // ratio, ratio, n_w // ratio, ratio, dv)
                .transpose(0, 2, 4, 1, 3)
                .reshape((n_h // ratio) * (n_w // ratio), dv * ratio * ratio))


def project_to_language(unshuffled, w1, b1, w2, b2):
    hidden = unshuffled @ w1.T + b1
    # Tanh GELU approximation, used here only for a small CPU shape example.
    gelu = 0.5 * hidden * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden**3)))
    return gelu @ w2.T + b2


def hc_pre(streams, pre_mix):
    """[source, feature] weighted by [source] -> one feature vector."""
    if streams.ndim != 2 or pre_mix.shape != (streams.shape[0],):
        raise ValueError("pre mix must align with the source-stream axis")
    return np.sum(pre_mix[:, None] * streams, axis=0)


def hc_post(sublayer_output, residual, post_mix, comb_source_destination):
    """HF hc_post axis order: comb[source, destination], sum over source."""
    n, d = residual.shape
    if (sublayer_output.shape != (d,) or post_mix.shape != (n,)
            or comb_source_destination.shape != (n, n)):
        raise ValueError("incompatible mHC stream, post, or comb shapes")
    residual_mix = np.sum(
        comb_source_destination[:, :, None] * residual[:, None, :], axis=0
    )
    return post_mix[:, None] * sublayer_output[None, :] + residual_mix


def main():
    rng = np.random.default_rng(17)
    n_h, n_w, patch, channels, dv, heads, ratio, language_dim = 6, 9, 2, 3, 8, 2, 3, 5
    patches = rng.normal(size=(n_h * n_w, channels, patch, patch))
    w_patch = rng.normal(size=(dv, channels * patch * patch))
    b_patch = rng.normal(size=(dv,))
    vision = patch_embed(patches, w_patch, b_patch)
    wqkv = rng.normal(size=(3 * dv, dv))
    bqkv = rng.normal(size=(3 * dv,))
    q, k, v = split_vision_qkv(vision, wqkv, bqkv, heads)
    assert q.shape == k.shape == v.shape == (n_h * n_w, heads, dv // heads)

    # Check the exact channel-major order produced by [C,H,W] + unfold.
    order = np.arange(2 * 2 * 2).reshape(4, 2)
    packed = aligner_unshuffle(order, 2, 2, 2)
    assert np.array_equal(packed, [[0, 2, 4, 6, 1, 3, 5, 7]])

    unshuffled = aligner_unshuffle(vision, n_h, n_w, ratio)
    assert unshuffled.shape == ((n_h // ratio) * (n_w // ratio), dv * ratio**2)
    w1 = rng.normal(size=(11, unshuffled.shape[1]))
    b1 = rng.normal(size=(11,))
    w2 = rng.normal(size=(language_dim, 11))
    b2 = rng.normal(size=(language_dim,))
    language = project_to_language(unshuffled, w1, b1, w2, b2)
    assert language.shape == ((n_h // ratio) * (n_w // ratio), language_dim)
    assert n_h * n_w * dv == language.shape[0] * dv * ratio**2

    # A cyclic permutation is non-symmetric: rows are sources, columns destinations.
    streams = np.array([[1., 10.], [2., 20.], [3., 30.]])
    pre = np.array([0.1, 0.2, 0.7])
    pre_output = hc_pre(streams, pre)
    assert np.allclose(pre_output, [2.6, 26.0])
    comb_source_destination = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    post = np.array([0.25, 0.5, 0.75])
    sublayer_output = np.array([2., -1.])
    post_output = hc_post(sublayer_output, streams, post, comb_source_destination)
    expected_residual_mix = streams[[2, 0, 1]]
    assert np.array_equal(comb_source_destination.T @ streams, expected_residual_mix)
    assert np.array_equal(post_output, post[:, None] * sublayer_output + expected_residual_mix)
    assert post_output.shape == (3, 2)
    assert not np.array_equal(comb_source_destination @ streams, expected_residual_mix)
    try:
        aligner_unshuffle(vision[:7 * 8], 7, 8, ratio)
    except ValueError:
        pass
    else:
        raise AssertionError("non-divisible grid must not be silently called lossless")
    print("PASS: patch/QKV shapes, unfold order, unshuffle count, projector width, mHC pre/post shapes and comb transpose")


if __name__ == "__main__":
    main()
