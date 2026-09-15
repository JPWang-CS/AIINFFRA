"""Routing, grouping and combine reference; not a production MoE implementation."""
import math


def route_topk(logits, k):
    if not logits or not 1 <= k <= len(logits) or not all(math.isfinite(x) for x in logits):
        raise ValueError("finite logits and 1 <= k <= E required")
    ids = sorted(range(len(logits)), key=lambda e: (-logits[e], e))[:k]
    maximum = max(logits[e] for e in ids)
    masses = [math.exp(logits[e] - maximum) for e in ids]
    total = sum(masses)
    return ids, [x / total for x in masses]


def group_routes(routes, experts):
    if experts <= 0:
        raise ValueError("positive expert count required")
    buckets = [[] for _ in range(experts)]
    for token, (ids, weights) in enumerate(routes):
        if len(ids) != len(weights) or len(ids) != len(set(ids)):
            raise ValueError("route ids and weights must match and be unique per token")
        for slot, (expert, weight) in enumerate(zip(ids, weights)):
            if not 0 <= expert < experts or not math.isfinite(weight) or weight < 0:
                raise ValueError("invalid expert or weight")
            buckets[expert].append((token, slot, weight))
    offsets = [0]
    for bucket in buckets:
        offsets.append(offsets[-1] + len(bucket))
    return offsets, [item for bucket in buckets for item in bucket]


def grouped_execute(inputs, routes, expert_functions):
    if len(inputs) != len(routes) or (inputs and any(len(row) != len(inputs[0]) for row in inputs)):
        raise ValueError("one route per token and rectangular inputs required")
    offsets, order = group_routes(routes, len(expert_functions))
    output = [[0.0] * len(row) for row in inputs]
    for expert, fn in enumerate(expert_functions):
        for token, slot, weight in order[offsets[expert]:offsets[expert+1]]:
            value = fn(inputs[token])
            if len(value) != len(output[token]):
                raise ValueError("expert output dimension mismatch")
            for d, v in enumerate(value):
                output[token][d] += weight * v
    return output


if __name__ == "__main__":
    ids, weights = route_topk([1,2,3,4], 2)
    assert ids == [3,2] and abs(sum(weights)-1) < 1e-12
    assert route_topk([7,7,7], 2)[0] == [0,1]  # explicit teaching tie rule
    assert route_topk([3,-9], 1)[1] == [1.0]
    routes = [route_topk(row, 2) for row in [[1,2,3,0],[8,1,3,0],[0,6,2,1]]]
    inputs = [[1,2],[3,4],[-1,5]]
    experts = [lambda x,e=e: [(e+1)*v + e for v in x] for e in range(4)]
    offsets, order = group_routes(routes, 4)
    assert offsets[-1] == 6 and offsets[-1] == offsets[-2]  # expert 3 is empty
    assert sorted((t,j) for t,j,_ in order) == [(t,j) for t in range(3) for j in range(2)]
    expected = []
    for x, (ids, ws) in zip(inputs, routes):
        expected.append([sum(w*experts[e](x)[d] for e,w in zip(ids,ws)) for d in range(2)])
    actual = grouped_execute(inputs, routes, experts)
    assert all(abs(x-y)<1e-12 for a,b in zip(actual,expected) for x,y in zip(a,b))
    assert 0.25*(2*3+4) != 2*(0.25*3)+4  # route weight cannot cross a bias
    assert group_routes([], 3) == ([0,0,0,0], [])
    assert group_routes([([2,0],[.7,.3]),([2,0],[.4,.6]),([1,2],[.5,.5])],3)[0] == [0,2,3,6]
    for invalid in [([0,0],[0.5,0.5]), ([4],[1]), ([0],[])]:
        try:
            group_routes([invalid], 4)
        except ValueError:
            continue
        raise AssertionError("invalid route accepted")
    print("PASS: top-k normalization/ties, permutation bijection, empty experts, grouped vs tokenwise combine")
