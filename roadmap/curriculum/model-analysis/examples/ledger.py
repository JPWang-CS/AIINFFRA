"""Dense/GQA Transformer-block arithmetic, not measured GPU performance."""
import json


def block_ledger(batch, sequence, layers, model_dim, query_heads, kv_heads, head_dim, intermediate, element_bytes):
    dimensions = [batch,sequence,layers,model_dim,query_heads,kv_heads,head_dim,intermediate,element_bytes]
    if any(not isinstance(x,int) or x <= 0 for x in dimensions):
        raise ValueError("positive integer dimensions required")
    if model_dim != query_heads*head_dim or query_heads % kv_heads:
        raise ValueError("inconsistent attention dimensions")
    linear_params = 2*model_dim**2 + 2*model_dim*kv_heads*head_dim + 3*model_dim*intermediate
    return {
        "linear_params_per_block": linear_params,
        "block_weight_bytes": layers*linear_params*element_bytes,
        "prefill_linear_flops": 2*batch*sequence*layers*linear_params,
        "prefill_causal_attention_flops": 4*batch*layers*query_heads*head_dim*(sequence*(sequence+1)//2),
        "decode_linear_flops": 2*batch*layers*linear_params,
        "decode_attention_flops": 4*batch*layers*query_heads*head_dim*sequence,
        "kv_payload_bytes": 2*batch*layers*sequence*kv_heads*head_dim*element_bytes,
    }


if __name__ == "__main__":
    one=block_ledger(1,4096,32,4096,32,8,128,11008,2)
    eight=block_ledger(8,4096,32,4096,32,8,128,11008,2)
    assert one["linear_params_per_block"] == 177209344
    assert one["kv_payload_bytes"] == 512*1024**2
    assert eight["block_weight_bytes"] == one["block_weight_bytes"]
    assert eight["decode_linear_flops"] == 8*one["decode_linear_flops"]
    assert eight["kv_payload_bytes"] == 8*one["kv_payload_bytes"]
    small=block_ledger(1,1,1,64,4,2,16,96,4)
    assert small["prefill_causal_attention_flops"] == small["decode_attention_flops"]
    print(json.dumps(one,indent=2))
    print("PASS: block-only ledger, batch scaling, causal S=1 boundary")
