"""Small scheduling and prefix-identity models, not a vLLM scheduler."""
import hashlib
import json


def allocate_tokens(demands, budget):
    if budget < 0 or any(n<0 for _,n in demands):
        raise ValueError("nonnegative demand and budget required")
    if len({name for name,_ in demands}) != len(demands):
        raise ValueError("request identities must be unique")
    assignments=[]
    for name,need in demands:
        take=min(need,budget)
        if take:
            assignments.append((name,take))
        budget-=take
    return assignments


def extra_pages(length, added, page_size):
    if min(length,added)<0 or page_size<=0:
        raise ValueError("invalid page request")
    return (length+added+page_size-1)//page_size-(length+page_size-1)//page_size


def prefix_identity(tokens, model_revision, adapter, position_base, tenant):
    # tokens is the complete prefix here; block-based caches also need parent identity.
    payload=[list(tokens),model_revision,adapter,position_base,tenant]
    return hashlib.sha256(json.dumps(payload,separators=(",",":")).encode()).hexdigest()


if __name__=="__main__":
    assert allocate_tokens([("decode",1),("prefill",8)],4)==[("decode",1),("prefill",3)]
    assert allocate_tokens([("prefill",8),("decode",1)],4)==[("prefill",4)]
    assert extra_pages(16,1,16)==1 and extra_pages(17,1,16)==0
    assert extra_pages(0,33,16)==3
    key=prefix_identity([1,2,3],"weights-a","none",0,"tenant-a")
    assert key==prefix_identity([1,2,3],"weights-a","none",0,"tenant-a")
    for args in [
        ([1,2,3],"weights-b","none",0,"tenant-a"),
        ([1,2,3],"weights-a","adapter-b",0,"tenant-a"),
        ([1,2,3],"weights-a","none",1,"tenant-a"),
        ([1,2,3],"weights-a","none",0,"tenant-b"),
        ([9,2,3],"weights-a","none",0,"tenant-a"),
    ]:
        assert prefix_identity(*args)!=key
    print("PASS: scheduling order/budget, page growth, full-prefix/model/adapter/position/tenant identity")
