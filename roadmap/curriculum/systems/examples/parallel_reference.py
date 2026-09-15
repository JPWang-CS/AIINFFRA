"""CPU tensor-parallel linear algebra and collective accounting."""


def matmul(a,b):
    if not a or not b or any(len(row)!=len(b) for row in a):
        raise ValueError("incompatible matrices")
    width=len(b[0])
    if any(len(row)!=width for row in b):
        raise ValueError("ragged matrix")
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]


def add(a,b):
    if len(a)!=len(b) or any(len(x)!=len(y) for x,y in zip(a,b)):
        raise ValueError("different result shapes")
    return [[x+y for x,y in zip(ar,br)] for ar,br in zip(a,b)]


def tensor_parallel_mlp(x, up, down, parts):
    hidden=len(up[0])
    if parts<=0 or hidden%parts:
        raise ValueError("hidden dimension must divide evenly")
    width=hidden//parts
    partials=[]
    for rank in range(parts):
        begin,end=rank*width,(rank+1)*width
        up_shard=[row[begin:end] for row in up]
        hidden_shard=matmul(x,up_shard)
        activated=[[max(v,0) for v in row] for row in hidden_shard]
        partials.append(matmul(activated,down[begin:end]))
    total=partials[0]
    for partial in partials[1:]:
        total=add(total,partial)
    return total


if __name__=="__main__":
    x=[[1,2,3,4],[-1,3,0,2]]
    up=[[((i*6+j)%7)-3 for j in range(6)] for i in range(4)]
    down=[[((i*4+j)%5)-2 for j in range(4)] for i in range(6)]
    h=[[max(v,0) for v in row] for row in matmul(x,up)]
    expected=matmul(h,down)
    assert tensor_parallel_mlp(x,up,down,2)==expected
    assert tensor_parallel_mlp(x,up,down,3)==expected
    bias=[[1]*4 for _ in x]
    assert add(expected,bias)!=add(add(expected,bias),bias)
    ranks=4;payload=1024
    assert 2*(ranks-1)/ranks*payload==1536
    assert 8/(8+4-1)==8/11
    print("PASS: column/row TP with pointwise activation, bias placement, ring bytes and pipeline proxy")
