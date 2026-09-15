"""Small mathematical checks for the paper reader, not production kernels."""
import numpy as np


def softmax(x):
    mass=np.exp(x-np.max(x,axis=-1,keepdims=True))
    return mass/np.sum(mass,axis=-1,keepdims=True)


def mla_content(q, latent, wk, wv):
    keys=np.einsum("tc,hcd->htd",latent,wk)
    values=np.einsum("tc,hcv->htv",latent,wv)
    direct_scores=np.einsum("hd,htd->ht",q,keys)
    absorbed_q=np.einsum("hd,hcd->hc",q,wk)
    absorbed_scores=np.einsum("hc,tc->ht",absorbed_q,latent)
    p=softmax(direct_scores/np.sqrt(q.shape[-1]))
    direct_out=np.einsum("ht,htv->hv",p,values)
    absorbed_p=softmax(absorbed_scores/np.sqrt(q.shape[-1]))
    latent_out=np.einsum("ht,tc->hc",absorbed_p,latent)
    absorbed_out=np.einsum("hc,hcv->hv",latent_out,wv)
    return direct_scores,absorbed_scores,direct_out,absorbed_out


def selected_attention(scores, values, selected):
    ids=np.asarray(selected,dtype=np.int64)
    if ids.ndim!=1 or len(ids)==0 or len(set(ids.tolist()))!=len(ids):
        raise ValueError("nonempty unique selection required")
    if np.any(ids<0) or np.any(ids>=len(scores)):
        raise ValueError("selection outside cache")
    return softmax(scores[ids])@values[ids]


if __name__=="__main__":
    rng=np.random.default_rng(8)
    q=rng.normal(size=(3,4));c=rng.normal(size=(7,2))
    wk=rng.normal(size=(3,2,4));wv=rng.normal(size=(3,2,5))
    s1,s2,y1,y2=mla_content(q,c,wk,wv)
    np.testing.assert_allclose(s1,s2,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(y1,y2,rtol=1e-12,atol=1e-12)
    scores=np.array([0.,1.,2.]);values=np.array([[0.],[1.],[9.]])
    dense=softmax(scores)@values
    sparse=selected_attention(scores,values,[1,2])
    assert not np.allclose(dense,sparse)
    np.testing.assert_allclose(selected_attention(scores,values,[0,1,2]),dense)
    # Unequal local normalizers must reweight both numerator and denominator.
    alpha=1/3
    assert abs((alpha*2+10)/(alpha+1)-8)<1e-12
    print("PASS: MLA score/output absorption; sparse selection is not dense identity; online merge")
