"""Does RoPE actually rotate? Two properties that only hold with the minus sign.

1. Norm preservation: a rotation cannot change ||q||.
2. Relative-position dependence: for a FIXED content vector placed at every
   position, q_m . k_n must depend only on (m - n). That translation
   invariance is the entire reason RoPE works.
"""
import torch

def rotate_half(x, negate):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2 if negate else x2, x1), dim=-1)

def apply(x, cos, sin, negate):
    return x * cos.unsqueeze(-2) + rotate_half(x, negate) * sin.unsqueeze(-2)

torch.manual_seed(0)
D, S = 64, 32
inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2).double() / D))
freqs = torch.outer(torch.arange(S).double(), inv)
emb = torch.cat((freqs, freqs), dim=-1)
cos, sin = emb.cos(), emb.sin()

# Same content at every position, so any variation in q.k is positional only.
qv = torch.randn(D, dtype=torch.float64)
kv = torch.randn(D, dtype=torch.float64)
q = qv.expand(1, S, 1, D).clone()
k = kv.expand(1, S, 1, D).clone()

for negate in (False, True):
    tag = "WITH minus (upstream)" if negate else "WITHOUT minus (our bug)"
    qe = apply(q, cos, sin, negate)
    ke = apply(k, cos, sin, negate)

    norm_err = (qe.norm(dim=-1) - q.norm(dim=-1)).abs().max().item()
    dots = torch.einsum("bmhd,bnhd->bmn", qe, ke)[0]

    off = 5
    vals = [dots[m + off, m].item() for m in range(S - off)]
    spread = max(vals) - min(vals)

    print(f"{tag}:")
    print(f"   max |norm change|                  = {norm_err:.3e}")
    print(f"   spread of q.k across offset {off}      = {spread:.3e}")
