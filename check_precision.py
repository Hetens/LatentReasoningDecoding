import torch, numpy as np
from experiments.probing.extract_activations import load_trm_model, load_test_data

dev = torch.device("cpu")
CKPT = "checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/TinyRecursiveReasoningModel_ACTV1 analytic-cobra/step_65100.pt"
model = load_trm_model("trm_base/config_pretrain_paper.yml", CKPT, dev, "data/sudoku-extreme-1k-aug-1000", "test")
model.eval(); inner = model.inner; cfg = model.config
raw = load_test_data("data/sudoku-extreme-1k-aug-1000", "test", max_examples=8)
b = {k: torch.from_numpy(raw[k][:4].astype(np.int32)) for k in ("inputs","labels","puzzle_identifiers")}

print("bf16 eps (relative spacing):", torch.finfo(torch.bfloat16).eps)
print("fp32 eps:", torch.finfo(torch.float32).eps)

@torch.no_grad()
def run(dtype, n_seg=40):
    m = model
    if dtype == torch.float32:
        m = m.float()
        inner_ = m.inner
        inner_.forward_dtype = torch.float32
    else:
        inner_ = m.inner
    si = dict(cos_sin=inner_.rotary_emb() if hasattr(inner_,"rotary_emb") else None)
    if dtype == torch.float32 and si["cos_sin"] is not None:
        si["cos_sin"] = tuple(t.float() for t in si["cos_sin"])
    emb = inner_._input_embeddings(b["inputs"], b["puzzle_identifiers"]).to(dtype)
    c = inner_.empty_carry(4)
    c = inner_.reset_carry(torch.ones(4,dtype=torch.bool), c)
    zH, zL = c.z_H.to(dtype), c.z_L.to(dtype)
    out=[]
    for s in range(n_seg):
        rs=[]
        for _T in range(cfg.H_cycles):
            for _i in range(cfg.L_cycles):
                new = inner_.L_level(zL, zH+emb, **si)
                r = ((new-zL).flatten(1).float().norm(dim=1)/zL.flatten(1).float().norm(dim=1).clamp(min=1e-9))
                rs.append(r.mean().item()); zL = new
            zH = inner_.L_level(zH, zL, **si)
        out.append(np.mean(rs))
    return np.array(out)

r16 = run(torch.bfloat16)
print("\nbf16 residual by segment:", " ".join(f"{v:.4f}" for v in r16[[0,1,3,7,15,31,39]]))
r32 = run(torch.float32)
print("fp32 residual by segment:", " ".join(f"{v:.4f}" for v in r32[[0,1,3,7,15,31,39]]))
print(f"\nbf16 floor {r16[-5:].mean():.5f}  vs  fp32 floor {r32[-5:].mean():.5f}")
