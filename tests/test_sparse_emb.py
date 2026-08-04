"""Check the sparse puzzle-embedding optimizer scatters per-id and descends."""
import sys, torch
sys.path.insert(0, "trm_base"); sys.path.insert(0, ".")
from sparse_embedding import _sparse_emb_signsgd_dist

N, D, NUM_IDS = 8, 4, 3
LR, WD = 0.1, 0.0

torch.manual_seed(0)
weights = torch.zeros(NUM_IDS, D)
# ids 0,1,2 repeated; row gradients all +1 so sign(sum) = +1 for every touched id
ids = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.int32)
g = torch.ones(N, D)

_sparse_emb_signsgd_dist(g, ids, weights, lr=LR, weight_decay=WD, world_size=1)

print("weights after one step (all grads +1):")
print(weights)
ok_desc = torch.allclose(weights, torch.full((NUM_IDS, D), -LR))
print(f"descends (all -{LR}): {ok_desc}")

# id-2 gets a negative gradient, ids 0/1 positive -> row 2 must move the other way
weights2 = torch.zeros(NUM_IDS, D)
g2 = torch.ones(N, D)
g2[ids == 2] = -1.0
_sparse_emb_signsgd_dist(g2, ids, weights2, lr=LR, weight_decay=WD, world_size=1)
print("\nweights after one step (id 2 grad -1):")
print(weights2)
ok_scatter = (torch.allclose(weights2[0], torch.full((D,), -LR))
              and torch.allclose(weights2[1], torch.full((D,), -LR))
              and torch.allclose(weights2[2], torch.full((D,), +LR)))
print(f"per-id scatter correct: {ok_scatter}")

# an untouched id must not move
weights3 = torch.zeros(NUM_IDS, D)
ids3 = torch.zeros(N, dtype=torch.int32)
_sparse_emb_signsgd_dist(torch.ones(N, D), ids3, weights3, lr=LR, weight_decay=WD, world_size=1)
ok_untouched = torch.allclose(weights3[1], torch.zeros(D)) and torch.allclose(weights3[2], torch.zeros(D))
print(f"\nuntouched ids unchanged: {ok_untouched}")

# weight decay applies to touched rows only
w4 = torch.ones(NUM_IDS, D)
_sparse_emb_signsgd_dist(torch.zeros(N, D), ids3, w4, lr=LR, weight_decay=1.0, world_size=1)
ok_wd = torch.allclose(w4[0], torch.full((D,), 1.0 - LR * 1.0)) and torch.allclose(w4[1], torch.ones(D))
print(f"decoupled weight decay on touched rows: {ok_wd}")

print("\nRESULT:", "PASS" if all([ok_desc, ok_scatter, ok_untouched, ok_wd]) else "FAIL")
