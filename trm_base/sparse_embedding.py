from jaxtyping import Union
import torch
from torch import nn
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, ParamsT
from .common import trunc_normal_init_

class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int,batch_size: int, init_std: float, cast_to:torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        #real weights
        #trunc normal init
        self.weights = nn.Buffer(trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)),std=init_std), persistent=True)

        #Local Weights and IDs
        # Local Embeddings, with gradients, not persistent
        self.local_weights = nn.Buffer(torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False)

        #Local embedding IDs, not persisitent
        self.local_ids = nn.Buffer(torch.zeros(batch_size, dtype=torch.int32), persistent=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # test mode, no gradient
            return self.weights[input].to(self.cast_to)

        else:
            with torch.no_grad():
                self.local_weights.copy_(self.weights[input])
                self.local_ids.copy_(input)
            
            return self.local_weights.to(self.cast_to)


class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
    def __init__(self, params: ParamsT, world_size: int, lr: Union[float, torch.Tensor] = 1e-3, weight_decay:float = 1e-2):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
    
        defaults = dict(lr=lr, weight_decay=weight_decay, world_size=world_size)
        super().__init__(params, defaults)

    
    @torch.no_grad()
    def step(self, closure = None):
        for group in self.param_groups:
            #find the sparse embedding weights
            local_weights_grad = None
            local_ids = None
            weights = None

            assert len(group['params']) == 3
            for p in group['params']:
                if p.requires_grad:
                    local_weights_grad = p.grad

                elif p.ndim == 1:
                    local_ids = p
                elif p.ndim == 2:
                    weights = p
                else:
                    assert False

            assert local_ids is not None
            assert weights is not None

            #Apply SignSGD
            # Adam \approx SignSGD if  gradiant is very sparse

            if local_weights_grad is not None:
                _sparse_emb_signsgd_dist(
                    local_weights_grad,
                    local_ids, 
                    weights,

                    lr = group['lr'],
                    weight_decay = group['weight_decay'],
                    world_size = group['world_size'],
                )
            

def _sparse_emb_signsgd_dist( local_weights_grad: torch.Tensor, local_ids: torch.Tensor, weights: torch.Tensor, lr: float, weight_decay: float, world_size: int) -> None:
    N,D = local_weights_grad.shape

    #all gather
    all_weights_grad = local_weights_grad
    all_ids = local_ids

    if world_size > 1:
        all_weights_grad =torch.empty((world_size*N, D), dtype =local_weights_grad.dtype, device =local_weights_grad.device)
        all_ids = torch.empty((world_size*N,), dtype =local_ids.dtype, device =local_ids.device)

        dist.all_gather_into_tensor(all_weights_grad, local_weights_grad)
        dist.all_gather_into_tensor(all_ids, local_ids)

    #unique
    grad_ids, inv = all_ids.unique(return_inverse = True)
    grad = torch.zeros((grad_ids.shape[0], D), dtype =local_weights_grad.dtype, device =local_weights_grad.device)

    grad.scatter_add_(0, inv.unsqueeze(-1).expand(1,D), all_weights_grad)


    #signSGD decoupled with weight decay
    p = weights[grad_ids]

    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha = lr)

    weights[grad_ids] = p