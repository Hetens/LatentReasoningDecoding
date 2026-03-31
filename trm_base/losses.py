from typing import Any, Tuple, Dict, Sequence, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

IGNORE_LABEL_ID = -100

def s(x: torch.Tensor, epsilon: float = 1e-30):
    return torch.where(x<0, 1/(1-x+epsilon), x+1)

def log_stablemax(x: torch.Tensor, dim: int = -1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim = dim, keepdim = True))

def stablemax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, ignore_index:int =-100, valid_mask= None):
    log_probs = log_stablemax(logits.to(torch.float64), dim =-1)

    if valid_mask is None:
        valid_mask =(labels != ignore_index)

    transformed_labels = torch.where(valid_mask,labels, 0).to(torch.int64)
    prediction_logprobs = torch.gather(log_probs, dim=-1, index=transformed_labels.unsqueeze(-1)).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)

def softmax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, ignore_index:int =-100):
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction='none').view(labels.shape)

class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type:str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs) # type ignore

    def forward(self, return_keys:Sequence[str], **kwarg) ->Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]],torch.Tensor]:
        #model logits
        # B * SeqLen * D_vocab
        new_carry, outputs =self.model(**kwarg)
        labels = new_carry.current_data['labels']

        logits = outputs["logits"]
        mask = labels != IGNORE_LABEL_ID
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
        is_correct = mask & (torch.argmax(logits, dim=-1) == labels)
        seq_is_correct = is_correct.sum(-1) == loss_counts

        # Must stay outside no_grad so backward() sees a grad_fn (upstream wraps this block — breaks PyTorch training).
        lm_loss = (self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum"
        )
        q_continue_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum"
            )
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

        with torch.no_grad():
            outputs["preds"] = torch.argmax(logits, dim=-1)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics: Dict[str, torch.Tensor] = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
                "lm_loss": lm_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            }
            if "target_q_continue" in outputs:
                metrics["q_continue_loss"] = q_continue_loss.detach()

            detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()