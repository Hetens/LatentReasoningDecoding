from dataclasses import dataclass
from jaxtyping import Tuple, List, Dict, Optional
from torch import nn
import copy
import torch
import math
from pydantic import BaseModel
import random
from torch.nn import functional as F
from .sparse_embedding import CastedSparseEmbedding
from .layers import CastedLinear, CastedEmbedding, RotaryEmbedding, Attention, SwiGLU, rms_norm, CosSin
from .losses import ACTLossHead
from .common import trunc_normal_init_

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H : torch.Tensor
    z_L : torch.Tensor

@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]

class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size:int
    seq_len:int
    puzzle_emb_ndim:int
    num_puzzle_identifiers:int
    vocab_size:int

    H_cycles:int
    L_cycles:int

    H_layers:int
    L_layers:int

    #transformer config
    hidden_size:int
    expansion: float
    num_heads:int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    #Halting Q-Learning config
    halt_max_steps:int
    halt_exploration_prob: float

    forward_dtype:str = "bfloat16"

    #Alexia?? added
    mlp_t:bool = False
    puzzle_emb_len:int = 16
    no_ACT_continue: bool = True

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config):
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim// -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        
            self.mlp_t = SwiGLU(hidden_size = self.config.seq_len + self.puzzle_emb_len, expansion = self.config.expansion)

        else:
            self.self_attn = Attention(
                hidden_size = config.hidden_size,
                head_dim = config.hidden_size // config.num_heads, 
                num_heads = config.num_heads,
                #what is the point of this?
                num_key_value_heads = config.num_heads,
                causal = False)
        
        self.mlp = SwiGLU(
            hidden_size = config.hidden_size,
            expansion = config.expansion
            )

        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        #B L D = hidden_states.shape
        # Post Norm

        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states =rms_norm(hidden_states + out, variance_epsilon=self.norm_eps )
            hidden_states = hidden_states.transpose(1,2)
        else:
            #self attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin = cos_sin, hidden_states = hidden_states), variance_epsilon=self.norm_eps)
        
        # fully connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon = self.norm_eps)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
    
    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection

        for layer in self.layers:
            hidden_states = layer(hidden_states = hidden_states, **kwargs)

        return hidden_states

class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        #I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0/self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std = embed_init_std, cast_to = self.forward_dtype)

        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias = False)

        self.q_head = CastedLinear(self.config.hidden_size, 2, bias = False)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len

        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim, self.config.batch_size, init_std = 0, cast_to = self.forward_dtype)
        

        #LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim = self.config.hidden_size// self.config.num_heads, max_position_embeddings = self.config.seq_len + self.puzzle_emb_len, base = self.config.rope_theta)

        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len,
            self.config.hidden_size, init_std = embed_init_std, cast_to = self.forward_dtype)
        else:
            pass

        #Reasoning LAyers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers = [TinyRecursiveReasoningModel_ACTV1Block(config = self.config) for _ in range(self.config.L_layers)])

        # initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype = self.forward_dtype),std = 1), persistent=True)

        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype = self.forward_dtype),std = 1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    
    def _input_embeddings(self, input:torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens(input.to(torch.int32))

        #Puzzle Embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len*self.config.hidden_size - puzzle_embedding.shape[-1]

            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size),embedding), dim = -2)

        #Positional Encodings
        if self.config.pos_encoding == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H = torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype = self.forward_dtype),
            z_L = torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype = self.forward_dtype),
        )
    
    def reset_carry(self,reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H = torch.where(reset_flag.view(-1,1,1), self.H_init, carry.z_H),
            z_L = torch.where(reset_flag.view(-1,1,1), self.L_init, carry.z_L),
        )
    
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None)

        #input encoding
        input_embeddings = self._input_embeddings(input = batch["inputs"], puzzle_identifiers = batch["puzzle_identifiers"])

        #forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L

        #h_cycles -1 without grad

        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                z_H = self.H_level(z_H, z_L, **seq_info)
            
        #1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        #LM outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H= z_H.detach(), z_L= z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        #Q-Head; uses the first puzzle embedding to decide whether to halt or continue
        q_logits = self.q_head(z_H[:,0]).to(torch.float32)

        return new_carry, output, (q_logits[...,0], q_logits[...,1])
    
class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    def __init__(self, config_dict:dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(config = self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch:Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry = self.inner.empty_carry(batch_size),
            steps = torch.zeros(batch_size, dtype = torch.int32),
            halted = torch.zeros(batch_size, dtype = torch.bool),
            current_data = {
                k: torch.empty_like(v) for k,v in batch.items()
            }
        )
    
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry,dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {k:torch.where(carry.halted.view((-1,) + (1,)*(batch[k].ndim -1)), batch[k], v) for k,v in carry.current_data.items()}

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            #Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # if training, and ACT is enabled.
            if self.training and (self.config.halt_max_steps > 1):

                #halt signal
                #note: during evaluation, always use max steps, this is to garuntee same halting, steps inside a batch for batching purpose.

                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits >= 0)
                else:
                    halted = halted | (q_halt_logits> q_continue_logits)

                #exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)* torch.randint_like(new_steps, low = 2, high = self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # compute target q
                    # note: no replay buffer and  target networks for comptuting target Q-values
                    # as batch_size is large theyre many paralllel envs

                    # similart concept as pqn https://arxiv.org/abs/2407.04811

                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_inner_carry, new_current_data)

                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits,torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry = new_inner_carry,
            steps = new_steps,
            halted = halted,
            current_data = new_current_data
        ), outputs