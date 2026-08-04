from typing import Optional, Any, Sequence, List, Tuple
from dataclasses import dataclass
import argparse
import os
import math
import yaml
import shutil
import numpy as np
import copy
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Upstream TRM uses AdamATan2 (pretrain.py: `from adam_atan2 import AdamATan2`),
# not AdamW. The difference is not cosmetic: AdamW's update is m/(sqrt(v)+eps),
# so as gradients shrink the update collapses and the per-step decoupled weight
# decay takes over, which is the frozen-loss equilibrium the maze runs hit.
# AdamATan2's atan2(m, sqrt(v)) is bounded and scale-invariant, so the update
# stays ~lr regardless of gradient scale. Imported lazily so the AdamW path
# still runs on machines without the CUDA extension built.
try:
    from adam_atan2 import AdamATan2
except ImportError:  # pragma: no cover - depends on the CUDA extension
    AdamATan2 = None

import tqdm
import wandb
import coolname
import pydantic

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from functions import load_model_class, get_model_source_path
from sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from ema import EMAHelper

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def _should_torch_compile() -> bool:
    """``torch.compile`` / Inductor on CPU often requires a C++ compiler (MSVC ``cl`` on Windows)."""
    if os.environ.get("DISABLE_COMPILE", "").strip().lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("FORCE_TORCH_COMPILE", "").strip().lower() in ("1", "true", "yes"):
        return True
    return torch.cuda.is_available()

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig

class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str

class PretrainConfig(pydantic.BaseModel):
    #arch
    arch: ArchConfig
    data_paths: List[str]
    data_paths_test: List[str]
    #evaluators
    evaluators: List[EvaluatorConfig] = []

    #hyperparams
    # "adam_atan2" matches upstream TRM; "adamw" is the old behaviour, kept so
    # the two can be compared without swapping branches.
    optimizer: str = "adam_atan2"
    global_batch_size: int
    # Rows per forward/backward. When smaller than global_batch_size, gradients
    # are accumulated over global_batch_size // micro_batch_size micro-batches
    # before each optimizer step, so a large effective batch fits on one GPU.
    # None means no accumulation (micro == global), the original behaviour.
    micro_batch_size: Optional[int] = None
    epochs: int

    lr:float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1:float
    beta2:float

    #puzzle embeddings training
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # names (omit from YAML to auto-fill in load_synced_config on rank 0)
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    #extras
    seed: int =0
    # Cap rows loaded from disk (None = use full split). Train/test use separate limits.
    max_train_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0
    eval_save_outputs: List[str] = []

    #ema
    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(config: dict, overrides: Sequence[str]) -> dict:
    """Apply Hydra-style ``key=value`` / ``group.key=value`` CLI overrides.

    Values are parsed as YAML scalars so ``epochs=12800`` yields an int and
    ``ema=false`` a bool, matching how they would be written in the file.
    """
    out = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, _, raw = item.partition("=")
        parts = key.split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise TypeError(f"Override {item!r} descends into non-mapping at {p!r}")
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def load_composed_config(config_path: str) -> dict:
    """Load a YAML config with optional Hydra-style ``defaults`` list (no hydra-core needed)."""
    config_path = os.path.abspath(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        root = yaml.safe_load(f)
    if root is None:
        root = {}
    if not isinstance(root, dict):
        raise TypeError(f"Config root must be a mapping, got {type(root)}")

    root.pop("hydra", None)
    defaults = root.pop("defaults", None)
    merged: dict = {}

    if isinstance(defaults, list):
        for entry in defaults:
            if entry == "_self_":
                continue
            if not isinstance(entry, dict):
                continue
            for group, name in entry.items():
                base_dir = os.path.dirname(config_path)
                for ext in (".yaml", ".yml"):
                    subpath = os.path.join(base_dir, group, f"{name}{ext}")
                    if os.path.isfile(subpath):
                        break
                else:
                    raise FileNotFoundError(
                        f"Config composition: no file for '{group}: {name}' under {base_dir}/{group}/"
                    )
                with open(subpath, "r", encoding="utf-8") as sf:
                    sub = yaml.safe_load(sf)
                if sub is None:
                    sub = {}
                if not isinstance(sub, dict):
                    raise TypeError(f"{subpath} must contain a mapping at the top level")
                merged = deep_merge(merged, {group: sub})

    return deep_merge(merged, root)


@dataclass
class TrainState:
    model:nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps:int

    # Gradient accumulation. `step` counts optimizer steps, not micro-batches,
    # so the LR schedule, total_steps and checkpoint names stay on the same
    # scale they had before accumulation existed.
    grad_accum: int = 1
    micro_step: int = 0
    accum_metrics: Optional[dict] = None


def resolve_batching(config: PretrainConfig) -> Tuple[int, int]:
    """Return ``(micro_batch_size, grad_accum_steps)``.

    The dataloader and the model are sized by the micro batch; the loss scale
    and the LR schedule stay tied to the full ``global_batch_size``.
    """
    micro = config.micro_batch_size or config.global_batch_size
    if micro <= 0 or config.global_batch_size % micro != 0:
        raise ValueError(
            f"global_batch_size ({config.global_batch_size}) must be a positive "
            f"multiple of micro_batch_size ({micro})")
    return micro, config.global_batch_size // micro


def create_dataloader(config: PretrainConfig, split: str , rank:int, world_size:int, **kwargs):
    # Keys for PuzzleDatasetConfig only — DataLoader must not receive them.
    test_set_mode = kwargs.pop("test_set_mode")
    epochs_per_iter = kwargs.pop("epochs_per_iter")
    global_batch_size = kwargs.pop("global_batch_size")

    dataset_paths = (
        config.data_paths_test
        if test_set_mode and len(config.data_paths_test) > 0
        else config.data_paths
    )
    max_examples = (
        config.max_test_samples if test_set_mode else config.max_train_samples
    )
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed = config.seed,
        dataset_paths = dataset_paths,
        rank = rank,
        num_replicas = world_size,
        global_batch_size = global_batch_size,
        test_set_mode = test_set_mode,
        epochs_per_iter = epochs_per_iter,
        max_examples = max_examples,
    ), split = split)

    dataloader = DataLoader(dataset, batch_size = None, num_workers = 1, prefetch_factor = 8, pin_memory = True, persistent_workers = True, **kwargs)

    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank:int, world_size:int):
    # Tensors are micro-batch sized; only the optimizer sees the full batch.
    micro_batch_size, grad_accum = resolve_batching(config)

    # CastedSparseEmbedding.forward overwrites local_ids on every forward, but
    # local_weights.grad accumulates. Across micro-batches the optimizer would
    # therefore scatter the summed gradient onto only the LAST micro-batch's
    # ids. Harmless when every row shares one identifier (maze: 1), silently
    # wrong otherwise, so refuse rather than corrupt the embedding.
    if (grad_accum > 1 and config.arch.puzzle_emb_ndim > 0
            and train_metadata.num_puzzle_identifiers > 1):
        raise NotImplementedError(
            f"Gradient accumulation (grad_accum={grad_accum}) is not supported with "
            f"a multi-identifier puzzle embedding "
            f"(num_puzzle_identifiers={train_metadata.num_puzzle_identifiers}). "
            "CastedSparseEmbeddingSignSGD_Distributed would misattribute the "
            "accumulated gradient. Set micro_batch_size=global_batch_size.")

    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size = micro_batch_size // world_size,
        vocab_size = train_metadata.vocab_size,
        seq_len = train_metadata.seq_len,
        num_puzzle_identifiers = train_metadata.num_puzzle_identifiers, 
        causal = False, #non-autoregressive, but i think i want to try autoregressive
    )

    #Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)
        if _should_torch_compile():
            model = torch.compile(model)
        

        # load checkpoint if specified
        if rank == 0:
            load_checkpoint(model, config)

        #broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param.data, src = 0)
            
    # Dense optimizer: AdamATan2 upstream, AdamW only if explicitly asked for.
    if config.optimizer == "adam_atan2":
        if AdamATan2 is None:
            raise ImportError(
                "optimizer='adam_atan2' but the adam_atan2 package is not "
                "importable. Build it with scripts/unity/setup_adam_atan2.sh, "
                "or set optimizer=adamw to use the old path.")
        dense_optim_cls = AdamATan2
    elif config.optimizer == "adamw":
        dense_optim_cls = AdamW
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer!r}; expected "
                         "'adam_atan2' or 'adamw'")
    if rank == 0:
        print(f"Dense optimizer: {dense_optim_cls.__name__}", flush=True)

    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            dense_optim_cls(
                model.parameters(),
                lr = 0,
                weight_decay =config.weight_decay,
                betas = (config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [config.lr]

    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [config.puzzle_emb_lr]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # needs to be set by scheduler,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            dense_optim_cls(
                model.parameters(),
                lr = 0,  # set by the scheduler, as upstream does
                weight_decay =config.weight_decay,
                betas = (config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [config.puzzle_emb_lr, config.lr]
    
    return model, optimizers, optimizer_lrs


def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1, len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        
        sd_alpha[k] = comb_net
    
    net.load_state_dict(sd_alpha)
    return net

def cosine_scheduler_with_warmup_lr_lambda(current_step: int, *, base_lr:float,num_warmup_steps:int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

    return base_lr*(min_ratio + max(0.0, (1- min_ratio)*0.5*(1.0 + math.cos(math.pi*float(num_cycles)*2.0*progress))))

def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank:int, world_size:int):


    # Estimated total training steps. Counted in optimizer steps, so this is
    # unaffected by gradient accumulation.
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    _, grad_accum = resolve_batching(config)

    #Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank = rank, world_size = world_size)

    return TrainState(
        step = 0,
        total_steps = total_steps,

        model = model,
        optimizers = optimizers,
        optimizer_lrs = optimizer_lrs,
        carry = None,
        grad_accum = grad_accum,
    )

def save_train_state(config: PretrainConfig, train_state: TrainState):
    #FIXME: Only saved Model
    if config.checkpoint_path is None:
        return
    
    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}.pt"))


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint from {config.load_checkpoint}")

        state_dict = torch.load(config.load_checkpoint, map_location=device)

        # Checkpoints are saved from a torch.compile'd model, so keys carry an
        # "_orig_mod." prefix. Reconcile it against the target model rather
        # than assuming, so a checkpoint can be reloaded either way round
        # (the analysis scripts load these same files uncompiled).
        target_keys = model.state_dict().keys()
        has_prefix = any(k.startswith("_orig_mod.") for k in state_dict)
        wants_prefix = any(k.startswith("_orig_mod.") for k in target_keys)
        if has_prefix and not wants_prefix:
            state_dict = {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
        elif wants_prefix and not has_prefix:
            state_dict = {"_orig_mod." + k: v for k, v in state_dict.items()}

        #resize and reset puzzle emb if needed
        # Guarded: configs with puzzle_emb_ndim=0 (Sudoku, Maze) have no
        # puzzle_emb module at all, so reading its shape unconditionally
        # crashes every resume on those tasks.
        puzzle_emb_name = next(
            (k for k in state_dict if k.endswith("inner.puzzle_emb.weights")), None)
        puzzle_emb_mod = getattr(getattr(model, "model", model).inner, "puzzle_emb", None)
        if puzzle_emb_name is not None and getattr(puzzle_emb_mod, "weights", None) is not None:
            expected_shape = puzzle_emb_mod.weights.shape
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different: Found {puzzle_emb.shape}, expected {expected_shape}")

                # Reinitialize using mean
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous())

        model.load_state_dict(state_dict, assign = True)

def compute_lr(base_lr:float, config: PretrainConfig, train_state:TrainState):
    return cosine_scheduler_with_warmup_lr_lambda(
        current_step = train_state.step,
        base_lr = base_lr,
        num_warmup_steps = round(config.lr_warmup_steps),
        num_training_steps = train_state.total_steps,
        min_ratio = config.lr_min_ratio,
    )

def create_evaluators(config: PretrainConfig, eval_metadata:PuzzleDatasetMetadata) -> List[Any]:
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    evaluators = []

    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path = data_path, eval_metadata = eval_metadata, **cfg.__pydantic_extra__
            )
            evaluators.append(cls)
    
    return evaluators

def train_batch(config: PretrainConfig, train_state:TrainState, batch: Any, rank: int, world_size: int):
    """Run one micro-batch, stepping the optimizers every ``grad_accum`` calls.

    Returns ``(metrics, stepped)``. ``metrics`` is non-None only on rank 0 and
    only at an optimizer step, where it aggregates every micro-batch since the
    previous step. With ``grad_accum == 1`` this is the original behaviour.
    """
    if train_state.step >= train_state.total_steps:
        return None, False

    # to device
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    # Init Carry if it is none
    if train_state.carry is None:
        with torch.device(device):
            train_state.carry = train_state.model.initial_carry(batch)

    # forward
    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=[]
    )

    # Scale by the FULL batch, not the micro-batch, so the gradients summed
    # across grad_accum micro-batches are the mean over global_batch_size.
    ((1/config.global_batch_size) * loss).backward()

    # Hold metrics until the step so they cover the whole effective batch.
    if len(metrics) > 0:
        assert not any(v.requires_grad for v in metrics.values())
        if train_state.accum_metrics is None:
            train_state.accum_metrics = {k: v.detach().clone() for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                train_state.accum_metrics[k] += v.detach()

    train_state.micro_step += 1
    if train_state.micro_step < train_state.grad_accum:
        return None, False

    train_state.micro_step = 0
    train_state.step += 1

    #All reduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    #Apply optimizers
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step

        optim.step()
        optim.zero_grad()

#reduce metrics
    accum, train_state.accum_metrics = train_state.accum_metrics, None
    if accum:
        # reduce
        metrics_keys = list(sorted(accum.keys()))
        metric_values = torch.stack([accum[k] for k in metrics_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst = 0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k:metric_values[i] for i, k in enumerate(metrics_keys)}

            #postprocess
            count = max(reduced_metrics["count"], 1) # to avoid nans
            reduced_metrics = {f"train/{k}": v /(config.global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step

            return reduced_metrics, True

    return None, True


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup]
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)

        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # run evaluation
        set_ids= {k:idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}
        metrics_keys = []
        metric_values = None

        carry = None
        processed_batches = 0


        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processed batch {processed_batches}: {set_name}")

            # to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.device(device):
                carry = train_state.model.initial_carry(batch)
            
            # forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(carry = carry, batch = batch, return_keys = return_keys)

                inference_steps += 1
                if all_finish:
                    break
            
            if rank == 0:
                print(f" Completed Inference in {inference_steps} steps")
            
            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds[k].setdefault(k, [])
                        save_preds[k].append(v.cpu())
            
            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)
            
            del carry, loss, preds, batch, all_finish

            #Aggregate metrics
            set_id  = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype = torch.float32, device = device)
            
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        
        # concatenate save preds
        save_preds = {k: torch.cat(v, dim = 0) for k, v in save_preds.items()}

        # save preds
        if config.checkpoint_path is not None and len(save_preds):
            #Each rank save predictions independently
            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.pt"))

        del save_preds

        # reduce rank to 0
        if metric_values is not None:
            if world_size>1:
                dist.reduce(metric_values, dst = 0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = { 
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id] for metric_id, metric_name in enumerate(metric_keys)
                        } 
                    for set_id, set_name in enumerate(eval_metadata.sets) 
                    }
                #postprocess
                for set_name, m in reduced_metrics.items():
                    count= m.pop("count")
                    reduced_metrics[set_name] = {k:v/count for k, v in m.items()}

        
        #run evaluators
        if rank == 0:
            print(f"\n Running {len(evaluators)} evaluators .....")
        
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")

            #Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(config.checkpoint_path, f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}")
                os.makedirs(evaluator_save_path, exist_ok=True)

            #Run and LOg

            metrics = evaluator.result(evaluator_save_path, rank = rank, world_size = world_size, group = cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}
                reduced_metrics.update(metrics)
                print(f" Completed {evaluator.__class__.__name__} evaluation")
        
        if rank == 0:
            print("All evaluators completed")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return
    
    os.makedirs(config.checkpoint_path, exist_ok=True)

    #copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name),
    ]

    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))
    
    #dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)

def load_synced_config(raw_config: dict, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig.model_validate(raw_config)

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


def launch(config_path: str, overrides: Sequence[str] = ()) -> None:
    raw_config = load_composed_config(config_path)
    if overrides:
        raw_config = apply_overrides(raw_config, overrides)
        print(f"Config overrides: {list(overrides)}")
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(raw_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    # Loaders yield micro-batches; train_batch accumulates them up to
    # global_batch_size before stepping. Eval never accumulates, so it just
    # uses the same memory-safe micro size.
    micro_batch_size, grad_accum = resolve_batching(config)
    if RANK == 0 and grad_accum > 1:
        print(f"Gradient accumulation: {grad_accum} x {micro_batch_size} = {config.global_batch_size}", flush=True)

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=micro_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=micro_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    if config.ema:
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        for set_name, batch, _micro_gbs in train_loader:
            metrics, stepped = train_batch(config, train_state, batch, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
            # Once per optimizer step, not per micro-batch, or ema_rate would
            # be applied grad_accum times as often.
            if config.ema and stepped:
                ema_helper.update(train_state.model)

        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config, 
                train_state_eval, 
                eval_loader, 
                eval_metadata, 
                evaluators,
                rank=RANK, 
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                # wandb runs offline on the cluster, so echo the eval metrics
                # to stdout: this is the only way to watch a run's progress
                # without post-hoc parsing of the .wandb file. metrics is
                # nested as {set_name: {metric: value}} and wandb flattens it
                # with "/", so do the same here rather than dropping it all.
                flat = {}
                for k, v in metrics.items():
                    if isinstance(v, dict):
                        flat.update({f"{k}/{kk}": vv for kk, vv in v.items()})
                    else:
                        flat[k] = v
                print("EVAL step={} | {}".format(
                    train_state.step,
                    "  ".join(f"{k}={float(v):.4f}" for k, v in sorted(flat.items())
                              if isinstance(v, (int, float, np.floating, np.integer)))),
                    flush=True)

            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    _default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_pretrain.yml")
    _parser = argparse.ArgumentParser(description="Pretrain (YAML config; Hydra-style defaults supported without hydra-core).")
    _parser.add_argument("--config", type=str, default=_default_cfg, help="Path to main YAML config file.")
    _args, _overrides = _parser.parse_known_args()
    launch(_args.config, _overrides)