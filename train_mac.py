# /// script
# dependencies = [
#     "accelerate",
#     "adam-atan2-pytorch>=0.1.18",
#     "setuptools",
#     "titans-pytorch",
#     "tqdm",
#     "wandb"
# ]
# ///

import os
import sys
import random
import argparse
import importlib.util
import tqdm
import gzip
import numpy as np
from contextlib import nullcontext

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from adam_atan2_pytorch import AdoptAtan2

from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
    MemoryAttention
)

# load config from --config argument (default: configs/default.py)

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--config', default='configs/default.py')
_args, _ = _parser.parse_known_args()

def _load_config(path):
    _spec = importlib.util.spec_from_file_location('_config', os.path.abspath(path))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    for k, v in vars(_mod).items():
        if not k.startswith('_'):
            globals()[k] = v

_default_path = os.path.join(os.path.dirname(__file__), 'configs/default.py')
_load_config(_default_path)

_config_path = os.path.abspath(_args.config)
_config_name = os.path.splitext(os.path.basename(_config_path))[0]
if os.path.abspath(_config_path) != os.path.abspath(_default_path):
    _load_config(_config_path)

# RUN_NAME: config file name prefix + auto-generated suffix from config constants
# run name abbreviations: N=num_slots, k=top_k, h=heads, d=dim_head, +sh=shared_memory
#   lm=num_longterm_mem, ly=neural_mem_layers, sq=seq_len, bs=batch_size, ga=gradient_accumulate_every
_sparse_kda_router_loss_suffix = (f" rc={SPARSE_KDA_RECON_LOSS_WEIGHT}" if SPARSE_KDA_ROUTER_LOSS_TYPE == 'recon' and SPARSE_KDA_RECON_LOSS_WEIGHT > 0 else "") + (f" bal={SPARSE_KDA_BAL_LOSS_WEIGHT}" if SPARSE_KDA_ROUTER_LOSS_TYPE == 'bal' and SPARSE_KDA_BAL_LOSS_WEIGHT > 0 else "")
_sparse_kda_suffix = f' N={SPARSE_KDA_NUM_SLOTS} k={SPARSE_KDA_TOP_K}{" +sh" if SPARSE_KDA_USE_SHARED_MEMORY else ""} h={KDA_HEADS} d={KDA_DIM_HEAD}{_sparse_kda_router_loss_suffix}{ f" dl={SPARSE_KDA_DISTILL_LOSS_WEIGHT}@{SPARSE_KDA_DISTILL_EVERY}" if SPARSE_KDA_DISTILL_LOSS_WEIGHT > 0 else ""}{" +norm_q" if SPARSE_KDA_ROUTER_USE_NORMALIZED_Q else ""}' if MEMORY_TYPE == 'sparse_kda' else ''
_kda_suffix = f' h={KDA_HEADS} d={KDA_DIM_HEAD}' if MEMORY_TYPE == 'kda' else ''
RUN_NAME = f'[{_config_name}] {MEMORY_TYPE}{_sparse_kda_suffix}{_kda_suffix} ly={NEURAL_MEM_LAYERS} sq={SEQ_LEN} bs={BATCH_SIZE} ga={GRADIENT_ACCUMULATE_EVERY}'

# wandb experiment tracker

import wandb
wandb.init(project = PROJECT_NAME, name = RUN_NAME, mode = 'disabled' if not WANDB_ONLINE else 'online')

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def set_sparse_kda_oracle_debug(model, step):
    """Set oracle debug flag on all SparseKDAMemory modules — fires on the next forward pass."""
    from titans_pytorch.kda_memory import SparseKDAMemory
    for name, module in model.named_modules():
        if isinstance(module, SparseKDAMemory):
            module._oracle_debug_next = True
    tqdm.tqdm.write(f'\n[oracle debug scheduled @ step {step}]')

# memory model

if MEMORY_TYPE == 'kda':
    from titans_pytorch import create_kda_memory_for_mac

    print(f"Using KDA Memory (Kimi Delta Attention)")
    print(f"  - Chunk size: {KDA_CHUNK_SIZE}")
    print(f"  - Use chunk mode: {KDA_USE_CHUNK}")

    neural_memory_model = create_kda_memory_for_mac(
        dim = 64,
        chunk_size = KDA_CHUNK_SIZE,
        use_chunk = KDA_USE_CHUNK,
        heads = KDA_HEADS,
        dim_head = KDA_DIM_HEAD,
    )
elif MEMORY_TYPE == 'sparse_kda':
    from titans_pytorch.kda_memory import create_sparse_kda_memory_for_mac

    print(f"Using Sparse KDA Memory (SM-KDA)")
    print(f"  - Num memory slots (N): {SPARSE_KDA_NUM_SLOTS}")
    print(f"  - Top-k per token: {SPARSE_KDA_TOP_K}")
    print(f"  - Shared memory: {SPARSE_KDA_USE_SHARED_MEMORY}")

    neural_memory_model = create_sparse_kda_memory_for_mac(
        dim = 64,
        num_memory_slots = SPARSE_KDA_NUM_SLOTS,
        top_k = SPARSE_KDA_TOP_K,
        heads = KDA_HEADS,
        dim_head = KDA_DIM_HEAD,
        use_shared_memory = SPARSE_KDA_USE_SHARED_MEMORY,
        router_hidden = SPARSE_KDA_ROUTER_HIDDEN,
        router_loss_type = SPARSE_KDA_ROUTER_LOSS_TYPE,
        router_use_normalized_q = SPARSE_KDA_ROUTER_USE_NORMALIZED_Q,
    )
elif USE_MEM_ATTENTION_MODEL:
    print("Using Memory Attention Model")
    neural_memory_model = MemoryAttention(
        dim = 64
    )
else:
    print(f"Using Neural Memory (TTT-based MLP)")
    print(f"  - Depth: {NEURAL_MEMORY_DEPTH}")
    neural_memory_model = MemoryMLP(
        dim = 64,
        depth = NEURAL_MEMORY_DEPTH
    )

# instantiate memory-as-context transformer

model = MemoryAsContextTransformer(
    num_tokens = 256,
    dim = 384,
    depth = 8,
    segment_len = WINDOW_SIZE,
    num_persist_mem_tokens = NUM_PERSIST_MEM,
    num_longterm_mem_tokens = NUM_LONGTERM_MEM,
    neural_memory_layers = NEURAL_MEM_LAYERS,
    neural_memory_segment_len = NEURAL_MEM_SEGMENT_LEN,
    neural_memory_batch_size = NEURAL_MEM_BATCH_SIZE,
    neural_mem_gate_attn_output = NEURAL_MEM_GATE_ATTN_OUTPUT,
    neural_mem_weight_residual = NEURAL_MEM_WEIGHT_RESIDUAL,
    neural_memory_qkv_receives_diff_views = NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
    use_flex_attn = USE_FLEX_ATTN,
    sliding_window_attn = SLIDING_WINDOWS,
    neural_memory_model = neural_memory_model,
    neural_memory_kwargs = dict(
        dim_head = 64,
        heads = 4,
        attn_pool_chunks = STORE_ATTN_POOL_CHUNKS,
        qk_rmsnorm = NEURAL_MEM_QK_NORM,
        momentum = NEURAL_MEM_MOMENTUM,
        momentum_order = NEURAL_MEM_MOMENTUM_ORDER,
        default_step_transform_max_lr = NEURAL_MEM_MAX_LR,
        use_accelerated_scan = USE_ACCELERATED_SCAN,
        per_parameter_lr_modulation = MEMORY_MODEL_PER_LAYER_LEARNED_LR,
        spectral_norm_surprises = NEURAL_MEM_SPEC_NORM_SURPRISES
    )
).cuda()

if USE_AMP:
    model = model.bfloat16()

model = torch.compile(model)

if MEMORY_TYPE in ('sparse_kda', 'kda'):
    from titans_pytorch.kda_memory import SparseKDAMemory, KDAMemory
    for _, module in model.named_modules():
        if isinstance(module, (SparseKDAMemory, KDAMemory)):
            module.enable_residual_logging()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
    data_train, data_val = np.split(data, [int(90e6)])
    data_train, data_val = map(torch.from_numpy, (data_train, data_val))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers = 4, pin_memory = True))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE, num_workers = 4, pin_memory = True))

# optimizer

optim = AdoptAtan2(model.parameters(), lr = LEARNING_RATE)

# training here

os.makedirs(PROFILE_OUTPUT_DIR, exist_ok = True)

def print_profiler_summary(prof):
    """Print profiler summary tables to stdout (Slurm-friendly)."""
    print('\n' + '=' * 80)
    print('PROFILER RESULTS — Top 20 CUDA ops by total time')
    print('=' * 80)
    print(prof.key_averages().table(sort_by = 'cuda_time_total', row_limit = 20))

    print('\n' + '=' * 80)
    print('PROFILER RESULTS — Top 20 CPU ops by total time')
    print('=' * 80)
    print(prof.key_averages().table(sort_by = 'cpu_time_total', row_limit = 20))

    print('\n' + '=' * 80)
    print('PROFILER RESULTS — Top 10 ops by CUDA memory usage')
    print('=' * 80)
    print(prof.key_averages().table(sort_by = 'self_cuda_memory_usage', row_limit = 10))

    print('\n' + '=' * 80)
    print('PROFILER RESULTS — Per-module CUDA time')
    print('=' * 80)
    print(prof.key_averages(group_by_stack_n=5).table(sort_by = 'cuda_time_total', row_limit = 30))

    # Also save chrome trace for optional local viewing
    trace_path = f'{PROFILE_OUTPUT_DIR}/trace.json.gz'
    prof.export_chrome_trace(trace_path)
    print(f'\nChrome trace saved to {trace_path}')
    print('(scp to local machine and open at chrome://tracing)')

profiler_context = torch.profiler.profile(
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule = torch.profiler.schedule(
        wait = PROFILE_WAIT,
        warmup = PROFILE_WARMUP,
        active = PROFILE_ACTIVE,
        repeat = PROFILE_REPEAT,
    ),
    on_trace_ready = lambda p: print_profiler_summary(p),
    record_shapes = False,
    profile_memory = False,
    with_stack = False,
    with_modules = False,
) if PROFILE_ENABLED else nullcontext()

with profiler_context as prof:
    for i in (pbar := tqdm.tqdm(range(NUM_BATCHES), mininterval = 10., desc = 'training')):
        model.train()

        # Log oracle debug results from previous step's forward (if any)
        if MEMORY_TYPE == 'sparse_kda':
            from titans_pytorch.kda_memory import SparseKDAMemory
            oracle_log = {}
            for name, module in model.named_modules():
                if isinstance(module, SparseKDAMemory):
                    results = getattr(module, '_oracle_debug_results', None)
                    if results is not None:
                        short_name = name.replace('_orig_mod.', '')
                        oracle_log[f'oracle/{short_name}/router_accuracy'] = results['router_accuracy']
                        oracle_log[f'oracle/{short_name}/random_baseline']  = results['random_baseline']
                        for slot_idx, (dist, best) in enumerate(zip(results['slot_dist'], results['best_slot_dist'])):
                            oracle_log[f'oracle/{short_name}/slot_{slot_idx}_dist']      = dist
                            oracle_log[f'oracle/{short_name}/slot_{slot_idx}_best_frac'] = best
                        module._oracle_debug_results = None
            if oracle_log:
                wandb.log(oracle_log, step=i)

        total_recon_loss = None

        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            task_loss = model(next(train_loader).cuda(non_blocking = True), return_loss = True)
            loss = task_loss

            if MEMORY_TYPE == 'sparse_kda':
                from titans_pytorch.kda_memory import SparseKDAMemory
                for _, module in model.named_modules():
                    if isinstance(module, SparseKDAMemory):
                        if SPARSE_KDA_ROUTER_LOSS_TYPE == 'recon':
                            recon = module.get_recon_loss()
                            if recon is not None:
                                loss = loss + SPARSE_KDA_RECON_LOSS_WEIGHT * recon
                                total_recon_loss = recon if total_recon_loss is None else total_recon_loss + recon
                            module.reset_recon_loss()
                        else:  # 'bal'
                            bal = module.get_bal_loss()
                            if bal is not None:
                                loss = loss + SPARSE_KDA_BAL_LOSS_WEIGHT * bal
                                total_recon_loss = bal if total_recon_loss is None else total_recon_loss + bal
                            module.reset_bal_loss()
                        distill = module.get_distill_loss()
                        if distill is not None:
                            loss = loss + SPARSE_KDA_DISTILL_LOSS_WEIGHT * distill
                        module.reset_distill_loss()

            loss.backward()

        tqdm.tqdm.write(f'training loss: {task_loss.item():.4f}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        _rate = pbar.format_dict.get('rate') or 0
        log_dict = dict(
            train_loss = task_loss.item(),
            secs_per_iter = 1.0 / _rate if _rate > 0 else 0,
            qps = _rate * BATCH_SIZE * GRADIENT_ACCUMULATE_EVERY * SEQ_LEN,
        )
        if MEMORY_TYPE == 'sparse_kda' and total_recon_loss is not None:
            log_dict['bal_loss' if SPARSE_KDA_ROUTER_LOSS_TYPE == 'bal' else 'recon_loss'] = total_recon_loss.item()
        wandb.log(log_dict, step = i)

        if MEMORY_TYPE in ('sparse_kda', 'kda') and i % SPARSE_KDA_LOG_HITRATE_EVERY == 0:
            from titans_pytorch.kda_memory import SparseKDAMemory, KDAMemory
            log_dict = {}
            for name, module in model.named_modules():
                short_name = name.replace('_orig_mod.', '').replace('.4', '')
                if isinstance(module, SparseKDAMemory):
                    rates = module.get_slot_hit_rates()
                    avg_weights = module.get_slot_avg_weights()
                    logit_mean, logit_std = module.get_slot_logit_stats()
                    rates_str   = ' '.join(f'{r:.3f}'  for r in rates.tolist())
                    weights_str = ' '.join(f'{w:.4f}'  for w in avg_weights.tolist())
                    mean_str    = ' '.join(f'{m:.4f}'  for m in logit_mean.tolist())
                    std_str     = ' '.join(f'{s:.4f}'  for s in logit_std.tolist())
                    tqdm.tqdm.write(f'[{short_name}] hit_rate:    [{rates_str}]')
                    tqdm.tqdm.write(f'[{short_name}] avg_weight:  [{weights_str}]')
                    tqdm.tqdm.write(f'[{short_name}] logit_mean:  [{mean_str}]')
                    tqdm.tqdm.write(f'[{short_name}] logit_std:   [{std_str}]')
                    for slot_idx in range(len(rates.tolist())):
                        log_dict[f'slot_hit_rate/{short_name}/slot_{slot_idx}']   = rates[slot_idx].item()
                        log_dict[f'slot_avg_weight/{short_name}/slot_{slot_idx}'] = avg_weights[slot_idx].item()
                        log_dict[f'slot_logit_mean/{short_name}/slot_{slot_idx}'] = logit_mean[slot_idx].item()
                        log_dict[f'slot_logit_std/{short_name}/slot_{slot_idx}']  = logit_std[slot_idx].item()
                    module.reset_slot_stats()
                if isinstance(module, (SparseKDAMemory, KDAMemory)):
                    # residual_norm: mean |v - kS| from the last token per forward call.
                    # Lower = memory has better recall for the keys it sees. Reflects learning trend.
                    residual_norm = module.get_residual_norm()
                    if residual_norm is not None:
                        log_dict[f'residual_norm/{short_name}'] = residual_norm
                        tqdm.tqdm.write(f'[{short_name}] residual_norm: {residual_norm:.4f}')
                    module.reset_residual_norm()
                    # attn_residual: ||o_mem - o_attn|| at last token.
                    # o_mem = q@S (memory read-out), o_attn = causal softmax attention.
                    # Lower = memory better approximates ideal attention.
                    attn_residual = module.get_attn_residual()
                    if attn_residual is not None:
                        log_dict[f'attn_residual/{short_name}'] = attn_residual
                        tqdm.tqdm.write(f'[{short_name}] attn_residual: {attn_residual:.4f}')
                    module.reset_attn_residual()
            if log_dict:
                wandb.log(log_dict, step = i)

        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_loss = model(next(val_loader).cuda(non_blocking = True), return_loss = True)
                tqdm.tqdm.write(f'validation loss: {val_loss.item():.4f}')
                wandb.log(dict(val_loss = val_loss.item()), step = i)

        if MEMORY_TYPE == 'sparse_kda' and SPARSE_KDA_DISTILL_EVERY > 0 and i % SPARSE_KDA_DISTILL_EVERY == 0:
            set_sparse_kda_oracle_debug(model, i)

        if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:PRIME_LENGTH].cuda()
            prime = decode_tokens(inp)
            print(f'{prime} \n\n {"*" * 100}')

            sample = model.sample(inp[None, ...], GENERATE_LENGTH, use_cache = USE_FAST_INFERENCE)
            output_str = decode_tokens(sample[0])
            print(output_str)

        if PROFILE_ENABLED:
            prof.step()
