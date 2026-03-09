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
import random
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

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 16
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = True
SEQ_LEN = 2048 # if SEQ_LEN is 512, KDA memory size has 8 x 128 entries which can cover all tokens already, even for 2048 seq len, the first 8 x 128 tokens won't benefit from KDA memory.

# neural memory related

# Choose memory type: 'neural' (TTT-based), 'kda' (linear attention), 'sparse_kda' (SM-KDA)
MEMORY_TYPE = 'sparse_kda'  # Options: 'neural', 'kda', 'sparse_kda'

NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 4 # TODO: we can set NUM_LONGTERM_MEM to 0, so only the memory matrix can retain longterm information, and see if that helps the neural memory model learn better longterm retention strategies. This also allows us to test the neural memory's ability to learn to write to the memory matrix in a way that retains longterm information, without relying on the presence of persist mem tokens which are designed to be longterm.
NEURAL_MEM_LAYERS = (2, 4, 6)                   # layers 2, 4, 6 have neural memory, can add more
NEURAL_MEM_GATE_ATTN_OUTPUT = False
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-1
USE_MEM_ATTENTION_MODEL = False
WINDOW_SIZE = 32
NEURAL_MEM_SEGMENT_LEN = 4                      # set smaller for more granularity for learning rate / momentum etc
NEURAL_MEM_BATCH_SIZE = 128                     # set smaller to update the neural memory weights more often as it traverses the sequence
SLIDING_WINDOWS = True
STORE_ATTN_POOL_CHUNKS = True                   # whether to use attention pooling for chunk derived momentum, per-layer lr mod, decay
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
NEURAL_MEM_WEIGHT_RESIDUAL = True               # learning to accept contributions from the weights of the previous neural mem layer brings about significant improvements. this was improvised and not in the paper, but inspired by the value residual learning free lunch paper
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True        # will allow the neural memory to select what layers from which to derive queries / keys / values, effectively allowing it to graft itself to the transformer in any way to be beneficial. this is to address an issue from a phd student who noted that the mem network is learning nothing more than wk @ wv. this also generalizes all possible ways to connect the neural memory to a transformer, a sort of NAS
NEURAL_MEM_SPEC_NORM_SURPRISES = True           # applying lessons from Muon optimizer to surprise updates, by spectral norming the surprises

# KDA memory specific settings (only used when MEMORY_TYPE = 'kda')
KDA_CHUNK_SIZE = NEURAL_MEM_SEGMENT_LEN * 8     # Chunk size for KDA (larger chunks = more efficient)
KDA_USE_CHUNK = True                            # Use chunked KDA (faster) vs recurrent (more flexible)
KDA_HEADS = 8                                   # Number of attention heads (memory matrix size = heads * dim_head^2)
KDA_DIM_HEAD = 128                               # Head dimension (default: dim // heads = 64)

# Sparse KDA settings (only used when MEMORY_TYPE = 'sparse_kda')
SPARSE_KDA_NUM_SLOTS = 8 # 8                        # N: total number of memory matrices
SPARSE_KDA_TOP_K = 4 # 4                            # k: how many slots each token activates
SPARSE_KDA_LOG_HITRATE_EVERY = 5                # how often to log slot hit rates to wandb
SPARSE_KDA_AUX_LOSS_WEIGHT = 0.01 # 0.01               # Switch Transformer load balance loss weight
SPARSE_KDA_USE_SHARED_MEMORY = False # True            # add a dense shared memory that all tokens read/write
SPARSE_KDA_DISTILL_EVERY = 5                  # how often to run oracle debug + distillation loss (0 = disabled)
SPARSE_KDA_DISTILL_LOSS_WEIGHT = 0.0          # weight for oracle distillation loss (0 = disabled)

# experiment related

PROJECT_NAME = 'titans-mac-transformer'
_sparse_kda_suffix = f' N={SPARSE_KDA_NUM_SLOTS} k={SPARSE_KDA_TOP_K} h={KDA_HEADS} d={KDA_DIM_HEAD}{"  +sh" if SPARSE_KDA_USE_SHARED_MEMORY else ""}{ f" aux={SPARSE_KDA_AUX_LOSS_WEIGHT}" if SPARSE_KDA_AUX_LOSS_WEIGHT > 0 else ""}{ f" dl={SPARSE_KDA_DISTILL_LOSS_WEIGHT}@{SPARSE_KDA_DISTILL_EVERY}" if SPARSE_KDA_DISTILL_LOSS_WEIGHT > 0 else ""}' if MEMORY_TYPE == 'sparse_kda' else ''
_kda_suffix = f' h={KDA_HEADS} d={KDA_DIM_HEAD}' if MEMORY_TYPE == 'kda' else ''
# run name abbreviations: N=num_slots, k=top_k, h=heads, d=dim_head, +sh=shared_memory
#   lm=num_longterm_mem, ly=neural_mem_layers, sq=seq_len, bs=batch_size, ga=gradient_accumulate_every
RUN_NAME = f'mac-{MEMORY_TYPE}{_sparse_kda_suffix}{_kda_suffix} lm={NUM_LONGTERM_MEM} ly={NEURAL_MEM_LAYERS} sq={SEQ_LEN} bs={BATCH_SIZE} ga={GRADIENT_ACCUMULATE_EVERY}'
WANDB_ONLINE = True # turn this on to pipe experiment to cloud

# perf related

USE_ACCELERATED_SCAN = False
USE_FLEX_ATTN = True
USE_FAST_INFERENCE = False
USE_AMP = True                                     # mixed precision (bf16) for tensor core utilization

# profiling related

PROFILE_ENABLED = False                            # set to True to enable profiling
PROFILE_OUTPUT_DIR = './profiler_logs'              # output directory for traces
PROFILE_WAIT = 40                                  # steps to skip (let torch.compile finish warming up)
PROFILE_WARMUP = 2                                 # steps to warm up profiler
PROFILE_ACTIVE = 3                                 # steps to actively record
PROFILE_REPEAT = 1                                 # number of profiling cycles (0 = repeat until end)

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
    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10., desc = 'training'):
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

        total_aux_loss = None

        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            task_loss = model(next(train_loader).cuda(non_blocking = True), return_loss = True)
            loss = task_loss

            if MEMORY_TYPE == 'sparse_kda':
                from titans_pytorch.kda_memory import SparseKDAMemory
                for _, module in model.named_modules():
                    if isinstance(module, SparseKDAMemory):
                        aux = module.get_aux_loss()
                        if aux is not None:
                            loss = loss + SPARSE_KDA_AUX_LOSS_WEIGHT * aux
                            total_aux_loss = aux if total_aux_loss is None else total_aux_loss + aux
                        module.reset_aux_loss()
                        distill = module.get_distill_loss()
                        if distill is not None:
                            loss = loss + SPARSE_KDA_DISTILL_LOSS_WEIGHT * distill
                        module.reset_distill_loss()

            loss.backward()

        tqdm.tqdm.write(f'training loss: {task_loss.item():.4f}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        log_dict = dict(train_loss = task_loss.item())
        if MEMORY_TYPE == 'sparse_kda' and total_aux_loss is not None:
            log_dict['aux_loss'] = total_aux_loss.item()
        wandb.log(log_dict, step = i)

        if MEMORY_TYPE == 'sparse_kda' and i % SPARSE_KDA_LOG_HITRATE_EVERY == 0:
            from titans_pytorch.kda_memory import SparseKDAMemory
            log_dict = {}
            for name, module in model.named_modules():
                if isinstance(module, SparseKDAMemory):
                    rates = module.get_slot_hit_rates()
                    avg_weights = module.get_slot_avg_weights()
                    logit_mean, logit_std = module.get_slot_logit_stats()
                    short_name = name.replace('_orig_mod.', '').replace('.4', '')
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
