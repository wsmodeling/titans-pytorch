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
BATCH_SIZE = 32
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = True
SEQ_LEN = 512

# neural memory related

# Choose memory type: 'neural' (TTT-based) or 'kda' (linear attention)
MEMORY_TYPE = 'neural'  # Options: 'neural', 'kda'

NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 4
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

# experiment related

PROJECT_NAME = 'titans-mac-transformer'
RUN_NAME = f'mac-{MEMORY_TYPE} - {NUM_LONGTERM_MEM} longterm mems, layers {NEURAL_MEM_LAYERS}'
WANDB_ONLINE = False # turn this on to pipe experiment to cloud

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
wandb.init(project = PROJECT_NAME, mode = 'disabled' if not WANDB_ONLINE else 'online')
wandb.run.name = RUN_NAME
wandb.run.save()

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# memory model

if MEMORY_TYPE == 'kda':
    from titans_pytorch import create_kda_memory_for_mac

    print(f"Using KDA Memory (Kimi Delta Attention)")
    print(f"  - Chunk size: {KDA_CHUNK_SIZE}")
    print(f"  - Use chunk mode: {KDA_USE_CHUNK}")

    # KDAMemory is used directly, not wrapped in NeuralMemory
    # dim=64 is a template; MAC Transformer recreates with transformer dim
    neural_memory_model = create_kda_memory_for_mac(
        dim = 64,
        chunk_size = KDA_CHUNK_SIZE,
        use_chunk = KDA_USE_CHUNK,
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

        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_loader).cuda(non_blocking = True), return_loss = True)
            loss.backward()

        print(f'training loss: {loss.item():.4f}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()
        wandb.log(dict(loss = loss.item()))

        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                loss = model(next(val_loader).cuda(non_blocking = True), return_loss = True)
                print(f'validation loss: {loss.item():.4f}')

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

