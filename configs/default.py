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
MEMORY_TYPE = 'kda'  # Options: 'neural', 'kda', 'sparse_kda'

NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 0
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
NEURAL_MEM_WEIGHT_RESIDUAL = True
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True
NEURAL_MEM_SPEC_NORM_SURPRISES = True

# KDA memory specific settings (only used when MEMORY_TYPE = 'kda')
KDA_CHUNK_SIZE = NEURAL_MEM_SEGMENT_LEN * 8     # Chunk size for KDA (larger chunks = more efficient)
KDA_USE_CHUNK = True                            # Use chunked KDA (faster) vs recurrent (more flexible)
KDA_HEADS = 8                                   # Number of attention heads (memory matrix size = heads * dim_head^2)
KDA_DIM_HEAD = 128                              # Head dimension (default: dim // heads = 64)

# Sparse KDA settings (only used when MEMORY_TYPE = 'sparse_kda')
SPARSE_KDA_NUM_SLOTS = 8                        # N: total number of memory matrices
SPARSE_KDA_TOP_K = 4                            # k: how many slots each token activates
SPARSE_KDA_LOG_HITRATE_EVERY = 5                # how often to log slot hit rates to wandb
SPARSE_KDA_ROUTER_LOSS_TYPE = 'recon'           # 'recon' = autoencoder reconstruction loss; 'bal' = Switch Transformer load balance loss
SPARSE_KDA_RECON_LOSS_WEIGHT = 0.5              # weight for router reconstruction loss (used when ROUTER_LOSS_TYPE='recon')
SPARSE_KDA_BAL_LOSS_WEIGHT = 0.01              # weight for Switch Transformer load balance loss (used when ROUTER_LOSS_TYPE='bal')
SPARSE_KDA_USE_SHARED_MEMORY = False            # add a dense shared memory that all tokens read/write
SPARSE_KDA_DISTILL_EVERY = 5                    # how often to run oracle debug + distillation loss (0 = disabled)
SPARSE_KDA_DISTILL_LOSS_WEIGHT = 0.00           # weight for oracle distillation loss (0 = disabled)
SPARSE_KDA_ROUTER_HIDDEN = None                 # None = linear router; int = MLP hidden width (e.g. key_dim*2)
SPARSE_KDA_ROUTER_USE_NORMALIZED_Q = False       # whether to feed per-head-normalized q to router (recommended)
SPARSE_KDA_USE_ORACLE_ROUTER = False             # debug: bypass learned router, select slots by mem-oracle dist
SPARSE_KDA_USE_LSH_WRITE = False                 # use LSH hash for write routing (True); False = write follows read top-1

# experiment related

PROJECT_NAME = 'sparse-kda-transformer'
WANDB_ONLINE = True

# perf related

USE_ACCELERATED_SCAN = False
USE_FLEX_ATTN = True
USE_FAST_INFERENCE = False
USE_AMP = True                                  # mixed precision (bf16) for tensor core utilization

# profiling related

PROFILE_ENABLED = False                         # set to True to enable profiling
PROFILE_OUTPUT_DIR = './profiler_logs'          # output directory for traces
PROFILE_WAIT = 40                               # steps to skip (let torch.compile finish warming up)
PROFILE_WARMUP = 2                              # steps to warm up profiler
PROFILE_ACTIVE = 3                              # steps to actively record
PROFILE_REPEAT = 1                              # number of profiling cycles (0 = repeat until end)
