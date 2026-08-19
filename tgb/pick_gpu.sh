#!/bin/bash
# Print the index of a GPU with at least $1 MiB free (default 22000).
# Judged by FREE memory, not by "used < 3000": on an 80 GB card a neighbour
# using 31 GB still leaves 50 GB, and a fixed used-threshold rejects it. An
# empty result here silently becomes CUDA_VISIBLE_DEVICES= , which vLLM
# reports as "No CUDA GPUs are available", so callers must check for empty.
#
# With $2 > 1, print that many comma-separated indices, or nothing if there are
# not that many. 14B needs this: its 28 GB of weights equal the free space on a
# borrowed card, so single-card leaves zero for the KV cache no matter what
# gpu_memory_utilization says, and only tensor parallelism fits it.
need=${1:-22000}
want=${2:-1}
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits |
  awk -F', ' -v n="$need" -v w="$want" '
    $2 - $3 >= n { g[++k] = $1; if (k == w) { s = g[1]
                     for (i = 2; i <= w; i++) s = s "," g[i]; print s; exit } }'
