#!/bin/bash

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <prefiller | decoder> [model]"
    exit 1
fi

if [[ $# -eq 1 ]]; then
    echo "Using default model: meta-llama/Meta-Llama-3.1-8B"
    MODEL="meta-llama/Meta-Llama-3.1-8B"
else
    echo "Using model: $2"
    MODEL=$2
fi

# The prefillers and decoders in LMCache use the same hash seed for all chunk keys.
# This seed must be aligned so that decoders can identify and retrieve KV cache
# entries stored by prefillers.
#
# WARNING: Using a fixed hash seed is insecure and makes the application vulnerable to
# denial-of-service attacks. In a production environment, this should be set to a
# secure random value. This is set to a fixed value for demonstration purposes only.
export PYTHONHASHSEED=${VLLM_PYTHON_HASH_SEED:-123}
export VLLM_LOGGING_LEVEL=DEBUG
# --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_record_shapes": 1, "torch_profiler_with_flops": 1, "torch_profiler_with_stack": 1, "torch_profiler_with_memory": 1}' \
export BLOCK_SIZE=64
export VLLM_TP=${VLLM_TP:-1}

if [[ $1 == "prefiller" ]]; then

    # python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \

    # 1st GPU as prefiller
    # OMP_NUM_THREADS=16 \
    # NEOReadDebugKeys=1 \
    # EnableSharedSystemUsmSupport=1 \
    VLLM_KV_CACHE_LAYOUT="NHD" \
    VLLM_OFFLOAD_KV_CACHE_TO_CPU=1 \
    $(which vllm) serve $MODEL \
    --port 8100 \
    --trust-remote-code \
    --max-model-len 20000 \
    --max-num-seqs 10 \
    --max-num-batched-tokens 20000 \
    --block-size $BLOCK_SIZE \
    --kv-transfer-config '{"kv_connector":"ShmConnector","kv_role":"kv_both"}' \
    --enforce-eager \
    -tp $VLLM_TP \
    --num-gpu-blocks-override $((2 * 20000 / BLOCK_SIZE)) \
    --offload-group-size 1 --offload-num-in-group 1 --offload-prefetch-step 2 \
    --enable-prefix-caching \
    # --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile_prefiller", "torch_profiler_record_shapes": 1, "torch_profiler_with_flops": 1, "torch_profiler_with_stack": 1, "torch_profiler_with_memory": 1}' \
    
elif [[ $1 == "decoder" ]]; then
    # Decoder listens on port 8200

    # 2nd GPU as decoder
    # OMP_NUM_THREADS=32 \
    # VLLM_CPU_OMP_THREADS_BIND="0-15|16-31" \
    NUMA_NODES=$(numactl -H | grep "^node [0-9]* cpus:" | awk '{print $2}' | tr '\n' ',' | sed 's/,$//') \
    VLLM_CPU_KVCACHE_SPACE=6 \
    VLLM_CPU_SGL_KERNEL="1" \
    numactl --cpunodebind=${NUMA_NODES} --membind=${NUMA_NODES} \
    $(which vllm) serve $MODEL \
    --port 8200 \
    --trust-remote-code \
    --max-model-len 30000 \
    --max-num-seqs 10 \
    --max-num-batched-tokens 1000 \
    --block-size $BLOCK_SIZE \
    --enable-prefix-caching \
    -tp $VLLM_TP \
    --kv-transfer-config '{"kv_connector":"ShmConnector","kv_role":"kv_both"}' \
    # --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile_decoder", "torch_profiler_record_shapes": 1, "torch_profiler_with_flops": 1, "torch_profiler_with_stack": 1, "torch_profiler_with_memory": 1}' \

else
    echo "Invalid role: $1"
    echo "Should be either prefiller, decoder"
    exit 1
fi
