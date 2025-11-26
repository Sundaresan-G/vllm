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
export TORCH_CUDA_ARCH_LIST="12.0"
export UCX_NET_DEVICES=all
export UCX_TLS=all
# export VLLM_KV_CACHE_LAYOUT="NHD" # Do not use this. VLLM accuracy goes for a toss when using this.
export VLLM_TORCH_PROFILER_WITH_STACK=1 
export VLLM_TORCH_PROFILER_WITH_FLOPS=1
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_DIR="."
export BLOCK_SIZE=32

if [[ $1 == "prefiller" ]]; then

    # python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \

    # 1st GPU as prefiller
    # OMP_NUM_THREADS=16 \
    VLLM_OFFLOAD_KV_CACHE_TO_CPU=1 \
    VLLM_CPU_OMP_THREADS_BIND="0-31" \
    VLLM_CPU_SGL_KERNEL="1" \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    VLLM_DOUBLE_BUFFER_PIPELINE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    numactl -C 0-31 \
    $(which vllm) serve $MODEL \
    --port 8100 \
    --max-model-len 2600 \
    --max-num-seqs 512 \
    --max-num-batched-tokens 4096 \
    --block-size $BLOCK_SIZE \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"'"$KV_BUFFER_DEVICE"'"}' \
    --enforce-eager \
    --no-enable-prefix-caching # Ensure prefiller does not use prefix caching when VLLM_OFFLOAD_KV_CACHE_TO_CPU=1
    
elif [[ $1 == "decoder" ]]; then
    # Decoder listens on port 8200

    source /swtools/cuda/12.9.0/cuda_vars.sh

    # 2nd GPU as decoder
    # OMP_NUM_THREADS=32 \
    VLLM_CPU_OMP_THREADS_BIND="32-63" \
    VLLM_CPU_SGL_KERNEL="1" \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
    VLLM_DOUBLE_BUFFER_PIPELINE=0 \
    CUDA_VISIBLE_DEVICES=1 \
    numactl -C 32-63 \
    $(which vllm) serve $MODEL \
    --port 8200 \
    --enforce-eager \
    --max-model-len 2600 \
    --max-num-seqs 512 \
    --max-num-batched-tokens 4096 \
    --block-size $BLOCK_SIZE \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"'"$KV_BUFFER_DEVICE"'"}'

else
    echo "Invalid role: $1"
    echo "Should be either prefiller, decoder"
    exit 1
fi
