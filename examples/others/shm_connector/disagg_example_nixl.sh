#!/bin/bash

# echo "Warning: LMCache disaggregated prefill support for vLLM v1 is experimental and subject to change."

set -xe

PIDS=()
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
INPUT_LEN=1024
OUTPUT_LEN=8
NUM_PROMPTS=5

# Switch to the directory of the current script
cd "$(dirname "${BASH_SOURCE[0]}")"

check_hf_token() {
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN is not set. Please set it to your Hugging Face token."
        exit 1
    fi
    if [[ "$HF_TOKEN" != hf_* ]]; then
        echo "HF_TOKEN is not a valid Hugging Face token. Please set it to your Hugging Face token."
        exit 1
    fi
    echo "HF_TOKEN is set and valid."
}

check_num_gpus() {
    # can you check if the number of GPUs are >=2 via nvidia-smi/rocm-smi?
    which rocm-smi > /dev/null 2>&1
    if [ $? -ne 0 ]; then
	num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
	num_gpus=$(rocm-smi --showid | grep Instinct | wc -l)
    fi

    if [ "$num_gpus" -lt 2 ]; then
        echo "You need at least 2 GPUs to run disaggregated prefill."
        exit 1
    else
        echo "Found $num_gpus GPUs."
    fi
}

ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    python3 -c "import $1" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        if [ "$1" == "nixl" ]; then
            echo "$1 is not installed. Please refer to https://github.com/ai-dynamo/nixl for installation."
        else
            echo "$1 is not installed. Please install it via pip install $1."
        fi
        exit 1
    else
        echo "$1 is installed."
    fi
}

cleanup() {
    echo "Stopping everything…"
    trap - INT TERM        # prevent re-entrancy
    kill -- $PIDS          
    wait                   # reap children so we don't leave zombies
    rm -f /dev/shm/*
    # kill -9 -- -$$         # in case any processes are still lingering
    exit 0
}

wait_for_server() {
  local port=$1
  local timeout_seconds=1200
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "localhost:${port}/healthcheck" > /dev/null; then
      return 0
    # Fallback to POST request to completions
    elif curl -s -X POST "localhost:${port}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"model": "test", "prompt": "test", "max_tokens": 1}' > /dev/null 2>&1; then
      return 0
    fi

    local now=$(date +%s)
    echo "Waiting for server on port $port..."
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server"
      return 1
    fi

    sleep 1
  done
}


main() {

    source /data/nfs_home/sundares/miniforge3/etc/profile.d/conda.sh
    # conda activate vllm_0.13.0_cpu_nonAvx
    # conda activate vllm_0.13.0_shm_xpu
    conda activate vllm_0.13.0_shm_cuda

    # check_hf_token
    # check_num_gpus
    # ensure_python_library_installed lmcache
    # ensure_python_library_installed nixl
    # ensure_python_library_installed pandas
    # ensure_python_library_installed datasets
    # ensure_python_library_installed vllm

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    echo "Launching prefiller, decoder and proxy..."
    echo "Please check prefiller.log, decoder.log and proxy.log for logs."

    # If VLLM_OFFLOAD_KV_CACHE_TO_CPU=1, then KV_BUFFER_DEVICE does not matter and it will be ignored.
    VLLM_LOGGING_PREFIX="PREFILLER " \
    bash prefiller_decoder_vllm_launcher.sh prefiller $MODEL \
        > >(tee prefiller.log) 2>&1 &
    prefiller_pid=$!
    PIDS+=($prefiller_pid)

    # conda activate vllm_0.13.0_cpu_nonAvx
    conda activate vllm_0.13.0_cpu

    VLLM_LOGGING_PREFIX="DECODER " \
    bash prefiller_decoder_vllm_launcher.sh decoder $MODEL \
        > >(tee decoder.log)  2>&1 &
    decoder_pid=$!
    PIDS+=($decoder_pid)

    # conda activate vllm_0.13.0_shm_xpu
    conda activate vllm_0.13.0_shm_cuda

    # Use proxy_server.py or toy_proxy_server.py
    # python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    python toy_proxy_server.py \
        --host localhost \
        --port 9000 \
        --prefiller-host localhost \
        --prefiller-port 8100 \
        --decoder-host localhost \
        --decoder-port 8200  \
        > >(tee proxy.log)    2>&1 &

    proxy_pid=$!
    PIDS+=($proxy_pid)

    wait_for_server 8100
    wait_for_server 8200
    wait_for_server 9000

    # echo "All servers are up. Starting benchmark..."

    # begin benchmark
    # cd ../../../benchmarks/

    # python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    $(which vllm) bench serve --port 9000 --seed $(date +%s) \
        --model $MODEL \
        --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
        --num-prompts $NUM_PROMPTS --max-concurrency 1 \
        2>&1 | tee benchmark.log
    # curl -X POST http://localhost:9000/v1/completions -H "Content-Type: application/json" -d '{    "model": "Qwen/Qwen2.5-1.5B-Instruct",    "prompt": "Write a detailed, vivid, and slightly humorous free-verse poem about the craft of software engineering and coding. Touch on long nights spent debugging, collaborating with teammates, wrestling with legacy code, and the relief when all the tests finally pass. Use clear imagery, a hopeful tone.", "max_tokens": 10,    "temperature": 0.7  }' |& tee benchmark.log

    # while true; do
    #     # python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    #     $(which vllm) bench serve --port 9000 --seed $(date +%s) \
    #         --model $MODEL \
    #         --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    #         --num-prompts $NUM_PROMPTS \
    #         2>&1 | tee benchmark.log
    # done

    # vllm bench serve --port 9000 --seed $(date +%s) \
    #     --model $MODEL \
    #     --dataset-name random --random-input-len 20 --random-output-len 10 \
    #     --num-prompts 2 --burstiness 1 --request-rate 3.6 | tee benchmark.log

    # vllm bench latency --port 9000 \
    #     --model $MODEL \

    # echo "Benchmarking done. Cleaning up..."

    # cleanup

    echo "==================================================="
    echo "All servers are up. You can send request now..."
    echo "Press Ctrl-C to terminate all instances."

    # Keep the script running until interrupted
    echo "Script is running. Waiting for termination signal..."
    echo "==================================================="

    # vllm bench serve --port 9000 --seed $(date +%s) \
    #     --model $MODEL \
    #     --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    #     --num-prompts $NUM_PROMPTS \
    #     2>&1 | tee benchmark.log

    while true; do
        sleep 1
    done

    cleanup

}

main
