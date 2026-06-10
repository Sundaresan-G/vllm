#!/bin/bash
##SBATCH --partition=b70
##SBATCH -w pcl-zen4
##SBATCH --partition=bmtxg31
##SBATCH --partition=rtx5070
##SBATCH --partition=h100
##SBATCH --cpus-per-task=60
#SBATCH --partition=b70
#SBATCH --job-name=vllm_b70
#SBATCH --output=slurm-b70-runs-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:59:00

# echo "Warning: LMCache disaggregated prefill support for vLLM v1 is experimental and subject to change."

PIDS=()

# Logging helper — prints [file:line func()] message to stderr
log() {
    local file="${BASH_SOURCE[1]##*/}"
    local line="${BASH_LINENO[0]}"
    local func="${FUNCNAME[1]:-main}"
    echo "[${file}:${line} ${func}()] $*" >&2
}

# MODEL="Qwen/Qwen2.5-1.5B-Instruct"
# MODEL="Qwen/Qwen2.5-7B"
MODEL="Qwen/Qwen3-30B-A3B"
# MODEL="sarvamai/sarvam-30b"
INPUT_LEN=16384
OUTPUT_LEN=32
NUM_PROMPTS=8

GPU_ENV="vllm_0.19.1_xpu"
CPU_ENV="vllm_0.19.1_cpu"
CONDA_BASE="/data/nfs_home/sundares/miniforge3"
SHM_CONNECTOR_DIR="/data/nfs_home/sundares/vllm/vllm/examples/others/shm_connector"

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

# Kill a PID and all its descendants at any depth (BFS collect, reverse kill)
kill_tree() {
    local root=$1
    local signal=${2:-TERM}
    [[ -z "$root" ]] && return 0
    local -a all=("$root")
    local i=0
    while (( i < ${#all[@]} )); do
        local children
        children=$(pgrep -P "${all[$i]}" 2>/dev/null) || true
        for c in $children; do all+=("$c"); done
        (( i++ ))
    done
    # Kill in reverse so leaves die before their parents
    for (( j=${#all[@]}-1; j>=0; j-- )); do
        kill -"$signal" "${all[$j]}" 2>/dev/null || true
    done
}

stop_server() {
    log "Stopping server…"
    # Graceful SIGTERM to each tracked PID tree
    for pid in "${PIDS[@]}"; do
        kill_tree "$pid" TERM
    done
    sleep 5
    # Force SIGKILL any survivors
    for pid in "${PIDS[@]}"; do
        kill_tree "$pid" KILL
    done
    wait -- "${PIDS[@]}" 2>/dev/null || true
    PIDS=()
    log "Cleaning up shared memory..."
    find /dev/shm -maxdepth 1 -user $USER -type f -exec rm -f {} +
}

cleanup() {
    local reason=${1:-interrupt}
    log "Stopping everything… (reason: ${reason})"
    trap '' INT TERM  # ignore signals during cleanup to prevent re-entrancy
    stop_server
    exit 0
}

wait_for_server() {
  # Usage: wait_for_server PORT1 PID1 [PORT2 PID2 ...]
  local -a ports=() pids=()
  while (( $# >= 2 )); do
    ports+=("$1")
    pids+=("$2")
    shift 2
  done

  local timeout_seconds=1200
  local start_time=$(date +%s)
  local -a done_flags
  for (( k=0; k<${#ports[@]}; k++ )); do done_flags+=(0); done

  while true; do
    local all_done=1
    for (( k=0; k<${#ports[@]}; k++ )); do
      [[ "${done_flags[$k]}" == "1" ]] && continue
      local port=${ports[$k]}
      local pid=${pids[$k]}
      if ! kill -0 "$pid" 2>/dev/null; then
        log "Process with PID $pid and port ${port} has terminated unexpectedly."
        cleanup "server on port ${port} died"
      fi
      if curl -s "localhost:${port}/health" > /dev/null 2>&1 || \
         curl -s -X POST "localhost:${port}/v1/completions" \
             -H "Content-Type: application/json" \
             -d '{"model": "test", "prompt": "test", "max_tokens": 1}' > /dev/null 2>&1; then
        log "Server on port ${port} is up."
        done_flags[$k]=1
      else
        all_done=0
        log "Waiting for server on port ${port}..."
      fi
    done
    [[ "$all_done" == "1" ]] && return 0

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      log "ERROR: Timeout waiting for servers on ports: ${ports[*]}"
      cleanup timeout
    fi

    sleep 1
  done
}


main() {

    set -x

    # check_hf_token
    # check_num_gpus
    # ensure_python_library_installed lmcache
    # ensure_python_library_installed nixl
    # ensure_python_library_installed pandas
    # ensure_python_library_installed datasets
    # ensure_python_library_installed vllm

    trap cleanup INT
    trap cleanup TERM

    log "Launching prefiller, decoder and proxy..."
    log "Please check prefiller.log, decoder.log and proxy.log for logs."

    (
        exec > >(tee decoder.log) 2>&1
        source $CONDA_BASE/etc/profile.d/conda.sh
        conda activate $CPU_ENV
        VLLM_TP=1 \
        VLLM_LOGGING_PREFIX="DECODER " \
        bash prefiller_decoder_vllm_launcher.sh decoder $MODEL
    ) &
    decoder_pid=$!
    PIDS+=($decoder_pid)

    (
        exec > >(tee prefiller.log) 2>&1
        source $CONDA_BASE/etc/profile.d/conda.sh
        conda activate $GPU_ENV
        # If VLLM_OFFLOAD_KV_CACHE_TO_CPU=1, then KV_BUFFER_DEVICE does not matter and it will be ignored.
        # ONEAPI_DEVICE_SELECTOR="level_zero:0,4;opencl:0,4" \
        VLLM_TP=1 \
        VLLM_LOGGING_PREFIX="PREFILLER " \
        bash prefiller_decoder_vllm_launcher.sh prefiller $MODEL
    ) &
    prefiller_pid=$!
    PIDS+=($prefiller_pid)

    (
        exec > >(tee proxy.log) 2>&1
        source $CONDA_BASE/etc/profile.d/conda.sh
        conda activate $GPU_ENV
        # Use proxy_server.py or toy_proxy_server.py
        # python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
        python "$SHM_CONNECTOR_DIR/toy_proxy_server.py" \
            --host 0.0.0.0 \
            --port 9000 \
            --prefiller-host localhost \
            --prefiller-port 8100 \
            --decoder-host localhost \
            --decoder-port 8200
    ) &
    proxy_pid=$!
    PIDS+=($proxy_pid)

    wait_for_server 8100 $prefiller_pid  8200 $decoder_pid  9000 $proxy_pid

    log "=================================================="
    log "All servers are up. You can send request now..."
    log "Press Ctrl-C to terminate all instances."

    # Keep the script running until interrupted
    log "Script is running. Waiting for termination signal..."
    log "=========================================="

    # begin benchmark
    # python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    # $(which vllm) bench serve --port 9000 --seed $(date +%s) \
    #     --model $MODEL \
    #     --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    #     --num-prompts $NUM_PROMPTS --max-concurrency 1 \
    #     2>&1 | tee benchmark.log

    # curl -X POST http://localhost:9000/v1/completions -H "Content-Type: application/json" -d '{    "model": "'"$MODEL"'",    "prompt": "Write a detailed, vivid, and slightly humorous free-verse poem about the craft of software engineering and coding. Touch on long nights spent debugging, collaborating with teammates, wrestling with legacy code, and the relief when all the tests finally pass. Use clear imagery, a hopeful tone.", "max_tokens": 10,    "temperature": 0.7  }' |& tee -a benchmark.log

    # curl -X POST http://localhost:9000/v1/completions -H "Content-Type: application/json" -d '{    "model": "'"$MODEL"'",    "prompt": "Write a rich, vivid, slightly humorous free-verse poem about the craft of software engineering and coding. Describe in detail long nights spent debugging elusive bugs, the glow of multiple monitors, half-finished mugs of cold coffee, and the quiet hum of machines in an almost empty office or home workspace. Show the emotional roller coaster of reading confusing legacy code, adding one more log line, watching stack traces scroll by, and wondering what the previous developer was thinking when they designed this system. Include scenes of collaboration: pair programming sessions, code review comments that are both kind and blunt, whiteboard diagrams that start neat and end as chaotic scribbles, and chat messages full of links to docs, tickets, and pull requests. Mention modern tools and rituals of the craft: version control, feature branches, continuous integration pipelines, flaky tests, deployment scripts, and dashboards that flip from red to green. Contrast the stress of production incidents, paging alerts, and frantic hotfixes with the quiet, satisfying moment when all tests finally pass, the pipeline is green, and the release is tagged.", "max_tokens": 100,    "temperature": 0.7  }' |& tee -a benchmark.log

    # To check the prefix caching effect
    # curl -X POST http://localhost:9000/v1/completions -H "Content-Type: application/json" -d '{    "model": "'"$MODEL"'",    "prompt": "Write a rich, vivid, slightly humorous free-verse poem about the craft of software engineering and coding.", "max_tokens": 100,    "temperature": 0.7  }' |& tee -a benchmark.log

    (
        source $CONDA_BASE/etc/profile.d/conda.sh
        conda activate $GPU_ENV
        $(which vllm) bench serve --port 9000 --seed $(date +%s) \
            --model $MODEL \
            --dataset-name prefix_repetition \
            --prefix-repetition-prefix-len $((INPUT_LEN/2)) \
            --prefix-repetition-suffix-len $((INPUT_LEN/2)) \
            --prefix-repetition-num-prefixes 2 \
            --prefix-repetition-output-len $OUTPUT_LEN \
            --num-prompts $NUM_PROMPTS \
            --max-concurrency 1 \
            2>&1 | tee benchmark_prefix_caching.log

        curl --fail-with-body -X POST http://localhost:9000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL"'",
            "prompt": [
            "You are a helpful AI assistant. The following context describes a large distributed inference system. The system uses paged KV caching, continuous batching, and tensor parallelism to serve large language models efficiently. Requests are scheduled by a central scheduler that tracks per-request KV cache block allocations. The KV cache is divided into fixed-size blocks, and a block table maps logical blocks to physical GPU memory. Prefix caching reuses KV blocks for identical prompt prefixes across requests, avoiding redundant computation. Now answer the following question: What are the main benefits of prefix caching in LLM serving?"
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }' \
        2>&1 | tee "accuracy_test.log"

        echo -e "\n\n" | tee -a "accuracy_test.log"

        curl --fail-with-body -X POST http://localhost:9000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL"'",
            "prompt": [
            "You are a helpful AI assistant. The following context describes a large distributed inference system. The system uses paged KV caching, continuous batching, and tensor parallelism to serve large language models efficiently. Requests are scheduled by a central scheduler that tracks per-request KV cache block allocations. The KV cache is divided into fixed-size blocks, and a block table maps logical blocks to physical GPU memory. Prefix caching reuses KV blocks for identical prompt prefixes across requests, avoiding redundant computation. Now answer the following question: What are the main benefits of prefix caching in LLM serving?",
            "You are a helpful AI assistant. The following context describes a large distributed inference system. The system uses paged KV caching, continuous batching, and tensor parallelism to serve large language models efficiently. Requests are scheduled by a central scheduler that tracks per-request KV cache block allocations. The KV cache is divided into fixed-size blocks, and a block table maps logical blocks to physical GPU memory. Prefix caching reuses KV blocks for identical prompt prefixes across requests, avoiding redundant computation. Now answer the following question: How does block-level prefix caching differ from token-level caching?",
            "You are a helpful AI assistant. The following context describes a large distributed inference system. The system uses paged KV caching, continuous batching, and tensor parallelism to serve large language models efficiently. Requests are scheduled by a central scheduler that tracks per-request KV cache block allocations. The KV cache is divided into fixed-size blocks, and a block table maps logical blocks to physical GPU memory. Prefix caching reuses KV blocks for identical prompt prefixes across requests, avoiding redundant computation. Now answer the following question: What workloads benefit most from prefix caching and why?"
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }' \
        2>&1 | tee -a "accuracy_test.log"
    )

    # while true; do
    #     sleep 1
    # done

    cleanup

}

main
