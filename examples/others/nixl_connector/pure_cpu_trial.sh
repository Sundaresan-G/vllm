#!/bin/bash
#SBATCH --partition=emr
#SBATCH --constraint="ddr5600"
#SBATCH --job-name=vllm_emr
#SBATCH --output=slurm-emr-runs-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:59:00

set -x

hostname
lscpu
numactl -H | grep -E 'node [0-9]+ (size|free)'
for node in /sys/devices/system/node/node[0-9]*; do
    nid=$(basename $node | tr -d 'node')
    awk -v n=$nid '/MemTotal/{t=$4} /MemFree/{f=$4} /FilePages/{c=$4} /Slab/{s=$4} END{printf "node%s: total=%dGB free=%dGB actual_used=%dGB page_cache=%dGB\n", n, t/1024/1024, f/1024/1024, (t-f-c-s)/1024/1024, (c+s)/1024/1024}' $node/meminfo
done

PIDS=()
# MODEL="sarvamai/sarvam-30b"
MODEL="Qwen/Qwen3-30B-A3B"
# MODEL="Qwen/Qwen2.5-1.5B"
# MODEL="Qwen/Qwen3-235B-A22B"
INPUT_LEN=16384
OUTPUT_LEN=32
NUM_PROMPTS=8

CPU_ENV="vllm_0.23.0_cpu"

CONDA_BASE="/data/nfs_home/sundares/miniforge3"
SCRIPT_DIR="/data/nfs_home/sundares/vllm/vllm/examples/others/nixl_connector"

export VLLM_LOGGING_LEVEL=DEBUG

# Logging helper — prints [file:line func()] message to stderr
log() {
    local file="${BASH_SOURCE[1]##*/}"
    local line="${BASH_LINENO[0]}"
    local func="${FUNCNAME[1]:-main}"
    echo "[${file}:${line} ${func}()] $*" >&2
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
  local port=$1
  local pid=$2
  local timeout_seconds=1200
  local start_time=$(date +%s)

  log "Waiting for server on port $port..."

  while true; do
    
    numactl -H | grep -E 'node [0-9]+ (size|free)'
    for node in /sys/devices/system/node/node[0-9]*; do
        nid=$(basename $node | tr -d 'node')
        awk -v n=$nid '/MemTotal/{t=$4} /MemFree/{f=$4} /FilePages/{c=$4} /Slab/{s=$4} END{printf "node%s: total=%dGB free=%dGB actual_used=%dGB page_cache=%dGB\n", n, t/1024/1024, f/1024/1024, (t-f-c-s)/1024/1024, (c+s)/1024/1024}' $node/meminfo
    done
    
    # Check if the process is still running before attempting curl
    if ! kill -0 "$pid" 2>/dev/null; then
        log "Process with PID $pid and port $port has terminated unexpectedly."
        cleanup unexpected
    fi

    if curl -s "localhost:${port}/health"; then
      return 0
    elif curl -s -X POST "localhost:${port}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"model": "test", "prompt": "test", "max_tokens": 1}' > /dev/null 2>&1; then
      return 0
    fi

    local now=$(date +%s)
    log "Waiting for server on port $port..."
    if (( now - start_time >= timeout_seconds )); then
      log "ERROR: Timeout waiting for server on port $port"
      cleanup timeout
    fi

    sleep 1
  done
}

main() {

    trap cleanup INT
    trap cleanup TERM

    cd $SCRIPT_DIR
    source $CONDA_BASE/etc/profile.d/conda.sh
    conda activate $CPU_ENV

    # python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    # --offload-group-size 1 --offload-num-in-group 1 --offload-prefetch-step 2 \
    # --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_record_shapes": 1, "torch_profiler_with_flops": 1, "torch_profiler_with_stack": 1, "torch_profiler_with_memory": 1}'

    # If VLLM_OFFLOAD_KV_CACHE_TO_CPU=1, then KV_BUFFER_DEVICE does not matter and it will be ignored.
    (
        VLLM_CPU_KVCACHE_SPACE=10 \
        VLLM_CPU_SGL_KERNEL="1" \
        VLLM_CPU_OMP_THREADS_BIND="0-63|64-127" \
        VLLM_KV_CACHE_LAYOUT="NHD" \
        VLLM_XPU_ENABLE_XPU_GRAPH=0 \
        $(which vllm) serve $MODEL --trust-remote-code --port 9000 --max-model-len 20000 --max-num-seqs 1  --max-num-batched-tokens 20000 -tp 2 --no-enable-prefix-caching --block-size 64
    ) &

    server_pid=$!
    PIDS+=($server_pid)

    wait_for_server 9000 $server_pid

    log "=================================================="
    log "All servers are up. You can send request now..."
    log "Press Ctrl-C to terminate all instances."

    # Keep the script running until interrupted
    log "Script is running. Waiting for termination signal..."
    log "=================================================="

    # echo "All servers are up. Starting benchmark..."

    # begin benchmark
    # cd ../../../benchmarks/

    # python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    $(which vllm) bench serve --port 9000 --seed $(date +%s) \
        --model $MODEL \
        --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
        --num-prompts $NUM_PROMPTS --max-concurrency 1 \
        2>&1 | tee benchmark_cpu.log

    # curl -X POST http://localhost:9000/v1/completions -H "Content-Type: application/json" -d '{    "model": "'"$MODEL"'",    "prompt": "Write a detailed, vivid, and slightly humorous free-verse poem about the craft of software engineering and coding. Touch on long nights spent debugging, collaborating with teammates, wrestling with legacy code, and the relief when all the tests finally pass. Use clear imagery, a hopeful tone.", "max_tokens": 10,    "temperature": 0.7  }' |& tee -a benchmark.log

    # curl -X POST http://localhost:9000/v1/completions -H "Content-Type: application/json" -d '{    "model": "'"$MODEL"'",    "prompt": "Write a rich, vivid, slightly humorous free-verse poem about the craft of software engineering and coding. Describe in detail long nights spent debugging elusive bugs, the glow of multiple monitors, half-finished mugs of cold coffee, and the quiet hum of machines in an almost empty office or home workspace. Show the emotional roller coaster of reading confusing legacy code, adding one more log line, watching stack traces scroll by, and wondering what the previous developer was thinking when they designed this system. Include scenes of collaboration: pair programming sessions, code review comments that are both kind and blunt, whiteboard diagrams that start neat and end as chaotic scribbles, and chat messages full of links to docs, tickets, and pull requests. Mention modern tools and rituals of the craft: version control, feature branches, continuous integration pipelines, flaky tests, deployment scripts, and dashboards that flip from red to green. Contrast the stress of production incidents, paging alerts, and frantic hotfixes with the quiet, satisfying moment when all tests finally pass, the pipeline is green, and the release is tagged.", "max_tokens": 100,    "temperature": 0.7  }' |& tee -a benchmark.log

    # curl -X POST http://localhost:9000/v1/completions -H "Content-Type: application/json" -d '{    "model": "'"$MODEL"'",    "prompt": "Write a rich, vivid, slightly humorous free-verse poem about the craft of software engineering and coding. Describe in detail long nights spent debugging elusive bugs, the glow of multiple monitors, half-finished mugs of cold coffee, and the quiet hum of machines in an almost empty office or home workspace. Show the emotional roller coaster of reading confusing legacy code, adding one more log line, watching stack traces scroll by, and wondering what the previous developer was thinking when they designed this system. Include scenes of collaboration: pair programming sessions, code review comments that are both kind and blunt, whiteboard diagrams that start neat and end as chaotic scribbles, and chat messages full of links to docs, tickets, and pull requests. Mention modern tools and rituals of the craft: version control, feature branches, continuous integration pipelines, flaky tests, deployment scripts, and dashboards that flip from red to green. Contrast the stress of production incidents, paging alerts, and frantic hotfixes with the quiet, satisfying moment when all tests finally pass, the pipeline is green, and the release is tagged. Use concrete imagery that developers recognize, add gentle inside jokes about off by one errors and mysterious race conditions, and keep the overall tone hopeful and affirming. Celebrate the creativity, persistence, and teamwork that make software possible, and end on a note of cautious but genuine optimism about the next refactor, the next big feature, and the next late night that somehow feels worth it.", "max_tokens": 100,    "temperature": 0.7  }' |& tee -a benchmark.log

    # To check the prefix caching effect
    # curl -X POST http://localhost:9000/v1/completions -H "Content-Type: application/json" -d '{    "model": "'"$MODEL"'",    "prompt": "Write a rich, vivid, slightly humorous free-verse poem about the craft of software engineering and coding. Describe in detail long nights spent debugging elusive bugs, the glow of multiple monitors, half-finished mugs of cold coffee, and the quiet hum of machines in an almost empty office or home workspace. Show the emotional roller coaster of reading confusing legacy code, adding one more log line, watching stack traces scroll by, and wondering what the previous developer was thinking when they designed this system. Include scenes of collaboration: pair programming sessions, code review comments that are both kind and blunt, whiteboard diagrams that start neat and end as chaotic scribbles, and chat messages full of links to docs, tickets, and pull requests. Mention modern tools and rituals of the craft: version control, feature branches, continuous integration pipelines, flaky tests, deployment scripts, and dashboards that flip from red to green. Contrast the stress of production incidents, paging alerts, and frantic hotfixes with the quiet, satisfying moment when all tests finally pass, the pipeline is green, and the release is tagged. Use concrete imagery that developers recognize, add gentle inside jokes about off by one errors and mysterious race conditions, and keep the overall tone hopeful and affirming. Celebrate the creativity, persistence, and teamwork that make software possible, and end on a note of cautious but genuine optimism about the next refactor, the next big feature, and the next late night that somehow feels worth it.", "max_tokens": 100,    "temperature": 0.7  }' |& tee -a benchmark.log

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

    # vllm bench serve --port 9000 --seed $(date +%s) \
    #     --model $MODEL \
    #     --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    #     --num-prompts $NUM_PROMPTS \
    #     2>&1 | tee benchmark.log

    # while true; do
    #     sleep 1
    # done

    # Accuracy / correctness spot-check
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
    2>&1 | tee accuracy_test_cpu.log

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
    2>&1 | tee -a accuracy_test_cpu.log

    cleanup success

}

main
