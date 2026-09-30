#!/bin/bash
#SBATCH --partition=bmtxb70
##SBATCH --partition=rtx5070
#SBATCH --job-name=vllm_b70
#SBATCH --output=slurm-b70-runs-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:59:00

set -x

hostname
lscpu
numactl -H | grep -E 'node [0-9]+ (size|free)'
for node in /sys/devices/system/node/node[0-9]*; do
    nid=$(basename $node | tr -d 'node')
    awk -v n=$nid '/MemTotal/{t=$4} /MemFree/{f=$4} /FilePages/{c=$4} /Slab/{s=$4} END{printf "node%s: total=%.2fGB free=%.2fGB actual_used=%.2fGB page_cache=%.2fGB\n", n, t/1024/1024, f/1024/1024, (t-f-c-s)/1024/1024, (c+s)/1024/1024}' $node/meminfo
done
sudo nvtop -s | grep mem_util | awk '{print "GPU " NR-1 ":", $0}'

# sudo /etc/pcl_cleanup_memory.sh
# numactl -H | grep -E 'node [0-9]+ (size|free)'
# ps aux --sort=-%mem | head -6

PIDS=()
# MODEL="sarvamai/sarvam-30b"
# MODEL="cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
# MODEL="Qwen/Qwen3-30B-A3B-FP8"
# MODEL="Qwen/Qwen3-30B-A3B"
# MODEL="moonshotai/Kimi-Linear-48B-A3B-Instruct"
MODEL="moonshotai/Kimi-K3"
# MODEL="Qwen/Qwen3-235B-A22B"
# MODEL="Qwen/Qwen2.5-1.5B"
INPUT_LEN=4
OUTPUT_LEN=21
NUM_PROMPTS=2

# UVA CPU offloading: only expert weights are offloaded (matched by the
# ".experts." name segment), up to CPU_OFFLOAD_GB per GPU. Set to 0 to disable.
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-2000}
CPU_OFFLOAD_PARAMS=${CPU_OFFLOAD_PARAMS:-experts}

# Fast iteration knobs for debugging (no accuracy):
#   DUMMY_WEIGHTS=1  -> --load-format dummy: skip reading checkpoint shards,
#                       init random weights on device. Every kernel/memory path
#                       still runs (offload path included); output is garbage.
#   NUM_LAYERS=<N>   -> truncate to the first N decoder layers. Kimi-K3 nests
#                       num_hidden_layers under text_config, so we override that.
#                       Layer types/AttnRes use membership/modular tests, so
#                       truncating is safe. N=4 hits every kernel once (KDA dense
#                       MLP, MoE, MLA full-attn at idx3, one AttnRes block); N=13
#                       also crosses the attn_res_block_size=12 boundary.
DUMMY_WEIGHTS=${DUMMY_WEIGHTS:-1}
NUM_LAYERS=${NUM_LAYERS:-47}

# Tensor-parallel size / GPU count. For memory-allocation debugging, run a
# single GPU with a few layers so the process is isolated and the per-node
# NUMA counters in [UVA-ALLOC] reflect exactly one worker:
#   TP=1 NUM_LAYERS=6 sbatch pure_gpu_trial.sh
TP=${TP:-8}

EXTRA_SERVE_ARGS=""
if [[ "$DUMMY_WEIGHTS" == "1" ]]; then
    EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS} --load-format dummy"
fi
if [[ "$NUM_LAYERS" != "0" ]]; then
    EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS} --hf-overrides '{\"text_config\":{\"num_hidden_layers\":${NUM_LAYERS}}}'"
fi

# Torch profiler: set USE_PROFILER=1 to enable the torch profiler on the server
# (--profiler-config) and trigger a capture window during `vllm bench serve`
# (--profile). Traces land in ./vllm_profile. Default off to keep memory-debug
# runs light. Override the config JSON via PROFILER_CONFIG_JSON if needed.
USE_PROFILER=${USE_PROFILER:-0}
PROFILER_CONFIG_JSON=${PROFILER_CONFIG_JSON:-'{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_record_shapes": 1, "torch_profiler_with_flops": 1, "torch_profiler_with_stack": 0, "torch_profiler_with_memory": 1}'}
PROFILER_SERVE_ARGS=""
PROFILE_BENCH_ARG=""
if [[ "$USE_PROFILER" == "1" ]]; then
    PROFILER_SERVE_ARGS="--profiler-config '${PROFILER_CONFIG_JSON}'"
    PROFILE_BENCH_ARG="--profile"
fi

# ---------------------------------------------------------------------------
ZE_AFFINITY_ENV=""
NUMACTL_PREFIX=""
PER_WORKER_ZE_ENV=""
NUMA_BIND_SERVE_ARGS=""
# ---------------------------------------------------------------------------
# Generalized multi-GPU NUMA placement. Set GPU_SCHEME to map the TP ranks onto
# physical GPUs (BDF-sorted, matching ZE_ENABLE_PCI_ID_DEVICE_ORDER=1) and their
# NUMA nodes. Works for any TP; the number of ranks is TP (set TP separately).
#
#   linear    - consecutive GPUs: rank g -> BDF GPU g. Fills node0 first, then
#               node1 (e.g. TP=8 -> GPUs 0,1,2,3,4,5,6,7; nodes 0 0 0 0 1 1 1 1).
#   interleave - round-robin across NUMA nodes: rank0 -> first node0 GPU, rank1
#               -> first node1 GPU, rank2 -> second node0 GPU, ... (e.g. TP=8 ->
#               GPUs 0,4,1,5,2,6,3,7; nodes 0 1 0 1 0 1 0 1).
#
# The visible-device order (ZE_AFFINITY_MASK) follows the chosen order, and
# --numa-bind-nodes carries the matching per-rank node so each worker's CPU +
# host memory bind to its GPU's node. PER_WORKER_ZE_AFFINITY=1 (default) also
# gives each worker a single-GPU ZE_AFFINITY_MASK so its pinned/USM-host buffers
# land on its own node instead of node0. Unset GPU_SCHEME to disable.
#
# Example: GPU_SCHEME=interleave TP=8 NUM_LAYERS=6 sbatch pure_gpu_trial.sh
GPU_SCHEME=${GPU_SCHEME:-linear}
if [[ -n "$GPU_SCHEME" ]]; then
    # BDF-sorted Intel GPU list: "<bdf> <numa_node>" per line.
    mapfile -t _gpu_bdf_node < <(
        for d in /sys/bus/pci/devices/*; do
            cls=$(cat "$d/class" 2>/dev/null)
            ven=$(cat "$d/vendor" 2>/dev/null)
            [[ "$ven" == "0x8086" && "$cls" == 0x0300* ]] || continue
            echo "$(basename "$d") $(cat "$d/numa_node" 2>/dev/null)"
        done | sort
    )
    n_gpu="${#_gpu_bdf_node[@]}"
    if [[ "$n_gpu" -lt "$TP" ]]; then
        echo "GPU_SCHEME: need $TP Intel GPUs, found $n_gpu" >&2
        exit 1
    fi
    # order[] = global BDF-sorted GPU indices assigned to ranks 0..TP-1.
    order=()
    case "$GPU_SCHEME" in
        linear)
            for ((g = 0; g < TP; g++)); do order+=("$g"); done
            ;;
        interleave)
            # Bucket GPU indices by NUMA node (ascending node id), preserving
            # BDF order within each node, then round-robin across the buckets.
            declare -A _bucket=()
            _nodes_seen=()
            for ((g = 0; g < n_gpu; g++)); do
                nn="${_gpu_bdf_node[g]##* }"
                [[ -z "$nn" || "$nn" == "-1" ]] && nn=0
                [[ -z "${_bucket[$nn]:-}" ]] && _nodes_seen+=("$nn")
                _bucket[$nn]="${_bucket[$nn]:-} $g"
            done
            mapfile -t _sorted_nodes < <(printf '%s\n' "${_nodes_seen[@]}" | sort -n)
            declare -A _cursor=()
            while [[ "${#order[@]}" -lt "$TP" ]]; do
                _progressed=0
                for nn in "${_sorted_nodes[@]}"; do
                    read -r -a _arr <<<"${_bucket[$nn]}"
                    idx="${_cursor[$nn]:-0}"
                    if [[ "$idx" -lt "${#_arr[@]}" ]]; then
                        order+=("${_arr[$idx]}")
                        _cursor[$nn]=$((idx + 1))
                        _progressed=1
                        [[ "${#order[@]}" -ge "$TP" ]] && break
                    fi
                done
                [[ "$_progressed" -eq 0 ]] && break
            done
            ;;
        *)
            echo "GPU_SCHEME: unknown scheme '$GPU_SCHEME' (use linear|interleave)" >&2
            exit 1
            ;;
    esac
    mask_ids=(); node_list=()
    rank=0
    for gi in "${order[@]}"; do
        entry="${_gpu_bdf_node[gi]}"
        bdf="${entry%% *}"
        nn="${entry##* }"
        [[ -z "$nn" || "$nn" == "-1" ]] && nn=0
        mask_ids+=("$gi")
        node_list+=("$nn")
        echo "GPU_SCHEME(${GPU_SCHEME}): rank${rank} -> xpu:${rank} = idx${gi}(${bdf}) node${nn}"
        rank=$((rank + 1))
    done
    ze_mask=$(
        IFS=,
        echo "${mask_ids[*]}"
    )
    # ZE_AFFINITY_MASK order defines the visible device order (xpu:i -> order[i]);
    # --numa-bind-nodes carries the matching per-rank node.
    ZE_AFFINITY_ENV="ZE_AFFINITY_MASK=${ze_mask}"
    NUMA_BIND_SERVE_ARGS="--numa-bind --numa-bind-nodes ${node_list[*]}"
    # Give each worker a single-GPU ZE_AFFINITY_MASK so Level-Zero device 0 in
    # the child == its local GPU, keeping pinned/USM-host memory on its node.
    PER_WORKER_ZE_AFFINITY=${PER_WORKER_ZE_AFFINITY:-1}
    if [[ "$PER_WORKER_ZE_AFFINITY" == "1" ]]; then
        PER_WORKER_ZE_ENV="VLLM_XPU_PER_WORKER_ZE_AFFINITY=1"
    fi
fi

GPU_ENV="vllm_0.28.1rc0+kimi_xpu"

CONDA_BASE="/data/nfs_home/sundares/miniforge3"
SCRIPT_DIR="/data/nfs_home/sundares/vllm/vllm/examples/others/nixl_connector"

# Set USE_UNITRACE=1 to trace only the benchmark window with Intel unitrace.
# vllm serve is launched paused; tracing is resumed just before `vllm bench
# serve` and stopped right after, so model load/warmup is excluded.
# Set UNITRACE_PROTOBUF=1 for Perfetto protobuf output (smaller, ui.perfetto.dev),
# or 0 for Chrome trace JSON (chrome://tracing).
USE_UNITRACE=${USE_UNITRACE:-0}
UNITRACE_PROTOBUF=${UNITRACE_PROTOBUF:-0}
UNITRACE_SESSION="kimi"

# export SYCL_UR_USE_LEVEL_ZERO_V2=0
# export LD_PRELOAD=/scratch/ddkalamk/mtrace/mDevTrace.so
# export MDEVTRACE_BT_SIZE=$((10*1024*1024)) 
# export MDEVTRACE_BT_FRAMES=24

export VLLM_LOGGING_LEVEL=DEBUG
# export VLLM_LOGGING_LEVEL=WARNING

# PyTorch pins host memory in powers of two by default: its CachingHostAllocator
# calls PowerOf2Ceil(size) on every pinned allocation (torch 2.13
# aten/src/ATen/core/CachingHostAllocator.h), so a 9.19 GiB expert weight
# reserves 16 GiB, 4.59 GiB -> 8 GiB, etc. (~1.74x blowup per MoE layer). Since
# UVA offload allocates each buffer once and holds it, free-list reuse is
# irrelevant, so disable the rounding. PINNED_MAX_ROUND_MB is the threshold in
# MiB above which allocations are NOT rounded; 0 disables rounding entirely.
# Set PINNED_MAX_ROUND_MB=max (or unset PYTORCH_ALLOC_CONF) to restore rounding.
PINNED_MAX_ROUND_MB=${PINNED_MAX_ROUND_MB:-0}
if [[ "$PINNED_MAX_ROUND_MB" != "max" ]]; then
    export PYTORCH_ALLOC_CONF="pinned_max_round_threshold_mb:${PINNED_MAX_ROUND_MB}"
fi

# Logging helper — prints [file:line func()] message to stderr
log() {
    local file="${BASH_SOURCE[1]##*/}"
    local line="${BASH_LINENO[0]}"
    local func="${FUNCNAME[1]:-main}"
    echo "[${file}:${line} ${func}()] $*" >&2
}

# Kill all processes in a session (setsid makes the launched PID the SID leader,
# so SID == PID. Session membership survives os.setsid() in children unlike PGID.)
kill_session() {
    local sid=$1
    local signal=${2:-TERM}
    [[ -z "$sid" ]] && return 0
    pkill -"$signal" -s "$sid" 2>/dev/null || true
}

stop_server() {
    log "Stopping server…"
    ps -u $USER -o pid,ppid,pgid,sid,cmd || true
    # Graceful SIGTERM to each tracked session
    for sid in "${PIDS[@]}"; do
        kill_session "$sid" TERM
    done
    sleep 30
    ps -u $USER -o pid,ppid,pgid,sid,cmd || true
    # Force SIGKILL any survivors
    for sid in "${PIDS[@]}"; do
        kill_session "$sid" KILL
    done
    wait -- "${PIDS[@]}" 2>/dev/null || true
    PIDS=()
    log "Cleaning up shared memory..."
    find /dev/shm -maxdepth 1 -user $USER -type f -exec rm -f {} +

    # Per-node memory diff
    while IFS='=:' read -r node free_kb actual_kb cache_kb; do
        local vfree="BASELINE_${node^^}_FREE"
        local vactual="BASELINE_${node^^}_ACTUAL"
        local vcache="BASELINE_${node^^}_CACHE"
        local b_free=${!vfree} b_actual=${!vactual} b_cache=${!vcache}
        [[ -z "$b_free" ]] && continue
        local lost_gb=$(echo "scale=2; ($b_free - $free_kb) / 1048576" | bc)
        local b_free_gb=$(echo "scale=2; $b_free / 1048576" | bc)
        local free_gb=$(echo "scale=2; $free_kb / 1048576" | bc)
        local b_actual_gb=$(echo "scale=2; $b_actual / 1048576" | bc)
        local actual_gb=$(echo "scale=2; $actual_kb / 1048576" | bc)
        local b_cache_gb=$(echo "scale=2; $b_cache / 1048576" | bc)
        local cache_gb=$(echo "scale=2; $cache_kb / 1048576" | bc)
        log "${node}: free: ${b_free_gb}GB -> ${free_gb}GB (lost ${lost_gb}GB)  |  actual_used: ${b_actual_gb}GB -> ${actual_gb}GB  |  page_cache: ${b_cache_gb}GB -> ${cache_gb}GB"
    done < <(snapshot_stats_per_node)
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
  local timeout_seconds=12000
  local start_time=$(date +%s)

  log "Waiting for server on port $port..."

  while true; do

    numactl -H | grep -E 'node [0-9]+ (size|free)'
    for node in /sys/devices/system/node/node[0-9]*; do
        nid=$(basename $node | tr -d 'node')
        awk -v n=$nid '/MemTotal/{t=$4} /MemFree/{f=$4} /FilePages/{c=$4} /Slab/{s=$4} END{printf "node%s: total=%.2fGB free=%.2fGB actual_used=%.2fGB page_cache=%.2fGB\n", n, t/1024/1024, f/1024/1024, (t-f-c-s)/1024/1024, (c+s)/1024/1024}' $node/meminfo
    done
    sudo nvtop -s | grep mem_util | awk '{print "GPU " NR-1 ":", $0}'

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

# Print per-node stats as "nodeN=free:actual_used:page_cache" lines (stdout, all in kB)
snapshot_stats_per_node() {
    for node in /sys/devices/system/node/node[0-9]*; do
        nid=$(basename $node | tr -d 'node')
        awk -v n=$nid '/MemTotal/{t=$4} /MemFree/{f=$4} /FilePages/{c=$4} /Slab/{s=$4} END{printf "node%s=%d:%d:%d\n", n, f, t-f-c-s, c+s}' $node/meminfo
    done
}

main() {

    trap cleanup INT
    trap cleanup TERM

    cd $SCRIPT_DIR
    source $CONDA_BASE/etc/profile.d/conda.sh
    conda activate $GPU_ENV

    # Capture memory baseline before server starts
    while IFS='=:' read -r node free_kb actual_kb cache_kb; do
        declare -g "BASELINE_${node^^}_FREE=$free_kb"
        declare -g "BASELINE_${node^^}_ACTUAL=$actual_kb"
        declare -g "BASELINE_${node^^}_CACHE=$cache_kb"
    done < <(snapshot_stats_per_node)
    log "Memory baseline: $(snapshot_stats_per_node | tr '\n' ' ')"

    # python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    # --offload-group-size 1 --offload-num-in-group 1 --offload-prefetch-step 2 \
    # --profiler-config '{\"profiler\": \"torch\", \"torch_profiler_dir\": \"./vllm_profile\", \"torch_profiler_record_shapes\": 1, \"torch_profiler_with_flops\": 1, \"torch_profiler_with_stack\": 1, \"torch_profiler_with_memory\": 1}'
    # --profiler-config '{\"profiler\": \"torch\", \"torch_profiler_dir\": \"./vllm_profile\", \"torch_profiler_record_shapes\": 1, \"torch_profiler_with_flops\": 1, \"torch_profiler_with_stack\": 1, \"torch_profiler_with_memory\": 1, \"capture_torch_profiler\": 1}'

    UNITRACE_PREFIX=""
    if [[ "$USE_UNITRACE" == "1" ]]; then
        mkdir -p ./unitrace_out
        if [[ "$UNITRACE_PROTOBUF" == "1" ]]; then
            UNITRACE_FORMAT_ARGS="--output-format protobuf -o kimi.pb"
        else
            UNITRACE_FORMAT_ARGS="--output-format json -o kimi.json"
        fi
        UNITRACE_PREFIX="unitrace \\
        --chrome-itt-logging   --chrome-sycl-logging  --chrome-call-logging --chrome-kernel-logging \\
        --start-paused --session ${UNITRACE_SESSION} --follow-child-process 1 \\
        --teardown-on-signal 15 ${UNITRACE_FORMAT_ARGS} \\
        --output-dir-path ./unitrace_out"
    fi

    setsid bash -c "
        # export ONEAPI_DEVICE_SELECTOR='level_zero:0,1,4,5;'

        sycl-ls --verbose

        # NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=0 \\
        ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 \\
        VLLM_LOG_STEP_TIMINGS=0 \\
        VLLM_XPU_ENABLE_XPU_GRAPH=1 \\
        TRITON_DEFAULT_BACKEND=intel \\
        ${ZE_AFFINITY_ENV} \\
        ${PER_WORKER_ZE_ENV} \\
        ${UNITRACE_PREFIX} \\
        ${NUMACTL_PREFIX} \\
        \$(which vllm) serve ${MODEL} --trust-remote-code \\
        --port 9000 --max-model-len 200 --max-num-seqs 10 -tp ${TP} --max-num-batched-tokens 200 \\
        --no-enable-prefix-caching --block-size 64 --num-gpu-blocks-override 10 \\
        --gpu-memory-utilization 0.80 \\
        --cpu-offload-gb ${CPU_OFFLOAD_GB} --cpu-offload-params ${CPU_OFFLOAD_PARAMS} \\
        ${EXTRA_SERVE_ARGS} \\
        ${NUMA_BIND_SERVE_ARGS} \\
        ${PROFILER_SERVE_ARGS} \\
        # --offload-group-size 1 --offload-num-in-group 1 --offload-prefetch-step 2 \\
        # --quantization fp8
    " > >(tee server.log) 2>&1 &

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

    if [[ "$USE_UNITRACE" == "1" ]]; then
        log "Resuming unitrace session ${UNITRACE_SESSION} before benchmark"
        unitrace --resume ${UNITRACE_SESSION}
    fi

    # python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    $(which vllm) bench serve --port 9000 --seed $(date +%s) \
        --model $MODEL --trust-remote-code \
        --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
        --num-prompts $NUM_PROMPTS --max-concurrency 1 ${PROFILE_BENCH_ARG} \
        2>&1 | tee benchmark_gpu.log

    if [[ "$USE_UNITRACE" == "1" ]]; then
        log "Stopping unitrace session ${UNITRACE_SESSION} after benchmark"
        unitrace --stop ${UNITRACE_SESSION}
    fi

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
    # curl --fail-with-body -X POST http://localhost:9000/v1/completions \
    # -H "Content-Type: application/json" \
    # -d '{
    #   "model": "'"$MODEL"'",
    #   "prompt": [
    #     "You are a helpful AI assistant. The following context describes a large distributed inference system. The system uses paged KV caching, continuous batching, and tensor parallelism to serve large language models efficiently. Requests are scheduled by a central scheduler that tracks per-request KV cache block allocations. The KV cache is divided into fixed-size blocks, and a block table maps logical blocks to physical GPU memory. Prefix caching reuses KV blocks for identical prompt prefixes across requests, avoiding redundant computation. Now answer the following question: What are the main benefits of prefix caching in LLM serving?"
    #   ],
    #   "min_tokens": 32,
    #   "max_tokens": 200,
    #   "temperature": 0.7
    # }' \
    # 2>&1 | tee accuracy_test_gpu.log

    # curl --fail-with-body -X POST http://localhost:9000/v1/completions \
    # -H "Content-Type: application/json" \
    # -d '{
    #   "model": "'"$MODEL"'",
    #   "prompt": [
    #     "You are a helpful AI assistant. The following context describes a large distributed inference system. The system uses paged KV caching, continuous batching, and tensor parallelism to serve large language models efficiently. Requests are scheduled by a central scheduler that tracks per-request KV cache block allocations. The KV cache is divided into fixed-size blocks, and a block table maps logical blocks to physical GPU memory. Prefix caching reuses KV blocks for identical prompt prefixes across requests, avoiding redundant computation. Now answer the following question: What are the main benefits of prefix caching in LLM serving?",
    #     "You are a helpful AI assistant. The following context describes a large distributed inference system. The system uses paged KV caching, continuous batching, and tensor parallelism to serve large language models efficiently. Requests are scheduled by a central scheduler that tracks per-request KV cache block allocations. The KV cache is divided into fixed-size blocks, and a block table maps logical blocks to physical GPU memory. Prefix caching reuses KV blocks for identical prompt prefixes across requests, avoiding redundant computation. Now answer the following question: How does block-level prefix caching differ from token-level caching?",
    #     "You are a helpful AI assistant. The following context describes a large distributed inference system. The system uses paged KV caching, continuous batching, and tensor parallelism to serve large language models efficiently. Requests are scheduled by a central scheduler that tracks per-request KV cache block allocations. The KV cache is divided into fixed-size blocks, and a block table maps logical blocks to physical GPU memory. Prefix caching reuses KV blocks for identical prompt prefixes across requests, avoiding redundant computation. Now answer the following question: What workloads benefit most from prefix caching and why?"
    #   ],
    #   "max_tokens": 200,
    #   "temperature": 0.7
    # }' \
    # 2>&1 | tee -a accuracy_test_gpu.log

    cleanup success

}

main
