#!/bin/bash
#SBATCH --partition=bmtxb70
#SBATCH --job-name=vllm_pp2tp8
#SBATCH --output=slurm-pp2tp8-%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:59:00

# Two-node pipeline-parallel online serving + benchmark for Kimi-K3 on XPU.
#
# Topology: PP=2 across the two nodes, TP=8 within each node -> world_size=16.
# Backend: multiprocessing (mp) multi-node. `data_parallel_backend` defaults to
# "mp", so pure TP+PP with `--nnodes 2` needs no --data-parallel-* flags.
#
# Node 0 (node-rank 0): full `vllm serve` API server on port 9000 + the
#   `vllm bench serve` client. Spawns global ranks 0..7.
# Node 1 (node-rank 1): `vllm serve --headless`. run_headless() detects
#   node_rank_within_dp > 0 and launches a worker-only MultiprocExecutor that
#   joins node 0's xccl process group. Spawns global ranks 8..15. No API server.
#
# The script re-execs itself under `srun` so the per-node GPU placement and the
# head/worker split run on both nodes. Submit from the repo:
#   sbatch examples/others/nixl_connector/pp_tp_two_node_serve.sh
# Full model (all layers) is the default; override like:
#   NUM_LAYERS=6 DUMMY_WEIGHTS=1 sbatch pp_tp_two_node_serve.sh

set -x

PIDS=()
MODEL=${MODEL:-"moonshotai/Kimi-K3"}
INPUT_LEN=${INPUT_LEN:-4}
OUTPUT_LEN=${OUTPUT_LEN:-21}
NUM_PROMPTS=${NUM_PROMPTS:-3}

# Pipeline- and tensor-parallel sizes. TP is the GPUs per node; PP is the node
# count. world_size = PP * TP must equal total GPUs across both nodes.
PP=${PP:-2}
TP=${TP:-8}

# Nodes actually used by this topology. Each node holds one PP stage's TP ranks,
# so only PP of the allocated nodes participate. Smaller topologies (e.g.
# PP=1 TP=1) use a single node; the rest idle until the job is torn down.
NNODES_USED=${NNODES_USED:-$PP}

# UVA CPU offloading of expert weights (matched by ".experts."), up to
# CPU_OFFLOAD_GB per GPU. Set to 0 to disable.
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-2000}
CPU_OFFLOAD_PARAMS=${CPU_OFFLOAD_PARAMS:-experts}

# DUMMY_WEIGHTS=1 -> --load-format dummy (random weights, every kernel/memory
#   path still runs; output is garbage but fast to bring up).
# NUM_LAYERS=<N>  -> truncate to the first N decoder layers (nested under
#   text_config for Kimi-K3). N=0 keeps the full model (all layers).
DUMMY_WEIGHTS=${DUMMY_WEIGHTS:-1}
NUM_LAYERS=${NUM_LAYERS:-0}

EXTRA_SERVE_ARGS=""
if [[ "$DUMMY_WEIGHTS" == "1" ]]; then
    EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS} --load-format dummy"
fi
if [[ "$NUM_LAYERS" != "0" ]]; then
    EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS} --hf-overrides '{\"text_config\":{\"num_hidden_layers\":${NUM_LAYERS}}}'"
fi

# Torch profiler: USE_PROFILER=1 enables the torch profiler on both serve
# processes (--profiler-config, so node-1 PP-stage workers also capture) and
# triggers a capture window during `vllm bench serve` (--profile). All 16
# workers can share ./vllm_profile: both the tensorboard .pt.trace.json traces
# and the profiler_out_<worker>.txt summary tables are keyed by the globally
# unique worker name (dp/pp/tp/dcp/ep + global rank), so nothing collides
# across nodes. Default off. Override the JSON via PROFILER_CONFIG_JSON.
USE_PROFILER=${USE_PROFILER:-0}
PROFILER_DIR=${PROFILER_DIR:-"./vllm_profile"}
PROFILER_CONFIG_JSON=${PROFILER_CONFIG_JSON:-'{"profiler": "torch", "torch_profiler_dir": "'"${PROFILER_DIR}"'", "torch_profiler_record_shapes": 1, "torch_profiler_with_flops": 1, "torch_profiler_with_stack": 1, "torch_profiler_with_memory": 1}'}
PROFILER_SERVE_ARGS=""
PROFILE_BENCH_ARG=""
if [[ "$USE_PROFILER" == "1" ]]; then
    PROFILER_SERVE_ARGS="--profiler-config '${PROFILER_CONFIG_JSON}'"
    PROFILE_BENCH_ARG="--profile"
fi

GPU_ENV=${GPU_ENV:-"vllm_0.28.1rc0+kimi_xpu"}
CONDA_BASE=${CONDA_BASE:-"/data/nfs_home/sundares/miniforge3"}
SCRIPT_DIR=${SCRIPT_DIR:-"/data/nfs_home/sundares/vllm/vllm/examples/others/nixl_connector"}

# ---------------------------------------------------------------------------
# Launcher: on the batch host, pick the head-node IP for torch.distributed /
# xccl rendezvous, then fan the script out to both nodes with srun. Everything
# after this block runs once per node (node-rank = $SLURM_PROCID).
# ---------------------------------------------------------------------------
if [[ -z "${PP_UNDER_SRUN:-}" ]]; then
    # The #SBATCH --nodes directive is parsed at submit time and can't see $PP,
    # so a small topology (e.g. PP=1) still gets over-allocated. If we hold more
    # nodes than the topology needs, resubmit once with the right --nodes so the
    # job only occupies NNODES_USED (= PP) nodes. RESIZE_ALLOC=0 skips this and
    # just idles the spare nodes instead.
    RESIZE_ALLOC=${RESIZE_ALLOC:-1}
    if [[ "$RESIZE_ALLOC" == "1" && -z "${PP_RESIZED:-}" \
          && "${SLURM_NNODES:-$NNODES_USED}" -gt "$NNODES_USED" ]]; then
        export PP TP NNODES_USED MODEL INPUT_LEN OUTPUT_LEN NUM_PROMPTS
        export CPU_OFFLOAD_GB CPU_OFFLOAD_PARAMS DUMMY_WEIGHTS NUM_LAYERS
        export USE_PROFILER GPU_ENV CONDA_BASE SCRIPT_DIR PP_RESIZED=1
        echo "[launcher] allocation=${SLURM_NNODES} nodes but topology needs" \
             "${NNODES_USED} (PP=${PP}); resubmitting with --nodes=${NNODES_USED}"
        exec sbatch --nodes="${NNODES_USED}" --export=ALL \
            "${SCRIPT_DIR}/pp_tp_two_node_serve.sh"
    fi

    mapfile -t _hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    MASTER_HOST="${_hosts[0]}"
    MASTER_ADDR=$(getent hosts "$MASTER_HOST" | awk '{print $1; exit}')
    [[ -z "$MASTER_ADDR" ]] && MASTER_ADDR="$MASTER_HOST"
    export MASTER_ADDR
    export MASTER_PORT="${MASTER_PORT:-29500}"
    export PP_UNDER_SRUN=1
    # SLURM copies the batch script to a node-0-only spool path, so $0 does not
    # exist on the other nodes. Re-exec the real script from the shared NFS
    # location (SCRIPT_DIR) that every node can read.
    SELF="${SCRIPT_DIR}/pp_tp_two_node_serve.sh"
    if [[ ! -f "$SELF" ]]; then
        echo "[launcher] cannot find shared script at $SELF" >&2
        exit 1
    fi
    echo "[launcher] nodes=${_hosts[*]} master_addr=${MASTER_ADDR} port=${MASTER_PORT} self=${SELF}"
    # Only fan out to the NNODES_USED nodes this topology needs (= PP). The rest
    # of the allocation (if any) is left idle rather than running a doomed
    # worker. For PP=1 this launches a single node.
    srun --nodes="${NNODES_USED}" --ntasks="${NNODES_USED}" \
        --kill-on-bad-exit=1 --ntasks-per-node=1 --export=ALL bash "$SELF"
    exit $?
fi

# ===========================================================================
# Per-node body (under srun)
# ===========================================================================
NODE_RANK=${SLURM_PROCID:-0}

hostname
echo "[node ${NODE_RANK}] master_addr=${MASTER_ADDR} port=${MASTER_PORT} PP=${PP} TP=${TP}"
numactl -H | grep -E 'node [0-9]+ (size|free)'
for node in /sys/devices/system/node/node[0-9]*; do
    nid=$(basename $node | tr -d 'node')
    awk -v n=$nid '/MemTotal/{t=$4} /MemFree/{f=$4} /FilePages/{c=$4} /Slab/{s=$4} END{printf "node%s: total=%.2fGB free=%.2fGB actual_used=%.2fGB page_cache=%.2fGB\n", n, t/1024/1024, f/1024/1024, (t-f-c-s)/1024/1024, (c+s)/1024/1024}' $node/meminfo
done

# ---------------------------------------------------------------------------
# Per-node NUMA GPU placement (same scheme as pure_gpu_trial.sh). Each node has
# its own 8 GPUs, so the BDF-sorted mask is computed locally; TP ranks on this
# node map onto this node's physical GPUs.
# ---------------------------------------------------------------------------
ZE_AFFINITY_ENV=""
PER_WORKER_ZE_ENV=""
NUMA_BIND_SERVE_ARGS=""
GPU_SCHEME=${GPU_SCHEME:-linear}
if [[ -n "$GPU_SCHEME" ]]; then
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
    order=()
    case "$GPU_SCHEME" in
        linear)
            for ((g = 0; g < TP; g++)); do order+=("$g"); done
            ;;
        interleave)
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
        echo "GPU_SCHEME(${GPU_SCHEME}) node${NODE_RANK}: rank${rank} -> xpu:${rank} = idx${gi}(${bdf}) node${nn}"
        rank=$((rank + 1))
    done
    ze_mask=$(
        IFS=,
        echo "${mask_ids[*]}"
    )
    ZE_AFFINITY_ENV="ZE_AFFINITY_MASK=${ze_mask}"
    NUMA_BIND_SERVE_ARGS="--numa-bind --numa-bind-nodes ${node_list[*]}"
    PER_WORKER_ZE_AFFINITY=${PER_WORKER_ZE_AFFINITY:-1}
    if [[ "$PER_WORKER_ZE_AFFINITY" == "1" ]]; then
        PER_WORKER_ZE_ENV="VLLM_XPU_PER_WORKER_ZE_AFFINITY=1"
    fi
fi

export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-DEBUG}

# Disable PyTorch's power-of-two pinned-host rounding so each UVA expert buffer
# is allocated at its true size (see pure_gpu_trial.sh for the rationale).
PINNED_MAX_ROUND_MB=${PINNED_MAX_ROUND_MB:-0}
if [[ "$PINNED_MAX_ROUND_MB" != "max" ]]; then
    export PYTORCH_ALLOC_CONF="pinned_max_round_threshold_mb:${PINNED_MAX_ROUND_MB}"
fi

# oneCCL cross-node transport. CCL_ATL_TRANSPORT accepts only ofi|mpi; for
# TCP/sockets use ofi + FI_PROVIDER=tcp.
export CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT:-ofi}
export FI_PROVIDER=${FI_PROVIDER:-tcp}

# Per-command env shared by the head and worker `vllm serve` invocations. The
# per-node affinity vars (set above) are expanded now; the result is spliced in
# as a command prefix inside each setsid block.
COMMON_SERVE_ENV="VLLM_XPU_MOE_GATHER_OFFLOAD=${VLLM_XPU_MOE_GATHER_OFFLOAD:-1} \
ZE_ENABLE_PCI_ID_DEVICE_ORDER=${ZE_ENABLE_PCI_ID_DEVICE_ORDER:-1} \
VLLM_LOG_STEP_TIMINGS=${VLLM_LOG_STEP_TIMINGS:-0} \
VLLM_XPU_ENABLE_XPU_GRAPH=${VLLM_XPU_ENABLE_XPU_GRAPH:-1} \
TRITON_DEFAULT_BACKEND=intel \
${ZE_AFFINITY_ENV} \
${PER_WORKER_ZE_ENV}"

log() {
    local file="${BASH_SOURCE[1]##*/}"
    local line="${BASH_LINENO[0]}"
    local func="${FUNCNAME[1]:-main}"
    echo "[${file}:${line} ${func}() node${NODE_RANK}] $*" >&2
}

kill_session() {
    local sid=$1
    local signal=${2:-TERM}
    [[ -z "$sid" ]] && return 0
    pkill -"$signal" -s "$sid" 2>/dev/null || true
}

stop_server() {
    log "Stopping server…"
    for sid in "${PIDS[@]}"; do
        kill_session "$sid" TERM
    done
    sleep 30
    for sid in "${PIDS[@]}"; do
        kill_session "$sid" KILL
    done
    wait -- "${PIDS[@]}" 2>/dev/null || true
    PIDS=()
    log "Cleaning up shared memory..."
    find /dev/shm -maxdepth 1 -user $USER -type f -exec rm -f {} + 2>/dev/null || true
}

# End the whole SLURM job so the headless worker on node 1 also exits once the
# head node has finished benchmarking (the worker's serve blocks until killed).
terminate_job() {
    log "Terminating SLURM job ${SLURM_JOB_ID} to stop all nodes."
    scancel "${SLURM_JOB_ID}" 2>/dev/null || true
}

cleanup() {
    local reason=${1:-interrupt}
    log "Stopping everything… (reason: ${reason})"
    trap '' INT TERM
    stop_server
    terminate_job
    exit 0
}

wait_for_server() {
    local port=$1
    local pid=$2
    local timeout_seconds=12000
    local start_time=$(date +%s)
    log "Waiting for server on port $port..."
    while true; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log "Server PID $pid terminated unexpectedly."
            cleanup unexpected
        fi
        if curl -s "localhost:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        local now=$(date +%s)
        if (( now - start_time >= timeout_seconds )); then
            log "ERROR: Timeout waiting for server on port $port"
            cleanup timeout
        fi
        sleep 2
    done
}

head_main() {
    trap cleanup INT
    trap cleanup TERM

    setsid bash -c "
        sycl-ls --verbose

        ${COMMON_SERVE_ENV} \\
        \$(which vllm) serve ${MODEL} --trust-remote-code \\
        --port 9000 --max-model-len 2000 --max-num-seqs 10 \\
        -tp ${TP} -pp ${PP} \\
        --nnodes ${NNODES_USED} --node-rank ${NODE_RANK} \\
        --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} \\
        --max-num-batched-tokens 2000 \\
        --no-enable-prefix-caching --block-size 64 --num-gpu-blocks-override 10 \\
        --gpu-memory-utilization 0.80 \\
        --cpu-offload-gb ${CPU_OFFLOAD_GB} --cpu-offload-params ${CPU_OFFLOAD_PARAMS} \\
        ${EXTRA_SERVE_ARGS} \\
        ${NUMA_BIND_SERVE_ARGS} \\
        ${PROFILER_SERVE_ARGS}
    " > >(tee server_head.log) 2>&1 &

    server_pid=$!
    PIDS+=($server_pid)

    wait_for_server 9000 $server_pid

    log "=================================================="
    log "Server up (PP=${PP} TP=${TP} across ${NNODES_USED} nodes). Benchmarking…"
    log "=================================================="

    $(which vllm) bench serve --port 9000 --seed $(date +%s) \
        --model $MODEL --trust-remote-code \
        --dataset-name random --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --num-prompts $NUM_PROMPTS --max-concurrency 1 ${PROFILE_BENCH_ARG} \
        2>&1 | tee benchmark_pp${PP}_tp${TP}.log

    cleanup success
}

worker_main() {
    trap 'exit 0' INT TERM

    setsid bash -c "
        sycl-ls --verbose

        ${COMMON_SERVE_ENV} \\
        \$(which vllm) serve ${MODEL} --trust-remote-code --headless \\
        --port 9001 --max-model-len 2000 --max-num-seqs 10 \\
        -tp ${TP} -pp ${PP} \\
        --nnodes ${NNODES_USED} --node-rank ${NODE_RANK} \\
        --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} \\
        --max-num-batched-tokens 2000 \\
        --no-enable-prefix-caching --block-size 64 --num-gpu-blocks-override 10 \\
        --gpu-memory-utilization 0.80 \\
        --cpu-offload-gb ${CPU_OFFLOAD_GB} --cpu-offload-params ${CPU_OFFLOAD_PARAMS} \\
        ${EXTRA_SERVE_ARGS} \\
        ${NUMA_BIND_SERVE_ARGS} \\
        ${PROFILER_SERVE_ARGS}
    " > >(tee server_worker.log) 2>&1 &

    worker_pid=$!
    PIDS+=($worker_pid)
    log "Headless worker serving (ranks $((TP * NODE_RANK))..$((TP * NODE_RANK + TP - 1)))."
    # Block until the head tears the job down (SIGTERM via scancel).
    wait "$worker_pid"
}

# Allocated-but-unused node (NODE_RANK >= NNODES_USED), e.g. the second node
# under PP=1. Do nothing but stay alive so srun keeps the step up until the
# head node finishes and scancels the job.
idle_main() {
    trap 'exit 0' INT TERM
    log "Node ${NODE_RANK} unused for PP=${PP} TP=${TP} (NNODES_USED=${NNODES_USED}); idling until teardown."
    while true; do
        sleep 300 &
        wait $!
    done
}

main() {
    cd "$SCRIPT_DIR"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$GPU_ENV"

    if (( NODE_RANK >= NNODES_USED )); then
        idle_main
    elif [[ "$NODE_RANK" == "0" ]]; then
        head_main
    else
        worker_main
    fi
}

main
