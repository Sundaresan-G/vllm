# Installation Instructions
## NVIDIA GPUs:
```bash
bash << 'SCRIPT' 2>&1 | tee build_cuda_0.26.1rc0_$(date +%Y%m%d_%H%M%S).log
set -xe
# Ensure that the tag is present as it is needed for proper versioning purpose
# git fetch vllm_public tag v0.26.1rc0 --no-tags
# git reset --hard v0.26.1rc0
CONDA_BASE="/data/nfs_home/sundares/miniforge3"
source $CONDA_BASE/etc/profile.d/conda.sh
conda create -n vllm_0.26.1rc0_cuda python==3.12 -y
mkdir -p $CONDA_BASE/envs/vllm_0.26.1rc0_cuda/etc/conda/activate.d
cat > $CONDA_BASE/envs/vllm_0.26.1rc0_cuda/etc/conda/activate.d/cuda-vars.activate.sh << 'EOF'
#!/bin/bash

[[ "$-" != *x* ]] && _xtrace_was_off=1 && set -x

source /swtools/cuda/12.9.0/cuda_vars.sh

if [[ -n "$_xtrace_was_off" ]]; then set +x; unset _xtrace_was_off; fi
EOF
conda activate vllm_0.26.1rc0_cuda
pip install "pip<26"
pip install -r requirements/cuda.txt --extra-index-url https://download.pytorch.org/whl/cu129 -v
pip install setuptools_scm
pip install setuptools_rust
# rm -rf .deps dist *.egg-info
# pip install . --no-build-isolation -v --extra-index-url https://download.pytorch.org/whl/cu129
# git ls-files --others --exclude='.vscode' --exclude='example*' --exclude="build*" --exclude=".deps" | xargs rm
# Ensure to comment out optional modules in setup.py
VLLM_VERSION_OVERRIDE="v0.26.1rc0" NVCC_THREADS=4 pip install -e . --no-build-isolation -v --extra-index-url https://download.pytorch.org/whl/cu129 --config-settings editable_mode=strict
TARGET_DIR=$(ls -dt build/__editable__.vllm-* 2>/dev/null | head -1) && \
[ -n "$TARGET_DIR" ] || { echo "Error: No editable build directory found"; exit 1; } && \
git ls-files --others --exclude='.vscode' --exclude='example*' --exclude="build*" --exclude=".deps" | \
grep -E "\.so$" | \
while IFS= read -r file; do \
  mkdir -p "$TARGET_DIR/$(dirname "$file")" && \
  mv "$file" "$TARGET_DIR/$file" && \
  echo "Moved: $file to $TARGET_DIR/$file"; \
done
# NIXL is required for the NixlConnector (see requirements/kv_connectors.txt).
# Pin the already-installed torch so nixl's other deps resolve without upgrading/replacing it.
pip freeze | grep -E '^torch==' > /tmp/nixl_torch_constraint.txt
pip install nixl --constraint /tmp/nixl_torch_constraint.txt
set +xe
SCRIPT
```
## Intel GPUs:
```bash
bash << 'SCRIPT' 2>&1 | tee build_xpu_0.26.1rc0_$(date +%Y%m%d_%H%M%S).log
set -xe
# Ensure that the tag is present as it is needed for proper versioning purpose
# git fetch origin tag v0.26.1rc0 --no-tags
# git reset --hard v0.26.1rc0
CONDA_BASE="/data/nfs_home/sundares/miniforge3"
source $CONDA_BASE/etc/profile.d/conda.sh
conda create -n vllm_0.26.1rc0_xpu python==3.12 -y
# Load oneAPI2025.3 and driver modules
mkdir -p $CONDA_BASE/envs/vllm_0.26.1rc0_xpu/etc/conda/activate.d
cat > $CONDA_BASE/envs/vllm_0.26.1rc0_xpu/etc/conda/activate.d/xpu-vars.activate.sh << 'EOF'
#!/bin/bash

[[ "$-" != *x* ]] && _xtrace_was_off=1 && set -x

# source /swtools/intel/2026.0/oneapi-vars.sh
source /swtools/intel/2025.3/oneapi-vars.sh
source /swtools/intel-gpu/latest/intel_gpu_vars.sh
# source /swtools/intel-gpu/26.01.36711.4/intel_gpu_vars.sh
# source /swtools/intel-gpu/main_20251004/intel_gpu_vars.sh
# export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

export FI_PROVIDER=tcp

if [[ -n "$_xtrace_was_off" ]]; then set +x; unset _xtrace_was_off; fi
EOF
conda activate vllm_0.26.1rc0_xpu
pip install "pip<26"
set -x
pip install -r requirements/xpu.txt --extra-index-url=https://download.pytorch.org/whl/xpu -v
# rm -rf .deps dist *.egg-info
# git ls-files --others --exclude='.vscode' --exclude='example*' --exclude="build*" --exclude=".deps" | xargs rm
VLLM_VERSION_OVERRIDE="v0.26.1rc0" VLLM_TARGET_DEVICE=xpu pip install -e . --no-build-isolation -v --extra-index-url=https://download.pytorch.org/whl/xpu --config-settings editable_mode=strict
TARGET_DIR=$(ls -dt build/__editable__.vllm-* 2>/dev/null | head -1) && \
[ -n "$TARGET_DIR" ] || { echo "Error: No editable build directory found"; exit 1; } && \
git ls-files --others --exclude='.vscode' --exclude='example*' --exclude="build*" --exclude=".deps" | \
grep -E "\.so$" | \
while IFS= read -r file; do \
  mkdir -p "$TARGET_DIR/$(dirname "$file")" && \
  mv "$file" "$TARGET_DIR/$file" && \
  echo "Moved: $file to $TARGET_DIR/$file"; \
done

# TODO: Currently no support for VLLM_OFFLOAD_KV_CACHE_TO_CPU due to NIXL. Implement it
# Now for installing vllm-xpu-kernels
# pip uninstall -y vllm-xpu-kernels
# cd ..
# [ -d ./vllm-xpu-kernels ] || git clone --single-branch --branch copy_cache_flash https://github.com/Sundaresan-G/vllm-xpu-kernels.git
# cd vllm-xpu-kernels
# git fetch origin tag v0.1.10 --no-tags
# git checkout f86cd8ac855b61ac4839d9fe2fe3f390a9baae2f

# git submodule update --init --recursive
# pip install -r requirements.txt
# rm -rf build
# VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf pip install --no-build-isolation . -v
# cd ../vllm

# NIXL is required for the NixlConnector (see requirements/kv_connectors.txt).
# Pin the already-installed torch so nixl's other deps resolve without upgrading/replacing it.
pip freeze | grep -E '^torch==' > /tmp/nixl_torch_constraint.txt
pip install nixl --constraint /tmp/nixl_torch_constraint.txt

pip uninstall -y triton triton-xpu
pip install triton-xpu==3.7.1 --extra-index-url=https://download.pytorch.org/whl/xpu

set +xe
SCRIPT
```
## CPUs:
```bash
bash << 'SCRIPT' 2>&1 | tee build_cpu_0.26.1rc0_$(date +%Y%m%d_%H%M%S).log
set -xe
# Ensure that the tag is present as it is needed for proper versioning purpose
# git fetch origin tag v0.26.1rc0 --no-tags
# git reset --hard v0.26.1rc0
CONDA_BASE="/data/nfs_home/sundares/miniforge3"
source $CONDA_BASE/etc/profile.d/conda.sh
source /swtools/intel/2025.3/oneapi-vars.sh
conda create -n vllm_0.26.1rc0_cpu python==3.12 -y
conda activate vllm_0.26.1rc0_cpu
pip install "pip<26"
pip install -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install setuptools_rust
pip install setuptools_scm
conda install -y gperftools
rm -rf .deps dist *.egg-info
# git ls-files --others --exclude='.vscode' --exclude='example*' --exclude="build*" --exclude=".deps" | xargs rm
# VLLM_CPU_AMXBF16=true VLLM_TARGET_DEVICE=cpu pip install . --no-build-isolation -v --extra-index-url https://download.pytorch.org/whl/cpu 
VLLM_VERSION_OVERRIDE="v0.26.1rc0" VLLM_CPU_AMXBF16=true VLLM_TARGET_DEVICE=cpu pip install -e . --no-build-isolation -v --extra-index-url https://download.pytorch.org/whl/cpu --config-settings editable_mode=strict
TARGET_DIR=$(ls -dt build/__editable__.vllm-* 2>/dev/null | head -1) && \
[ -n "$TARGET_DIR" ] || { echo "Error: No editable build directory found"; exit 1; } && \
git ls-files --others --exclude='.vscode' --exclude='example*' --exclude="build*" --exclude=".deps" | \
grep -E "\.so$" | \
while IFS= read -r file; do \
  mkdir -p "$TARGET_DIR/$(dirname "$file")" && \
  mv "$file" "$TARGET_DIR/$file" && \
  echo "Moved: $file to $TARGET_DIR/$file"; \
done
mkdir -p $CONDA_BASE/envs/vllm_0.26.1rc0_cpu/etc/conda/activate.d
cat > $CONDA_BASE/envs/vllm_0.26.1rc0_cpu/etc/conda/activate.d/cpu-vars.activate.sh << EOF
#!/bin/bash

[[ "\$-" != *x* ]] && _xtrace_was_off=1 && set -x

TC_PATH="$CONDA_BASE/envs/vllm_0.26.1rc0_cpu/lib/libtcmalloc_minimal.so"
IOMP_PATH="$CONDA_BASE/envs/vllm_0.26.1rc0_cpu/lib/libiomp5.so"

export LD_PRELOAD="\${TC_PATH}:\${IOMP_PATH}\${LD_PRELOAD:+:\${LD_PRELOAD}}"

if [[ -n "\$_xtrace_was_off" ]]; then set +x; unset _xtrace_was_off; fi
EOF
mkdir -p $CONDA_BASE/envs/vllm_0.26.1rc0_cpu/etc/conda/deactivate.d
cat > $CONDA_BASE/envs/vllm_0.26.1rc0_cpu/etc/conda/deactivate.d/cpu-vars.deactivate.sh << EOF
#!/bin/bash

[[ "\$-" != *x* ]] && _xtrace_was_off=1 && set -x

LD_PRELOAD=":\${LD_PRELOAD}:"
LD_PRELOAD="\${LD_PRELOAD//:\${TC_PATH}:/:}"
LD_PRELOAD="\${LD_PRELOAD//:\${IOMP_PATH}:/:}"

unset TC_PATH
unset IOMP_PATH

# Collapse any consecutive colons left by the removals
while [[ "\$LD_PRELOAD" == *::* ]]; do LD_PRELOAD="\${LD_PRELOAD//::/:}"; done
LD_PRELOAD="\${LD_PRELOAD#:}"
LD_PRELOAD="\${LD_PRELOAD%:}"
if [[ -z "\$LD_PRELOAD" ]]; then unset LD_PRELOAD; else export LD_PRELOAD; fi

if [[ -n "\$_xtrace_was_off" ]]; then set +x; unset _xtrace_was_off; fi
EOF
# NIXL is required for the NixlConnector (see requirements/kv_connectors.txt).
# Pin the already-installed torch so nixl's other deps resolve without upgrading/replacing it.
pip freeze | grep -E '^torch==' > /tmp/nixl_torch_constraint.txt
pip install nixl --constraint /tmp/nixl_torch_constraint.txt
set +xe
SCRIPT
```

# Run Scripts
## Disagg Prefill/Decode Serving:
```bash
# Launches a GPU prefiller + CPU decoder + toy proxy behind a single OpenAI-compatible
# endpoint, connected via NixlConnector. The run and evaluation (bench/accuracy) steps
# are part of it.
bash disagg_example_nixl.sh
```
Notes:
- `prefiller_decoder_vllm_launcher.sh prefiller|decoder <model>` is the per-role launcher
  invoked by `disagg_example_nixl.sh`. The prefiller (`kv_role: kv_producer`) runs with
  `VLLM_KV_CACHE_LAYOUT="NHD"`; the decoder (`kv_role: kv_consumer`) runs the CPU backend
  (`VLLM_CPU_KVCACHE_SPACE`, `VLLM_CPU_SGL_KERNEL=1`). Both use `--block-size 64` and
  `--kv-transfer-config {"kv_connector": "NixlConnector", ...}`.
- `toy_proxy_server.py` fans requests out to the prefiller then the decoder and stitches
  the streamed response together; it listens on port 9000 by default.
- `enforce_handshake_compat: false` in `kv_connector_extra_config` is a plumbing bypass
  for heterogeneous (GPU prefiller / CPU decoder) layouts.
- `VLLM_OFFLOAD_KV_CACHE_TO_CPU` (offloading the prefiller's KV cache to host memory) is
  **not supported yet** with NIXL. If/when it is, `UCX_TLS=tcp,sm,self` and
  `UCX_MEMTYPE_CACHE=n` would need to be set on both roles so NIXL's one-sided RDMA uses
  host transports instead of picking a CUDA transport against pinned host memory
  (`ptrace_scope=0` is required for the `sm`/CMA transport).

## Pure GPU/CPU:

### Nvidia/Intel GPU Server side
```bash
sbatch pure_gpu_trial.sh
```

### CPU Server side
```bash
sbatch pure_cpu_trial.sh
```

## Evaluation Scripts - Client side:
### For Performance measurement
```bash
export MODEL="Qwen/Qwen2.5-1.5B-Instruct"
export INPUT_LEN=8192
export OUTPUT_LEN=8
export NUM_PROMPTS=5
export VLLM_LOGGING_LEVEL=DEBUG 
# Remove profile at the end if not needed
$(which vllm) bench serve --port 9000 --seed $(date +%s)         --model $MODEL         --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN         --num-prompts $NUM_PROMPTS --max-concurrency 1 --ignore-eos --profile
```
### For accuracy test
```bash
export MODEL="Qwen/Qwen3-30B-A3B"
curl -X POST http://localhost:9000/v1/completions -H "Content-Type: application/json" -d '{    "model": "'"$MODEL"'",    "prompt": "Write a rich, vivid, slightly humorous free-verse poem about the craft of software engineering and coding. Describe in detail long nights spent debugging elusive bugs, the glow of multiple monitors, half-finished mugs of cold coffee, and the quiet hum of machines in an almost empty office or home workspace. Show the emotional roller coaster of reading confusing legacy code, adding one more log line, watching stack traces scroll by, and wondering what the previous developer was thinking when they designed this system. Include scenes of collaboration: pair programming sessions, code review comments that are both kind and blunt, whiteboard diagrams that start neat and end as chaotic scribbles, and chat messages full of links to docs, tickets, and pull requests. Mention modern tools and rituals of the craft: version control, feature branches, continuous integration pipelines, flaky tests, deployment scripts, and dashboards that flip from red to green. Contrast the stress of production incidents, paging alerts, and frantic hotfixes with the quiet, satisfying moment when all tests finally pass, the pipeline is green, and the release is tagged. Use concrete imagery that developers recognize, add gentle inside jokes about off by one errors and mysterious race conditions, and keep the overall tone hopeful and affirming. Celebrate the creativity, persistence, and teamwork that make software possible, and end on a note of cautious but genuine optimism about the next refactor, the next big feature, and the next late night that somehow feels worth it.", "max_tokens": 100,    "temperature": 0.7  }'
```