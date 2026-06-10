# Installation Instructions
## NVIDIA GPUs:
```bash
echo '
set -xe
# Ensure that the tag is present as it is needed for proper versioning purpose
# git fetch vllm_public tag v0.18.0 --no-tags
# git reset --hard v0.18.0
# conda create -n vllm_0.18.0_cuda python==3.12 -y
eval "$(conda shell.bash hook)"
conda activate vllm_0.18.0_cuda
source /swtools/cuda/12.9.0/cuda_vars.sh
# TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" pip install -r requirements/cuda.txt --extra-index-url https://download.pytorch.org/whl/cu129 -v 
# pip install setuptools_scm
# rm -rf .deps dist *.egg-info
# TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" pip install . --no-build-isolation -v --extra-index-url https://download.pytorch.org/whl/cu129
# git ls-files --others --exclude='.vscode' --exclude='example*' --exclude="build*" --exclude=".deps" | xargs rm
# Ensure to comment out optional modules in setup.py
NVCC_THREADS=4 TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" pip install -e . --no-build-isolation -v --extra-index-url https://download.pytorch.org/whl/cu129 --config-settings editable_mode=strict
TARGET_DIR=$(ls -dt build/__editable__.vllm-* 2>/dev/null | head -1) && \
[ -n "$TARGET_DIR" ] || { echo "Error: No editable build directory found"; exit 1; } && \
git ls-files --others --exclude='.vscode' --exclude='example*' --exclude="build*" --exclude=".deps" | \
grep -E "\.so$" | \
while IFS= read -r file; do \
  mkdir -p "$TARGET_DIR/$(dirname "$file")" && \
  mv "$file" "$TARGET_DIR/$file" && \
  echo "Moved: $file"; \
done
set +xe
' | bash 2>&1 | tee build_cuda_0.18.0_$(date +%Y%m%d_%H%M%S).log
```
## Intel GPUs:
```bash
bash << 'SCRIPT' 2>&1 | tee build_xpu_0.19.1_$(date +%Y%m%d_%H%M%S).log
set -xe
# Ensure that the tag is present as it is needed for proper versioning purpose
# git fetch origin tag v0.19.1 --no-tags
# git reset --hard v0.19.1
# conda create -n vllm_0.19.1_xpu python==3.12 -y
eval "$(conda shell.bash hook)"
# Load oneAPI2025.3 and driver modules
# mkdir -p ~/miniforge3/envs/vllm_0.19.1_xpu/etc/conda/activate.d
cat > ~/miniforge3/envs/vllm_0.19.1_xpu/etc/conda/activate.d/xpu-vars.activate.sh << 'EOF'
#!/bin/bash

[[ "$-" != *x* ]] && _xtrace_was_off=1 && set -x

source /swtools/intel/oneapi/2025.3/oneapi-vars.sh 
# source /swtools/intel-gpu/latest/intel_gpu_vars.sh
# source /swtools/intel-gpu/26.01.36711.4/intel_gpu_vars.sh
# source /swtools/intel-gpu/main_20251004/intel_gpu_vars.sh
# export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

export FI_PROVIDER=tcp

if [[ -n "$_xtrace_was_off" ]]; then set +x; unset _xtrace_was_off; fi
EOF
conda activate vllm_0.19.1_xpu
pip install "pip<26"
set -x
pip install -r requirements/xpu.txt --extra-index-url=https://download.pytorch.org/whl/xpu -v
# rm -rf .deps dist *.egg-info
# git ls-files --others --exclude='.vscode' --exclude='example*' --exclude="build*" --exclude=".deps" | xargs rm
VLLM_TARGET_DEVICE=xpu pip install -e . --no-build-isolation -v --extra-index-url=https://download.pytorch.org/whl/xpu --config-settings editable_mode=strict
pip uninstall -y triton triton-xpu
pip install triton-xpu==3.7.0 --extra-index-url https://download.pytorch.org/whl/xpu
set +xe
SCRIPT
```
## CPUs:
```bash
bash << 'SCRIPT' 2>&1 | tee build_cpu_0.19.1_$(date +%Y%m%d_%H%M%S).log
set -xe
# Ensure that the tag is present as it is needed for proper versioning purpose
# git fetch origin tag v0.19.1 --no-tags
# git reset --hard v0.19.1
conda create -n vllm_0.19.1_cpu python==3.12 -y
eval "$(conda shell.bash hook)"
mkdir -p ~/miniforge3/envs/vllm_0.19.1_cpu/etc/conda/activate.d
cat > ~/miniforge3/envs/vllm_0.19.1_cpu/etc/conda/activate.d/cpu-vars.activate.sh << 'EOF'
#!/bin/bash

[[ "$-" != *x* ]] && _xtrace_was_off=1 && set -x

TC_PATH="/data/nfs_home/sundares/miniforge3/envs/vllm_0.19.1_cpu/lib/libtcmalloc_minimal.so"
IOMP_PATH="/swtools/intel/oneapi/2025.3/lib/libiomp5.so"

export LD_PRELOAD="${TC_PATH}:${IOMP_PATH}${LD_PRELOAD:+:${LD_PRELOAD}}"

if [[ -n "$_xtrace_was_off" ]]; then set +x; unset _xtrace_was_off; fi
EOF
mkdir -p ~/miniforge3/envs/vllm_0.19.1_cpu/etc/conda/deactivate.d
cat > ~/miniforge3/envs/vllm_0.19.1_cpu/etc/conda/deactivate.d/cpu-vars.deactivate.sh << 'EOF'
#!/bin/bash

[[ "$-" != *x* ]] && _xtrace_was_off=1 && set -x

TC_PATH="/data/nfs_home/sundares/miniforge3/envs/vllm_0.19.1_cpu/lib/libtcmalloc_minimal.so"
IOMP_PATH="/swtools/intel/oneapi/2025.3/lib/libiomp5.so"

LD_PRELOAD=":${LD_PRELOAD}:"
LD_PRELOAD="${LD_PRELOAD//:${TC_PATH}:/:}"
LD_PRELOAD="${LD_PRELOAD//:${IOMP_PATH}:/:}"
# Collapse any consecutive colons left by the removals
while [[ "$LD_PRELOAD" == *::* ]]; do LD_PRELOAD="${LD_PRELOAD//::/:}"; done
LD_PRELOAD="${LD_PRELOAD#:}"
LD_PRELOAD="${LD_PRELOAD%:}"
if [[ -z "$LD_PRELOAD" ]]; then unset LD_PRELOAD; else export LD_PRELOAD; fi

if [[ -n "$_xtrace_was_off" ]]; then set +x; unset _xtrace_was_off; fi
EOF
conda activate vllm_0.19.1_cpu
pip install "pip<26"
pip install -r requirements/cpu-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
conda install -y gperftools
rm -rf .deps dist *.egg-info
# git ls-files --others --exclude='.vscode' --exclude='example*' --exclude="build*" --exclude=".deps" | xargs rm
# VLLM_CPU_AMXBF16=true VLLM_TARGET_DEVICE=cpu pip install . --no-build-isolation -v --extra-index-url https://download.pytorch.org/whl/cpu 
VLLM_CPU_AMXBF16=true VLLM_TARGET_DEVICE=cpu pip install -e . --no-build-isolation -v --extra-index-url https://download.pytorch.org/whl/cpu --config-settings editable_mode=strict
TARGET_DIR=$(ls -dt build/__editable__.vllm-* 2>/dev/null | head -1) && \
[ -n "$TARGET_DIR" ] || { echo "Error: No editable build directory found"; exit 1; } && \
git ls-files --others --exclude='.vscode' --exclude='example*' --exclude="build*" --exclude=".deps" | \
grep -E "\.so$" | \
while IFS= read -r file; do \
  mkdir -p "$TARGET_DIR/$(dirname "$file")" && \
  mv "$file" "$TARGET_DIR/$file" && \
  echo "Moved: $file"; \
done
set +xe
SCRIPT
```

# Run Scripts
## Disagg Prefill/Decode Serving:
```bash
# The run and evaluation scripts are part of it
bash disagg_example_shm.sh
```
## Pure GPU/CPU:

### Nvidia GPU Server side
```bash
export MODEL="Qwen/Qwen2.5-1.5B"
export VLLM_LOGGING_LEVEL=DEBUG 
# Optional
export VLLM_KV_CACHE_LAYOUT="NHD"
# Has effect for GPU only. For offloading KV cache to CPU
export VLLM_OFFLOAD_KV_CACHE_TO_CPU=0 
# TORCH_COMPILE_DISABLE=1
# Has effect for GPU only. For weights offloading
export VLLM_DOUBLE_BUFFER_PIPELINE=0 
# Remove profiler config if profiling not required
$(which vllm) serve $MODEL --port 9000 --max-model-len 9000     --max-num-seqs 10     --max-num-batched-tokens 70000 --enforce-eager --no-enable-prefix-caching --block-size 64 --offload-group-size 1 --offload-num-in-group 1 --offload-prefetch-step 2 --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_record_shapes": 1, "torch_profiler_with_flops": 1, "torch_profiler_with_stack": 1, "torch_profiler_with_memory": 1}'
```

### Intel GPU Server side
```bash
export MODEL="Qwen/Qwen2.5-7B"
export VLLM_LOGGING_LEVEL=DEBUG 
# Optional
export VLLM_KV_CACHE_LAYOUT="NHD"
# Has effect for GPU only. For offloading KV cache to CPU
export VLLM_OFFLOAD_KV_CACHE_TO_CPU=1 
# TORCH_COMPILE_DISABLE=1
# Remove profiler config if profiling not required
$(which vllm) serve $MODEL --port 9000 --max-model-len 9000     --max-num-seqs 10     --max-num-batched-tokens 70000 --enforce-eager --no-enable-prefix-caching --block-size 64 --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_record_shapes": 1, "torch_profiler_with_flops": 1, "torch_profiler_with_stack": 1, "torch_profiler_with_memory": 1}'
```

### CPU Server side
```bash
# Update the below paths accordingly
TC_PATH="/data/nfs_home/sundares/miniforge3/envs/vllm_0.19.1_cpu/lib/libtcmalloc_minimal.so"
IOMP_PATH="/swtools/intel/oneapi/2025.3/lib/libiomp5.so"

export LD_PRELOAD="${TC_PATH}:${IOMP_PATH}${LD_PRELOAD:+:${LD_PRELOAD}}"
export MODEL="Qwen/Qwen3-30B-A3B"
export VLLM_LOGGING_LEVEL=DEBUG 
# Optional
export VLLM_KV_CACHE_LAYOUT="NHD"
# Has effect for CPU only
export VLLM_CPU_SGL_KERNEL="1" 
export VLLM_CPU_OMP_THREADS_BIND="0-63|64-127"
export VLLM_CPU_KVCACHE_SPACE=40 
# TORCH_COMPILE_DISABLE=1
# Remove profiler config if profiling not required
$(which vllm) serve $MODEL --port 9000 --max-model-len 9000     --max-num-seqs 10     --max-num-batched-tokens 70000 --enforce-eager --no-enable-prefix-caching --block-size 64 --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_record_shapes": 1, "torch_profiler_with_flops": 1, "torch_profiler_with_stack": 1, "torch_profiler_with_memory": 1}'
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