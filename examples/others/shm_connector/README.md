# Installation Instructions
## NVIDIA GPUs:
```bash
echo '
set -xe
# Ensure that the tag is present as it is needed for proper versioning purpose
# git fetch origin tag v0.15.1 --no-tags
# git reset --hard v0.15.1
conda create -n vllm_0.15.1_shm_cuda python==3.12 -y
eval "$(conda shell.bash hook)"
conda activate vllm_0.15.1_shm_cuda
source /swtools/cuda/12.9.0/cuda_vars.sh
TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" pip install -r requirements/cuda.txt --extra-index-url https://download.pytorch.org/whl/cu129 -v 
pip install setuptools_scm
rm -rf .deps build dist *.egg-info
TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" pip install . --no-build-isolation -v --extra-index-url https://download.pytorch.org/whl/cu129 --no-cache-dir
# TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" pip install -e . --no-build-isolation -v --extra-index-url https://download.pytorch.org/whl/cu129 --no-cache-dir --config-settings editable_mode=strict
set +xe
' | bash 2>&1 | tee build_cuda_$(date +%Y%m%d_%H%M%S).log
```
## Intel GPUs:
```bash
echo '
set -xe
# Ensure that the tag is present as it is needed for proper versioning purpose
# git fetch origin tag v0.15.1 --no-tags
# git reset --hard v0.15.1
conda create -n vllm_0.15.1_shm_xpu python==3.12 -y
eval "$(conda shell.bash hook)"
# Load oneAPI2025.2 and driver modules
mkdir ~/miniforge3/envs/vllm_0.15.1_shm_xpu/etc/conda/activate.d
cp ~/miniforge3/envs/vllm_0.13.0_shm_xpu/etc/conda/activate.d/xpu-vars.activate.sh ~/miniforge3/envs/vllm_0.15.1_shm_xpu/etc/conda/activate.d
conda activate vllm_0.15.1_shm_xpu
pip install -r requirements/xpu.txt --extra-index-url=https://download.pytorch.org/whl/xpu -v
rm -rf .deps build dist *.egg-info
VLLM_TARGET_DEVICE=xpu pip install . --no-build-isolation -v --extra-index-url=https://download.pytorch.org/whl/xpu --no-cache-dir
set +xe
' | bash 2>&1 | tee build_xpu_$(date +%Y%m%d_%H%M%S).log
```
## CPUs:
```bash
echo '
set -xe
# Ensure that the tag is present as it is needed for proper versioning purpose
# git fetch origin tag v0.15.1 --no-tags
# git reset --hard v0.15.1
conda create -n vllm_0.15.1_shm_cpu python==3.12 -y
eval "$(conda shell.bash hook)"
conda activate vllm_0.15.1_shm_cpu
pip install -r requirements/cpu-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
rm -rf .deps build dist *.egg-info
VLLM_CPU_AMXBF16=true VLLM_TARGET_DEVICE=cpu pip install . --no-build-isolation -v --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir 
# VLLM_CPU_AMXBF16=true VLLM_TARGET_DEVICE=cpu pip install -e . --no-build-isolation -v --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir --config-settings editable_mode=strict
set +xe
' | bash 2>&1 | tee build_cpu_$(date +%Y%m%d_%H%M%S).log
```

# Run Scripts
## Disagg Prefill/Decode Serving:
```bash
# The run and evaluation scripts are part of it
bash disagg_example_shm.sh
```
## Pure GPU/CPU:

### GPU Server side
```bash
export MODEL="Qwen/Qwen3-30B-A3B"
export VLLM_LOGGING_LEVEL=DEBUG 
# Optional
export VLLM_KV_CACHE_LAYOUT="NHD"
# Has effect for GPU only. For offloading KV cache to CPU
export VLLM_OFFLOAD_KV_CACHE_TO_CPU=0 
# TORCH_COMPILE_DISABLE=1
# Has effect for GPU only. For weights offloading
export VLLM_DOUBLE_BUFFER_PIPELINE=0 
# Remove profiler config if profiling not required
$(which vllm) serve $MODEL --port 9000 --max-model-len 9000     --max-num-seqs 10     --max-num-batched-tokens 70000 --enforce-eager --no-enable-prefix-caching --block-size 64 --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_record_shapes": 1, "torch_profiler_with_flops": 1, "torch_profiler_with_stack": 1, "torch_profiler_with_memory": 1}'
```

### CPU Server side
```bash
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
export MODEL="Qwen/Qwen3-30B-A3B"
export INPUT_LEN=8192
export OUTPUT_LEN=8
export NUM_PROMPTS=1
export VLLM_LOGGING_LEVEL=DEBUG 
# Remove profile at the end if not needed
$(which vllm) bench serve --port 9000 --seed $(date +%s)         --model $MODEL         --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN         --num-prompts $NUM_PROMPTS --max-concurrency 1 --ignore-eos --profile
```
### For accuracy test
```bash
export MODEL="Qwen/Qwen3-30B-A3B"
curl -X POST http://localhost:9000/v1/completions -H "Content-Type: application/json" -d '{    "model": "'"$MODEL"'",    "prompt": "Write a rich, vivid, slightly humorous free-verse poem about the craft of software engineering and coding. Describe in detail long nights spent debugging elusive bugs, the glow of multiple monitors, half-finished mugs of cold coffee, and the quiet hum of machines in an almost empty office or home workspace. Show the emotional roller coaster of reading confusing legacy code, adding one more log line, watching stack traces scroll by, and wondering what the previous developer was thinking when they designed this system. Include scenes of collaboration: pair programming sessions, code review comments that are both kind and blunt, whiteboard diagrams that start neat and end as chaotic scribbles, and chat messages full of links to docs, tickets, and pull requests. Mention modern tools and rituals of the craft: version control, feature branches, continuous integration pipelines, flaky tests, deployment scripts, and dashboards that flip from red to green. Contrast the stress of production incidents, paging alerts, and frantic hotfixes with the quiet, satisfying moment when all tests finally pass, the pipeline is green, and the release is tagged. Use concrete imagery that developers recognize, add gentle inside jokes about off by one errors and mysterious race conditions, and keep the overall tone hopeful and affirming. Celebrate the creativity, persistence, and teamwork that make software possible, and end on a note of cautious but genuine optimism about the next refactor, the next big feature, and the next late night that somehow feels worth it.", "max_tokens": 100,    "temperature": 0.7  }'
```