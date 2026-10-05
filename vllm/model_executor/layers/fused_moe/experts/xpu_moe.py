# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kInt4Static,
    kInt4Static32,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform

if current_platform.is_xpu():
    from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe


logger = init_logger(__name__)


def _validate_expert_map(
    expert_map: torch.Tensor,
    global_num_experts: int,
    local_num_experts: int,
    device: torch.device,
) -> None:
    """Validate that ``expert_map`` matches the expected XPU MoE contract.

    Raises:
        ValueError: If the dtype, shape, contiguity, device, or value range
            of ``expert_map`` does not match the expected contract.
    """
    if expert_map.dtype != torch.int32:
        raise ValueError("XPU MoE expert_map must have dtype torch.int32")
    if expert_map.dim() != 1 or expert_map.numel() != global_num_experts:
        raise ValueError(
            "XPU MoE expert_map must have shape "
            f"({global_num_experts},), got {tuple(expert_map.shape)}"
        )
    if not expert_map.is_contiguous():
        raise ValueError("XPU MoE expert_map must be contiguous")
    if expert_map.device != device:
        raise ValueError(
            "XPU MoE expert_map and hidden states must be on the same device"
        )
    if bool(((expert_map < -1) | (expert_map >= local_num_experts)).any()):
        raise ValueError(
            "XPU MoE expert_map values must be -1 or local expert IDs in "
            f"[0, {local_num_experts})"
        )


def prepare_fp8_moe_layer_for_xpu(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transpose fp8 MoE weights and scales into the XPU kernel layout.

    Returns:
        The ``(w13, w13_scale, w2, w2_scale)`` tuple with weights transposed
        on their last two dimensions and 3-D scales similarly transposed.
    """
    if w13_scale is not None and w13_scale.ndim == 3:
        w13_scale = w13_scale.transpose(-1, -2).contiguous()
    if w2_scale is not None and w2_scale.ndim == 3:
        w2_scale = w2_scale.transpose(-1, -2).contiguous()
    return (
        w13.transpose(-1, -2).contiguous(),
        w13_scale,
        w2.transpose(-1, -2).contiguous(),
        w2_scale,
    )


# JIT-compiled SYCL kernel that gathers selected expert rows from a (possibly
# UVA / pinned-host) ``[E, bpe]`` uint8 table into a contiguous device buffer.
# Under multi-GPU PCIe contention it holds the contiguous link rate
# (~25 GiB/s/GPU) where ``torch.index_select`` on the UVA view collapses to
# about half that, so the offload gather prefers this kernel and falls back to
# ``index_select`` only when it is unavailable.
_GATHER_EXPERTS_SYCL = r"""
#include <sycl/sycl.hpp>
#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>
#include <ATen/DeviceGuard.h>
#include <algorithm>
#include <cstdint>

namespace {

// One work-group row copies one output expert; the second range dimension is a
// grid-stride loop over 16-byte vectors so wide loads stay coalesced. Reads of
// `src` cross PCIe when `src` is a pinned-host (UVA) allocation.
class gather_experts_kernel {
 public:
  gather_experts_kernel(
      uint8_t* dst,
      const uint8_t* src,
      const int32_t* ids,
      int64_t num_active,
      int64_t vecs_per_expert)
      : dst_(dst),
        src_(src),
        ids_(ids),
        num_active_(num_active),
        vpe_(vecs_per_expert) {}

  void operator()(sycl::nd_item<2> item) const {
    using vec_t = sycl::vec<uint32_t, 4>;  // 16-byte copy unit
    const int64_t row = item.get_global_id(0);
    if (row >= num_active_) return;
    const int64_t src_row = static_cast<int64_t>(ids_[row]);
    const vec_t* __restrict src_vec =
        reinterpret_cast<const vec_t*>(src_) + src_row * vpe_;
    vec_t* __restrict dst_vec = reinterpret_cast<vec_t*>(dst_) + row * vpe_;
    const int64_t stride = item.get_global_range(1);
    for (int64_t c = item.get_global_id(1); c < vpe_; c += stride) {
      dst_vec[c] = src_vec[c];
    }
  }

 private:
  uint8_t* dst_;
  const uint8_t* src_;
  const int32_t* ids_;
  int64_t num_active_;
  int64_t vpe_;
};

}  // namespace

void gather_experts_out(
    torch::Tensor dst,         // [num_active, bpe] uint8, device
    torch::Tensor src,         // [num_experts, bpe] uint8, UVA or device
    torch::Tensor expert_ids)  // [num_active] int32, device
{
  TORCH_CHECK(dst.scalar_type() == at::kByte, "dst must be uint8");
  TORCH_CHECK(src.scalar_type() == at::kByte, "src must be uint8");
  TORCH_CHECK(expert_ids.scalar_type() == at::kInt, "expert_ids must be int32");
  TORCH_CHECK(dst.dim() == 2 && src.dim() == 2, "dst/src must be 2D");
  const int64_t bpe = src.size(1);
  TORCH_CHECK(dst.size(1) == bpe, "dst/src byte width mismatch");
  TORCH_CHECK(bpe % 16 == 0, "bytes_per_expert must be a multiple of 16");
  const int64_t num_active = dst.size(0);
  TORCH_CHECK(
      expert_ids.numel() == num_active, "expert_ids length must match dst rows");
  TORCH_CHECK(
      dst.is_contiguous() && expert_ids.is_contiguous(),
      "dst and expert_ids must be contiguous");

  const at::DeviceGuard guard(dst.device());
  const int64_t vpe = bpe / 16;
  auto* dst_ptr = reinterpret_cast<uint8_t*>(dst.data_ptr());
  auto* src_ptr = reinterpret_cast<const uint8_t*>(src.data_ptr());
  auto* ids_ptr = expert_ids.data_ptr<int32_t>();

  auto& queue = c10::xpu::getCurrentXPUStream(dst.device().index()).queue();

  constexpr int64_t col_wg = 256;
  const int64_t col_groups =
      std::min<int64_t>((vpe + col_wg - 1) / col_wg, 256);
  sycl::range<2> local(1, col_wg);
  sycl::range<2> global(num_active, col_groups * col_wg);

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(global, local),
        gather_experts_kernel(dst_ptr, src_ptr, ids_ptr, num_active, vpe));
  });
}

PYBIND11_MODULE(xpu_gather_experts_ext, m) {
  m.def(
      "gather_experts_out",
      &gather_experts_out,
      "Gather selected expert rows from (UVA) src into a contiguous device "
      "buffer dst.");
}
"""

_gather_experts_ext = None
_gather_experts_build_failed = False


def _get_gather_experts_ext():
    """Return the cached inline SYCL gather extension, or ``None``.

    The first call JIT-compiles the kernel (torch serializes concurrent builds
    across ranks via its own file lock). On any build failure the error is
    logged once and ``None`` is returned so callers fall back to a pure-torch
    gather instead of crashing the serving process.
    """
    global _gather_experts_ext, _gather_experts_build_failed
    if _gather_experts_ext is not None:
        return _gather_experts_ext
    if _gather_experts_build_failed:
        return None

    from torch.utils.cpp_extension import load_inline

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        _gather_experts_build_failed = True
        return None

    try:
        _gather_experts_ext = load_inline(
            name="xpu_gather_experts_ext",
            cpp_sources=[],
            sycl_sources=_GATHER_EXPERTS_SYCL,
            extra_sycl_cflags=["-O3", "-std=c++17"],
            with_sycl=True,
            verbose=False,
            keep_intermediates=False,
        )
    except Exception:
        _gather_experts_build_failed = True
        logger.warning(
            "Failed to build the XPU gather-experts SYCL kernel; falling "
            "back to torch.index_select for MoE weight offload gather.",
            exc_info=True,
        )
        return None
    return _gather_experts_ext


def _gather_rows(t: torch.Tensor | None, ids: torch.Tensor) -> torch.Tensor | None:
    """Gather rows ``ids`` (dim 0) of ``t`` into a contiguous device buffer.

    For offloaded weights ``t`` is a UVA view of host memory, so the gather is
    the PCIe transfer. The SYCL ``gather_experts_out`` kernel holds the
    contiguous link rate under multi-GPU contention; it needs a 2D ``[E, bpe]``
    uint8 source with ``bpe`` a multiple of 16, so each expert's bytes are
    flattened (a no-copy view for contiguous ``t``) and the result is viewed
    back to ``t``'s dtype and shape. Falls back to ``index_select`` when the
    kernel is unavailable or the byte width is not 16-aligned.
    """
    if t is None:
        return None
    num_active = int(ids.numel())
    num_experts = t.shape[0]
    if t.is_contiguous():
        src = t.view(torch.uint8).reshape(num_experts, -1)
        bpe = src.shape[1]
        if bpe % 16 == 0:
            ext = _get_gather_experts_ext()
            if ext is not None:
                dst = torch.empty(
                    num_active, bpe, dtype=torch.uint8, device=t.device
                )
                ext.gather_experts_out(dst, src, ids.to(torch.int32))
                return dst.view(t.dtype).reshape(num_active, *t.shape[1:])
    if t.element_size() == 1:
        return t.view(torch.uint8).index_select(0, ids).view(t.dtype)
    return t.index_select(0, ids).contiguous()


def _run_gather_offload_pipe(
    impl: "XpuFusedMoe",
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    a1q_scale: torch.Tensor | None,
) -> None:
    """Gather the routed experts to device, then run the packed GEMM.

    Replaces the grouped GEMM's on-demand streaming of ``impl.w13``/``impl.w2``
    straight from the UVA view with an explicit high-bandwidth gather of just
    the routed experts into contiguous device buffers, then runs the grouped
    GEMM on those packed weights with a remapped expert space.

    The implementation is fully static-shaped and free of device->host syncs so
    it is safe to trace inside an XPU command-graph (cudagraph) capture: it
    gathers a fixed ``max_active = min(num_local, num_routings)`` rows (padding
    unused slots) rather than the data-dependent count of distinct experts, and
    builds the packed routing map with on-device comparisons, cumsum and
    reductions instead of ``torch.unique``, scatter or ``tensor.item()``.
    """
    if getattr(impl, "_use_ref", False):
        impl.apply(
            output=output,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            a1q_scale=a1q_scale,
        )
        return
    num_local = impl.num_experts
    device = topk_ids.device
    # Static upper bound on the number of distinct routed experts: one per
    # (token, top-k) routing, capped at the local expert count. Derived from
    # tensor shapes (known at trace time), never from tensor values.
    num_routings = topk_ids.shape[0] * topk_ids.shape[1]
    max_active = min(num_local, num_routings)

    flat_ids = topk_ids.reshape(-1).to(torch.int64)
    if expert_map is not None:
        local_ids = expert_map.index_select(0, flat_ids)
        map_len = expert_map.numel()
    else:
        local_ids = flat_ids
        map_len = num_local
    valid = local_ids >= 0
    local_safe = local_ids.clamp(min=0)

    expert_range = torch.arange(num_local, dtype=torch.int32, device=device)
    # used[e] is True if expert e is routed by any valid routing. Built with a
    # one-hot comparison + reduction rather than scatter / index_put_, which
    # trigger an event wait that is illegal during command-graph capture.
    onehot = (local_safe.unsqueeze(1) == expert_range.unsqueeze(0)) & (
        valid.unsqueeze(1)
    )
    used = onehot.any(0)  # [num_local] bool
    used_i = used.to(torch.int32)
    # Exclusive prefix count -> packed slot of each used expert.
    pos = used_i.cumsum(0).to(torch.int32) - used_i

    # gather_ids[s] = expert id assigned to packed slot s (0 for padded slots).
    # Built with a comparison + reduction for the same capture-safety reason.
    slot_range = torch.arange(max_active, dtype=torch.int32, device=device)
    match = (pos.unsqueeze(0) == slot_range.unsqueeze(1)) & used.unsqueeze(0)
    gather_ids = (match.to(torch.int32) * expert_range.unsqueeze(0)).sum(1)
    gather_ids = gather_ids.to(torch.int32)

    # Global expert id -> packed slot (or -1 if not routed).
    if expert_map is not None:
        em = expert_map.to(torch.int64)
        em_safe = em.clamp(min=0)
        pm = pos.index_select(0, em_safe)
        routed = (em >= 0) & used.index_select(0, em_safe)
        packed_map = torch.where(
            routed, pm, torch.full_like(pm, -1)
        ).to(torch.int32)
    else:
        packed_map = torch.where(
            used, pos, torch.full_like(pos, -1)
        ).to(torch.int32)

    gathered = (
        _gather_rows(impl.w13, gather_ids),
        _gather_rows(impl.w2, gather_ids),
        _gather_rows(impl.gemm1_wei_scales, gather_ids),
        _gather_rows(impl.gemm2_wei_scales, gather_ids),
        _gather_rows(impl.w13_bias, gather_ids),
        _gather_rows(impl.w2_bias, gather_ids),
    )

    saved = (
        impl.w13,
        impl.w2,
        impl.gemm1_wei_scales,
        impl.gemm2_wei_scales,
        impl.w13_bias,
        impl.w2_bias,
        impl.num_experts,
        impl.local_experts_num,
        impl.total_experts_num,
    )
    try:
        (
            impl.w13,
            impl.w2,
            impl.gemm1_wei_scales,
            impl.gemm2_wei_scales,
            impl.w13_bias,
            impl.w2_bias,
        ) = gathered
        impl.num_experts = max_active
        impl.local_experts_num = max_active
        impl.total_experts_num = map_len
        impl._apply_kernel(
            output,
            hidden_states,
            topk_weights,
            topk_ids,
            expert_map=packed_map,
            a1q_scale=a1q_scale,
        )
    finally:
        (
            impl.w13,
            impl.w2,
            impl.gemm1_wei_scales,
            impl.gemm2_wei_scales,
            impl.w13_bias,
            impl.w2_bias,
            impl.num_experts,
            impl.local_experts_num,
            impl.total_experts_num,
        ) = saved


class XPUExperts(mk.FusedMoEExpertsModular):
    """Fused MoE expert execution backed by ``vllm_xpu_kernels.XpuFusedMoe``."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        """Initialize the expert module and check for Xe2/Xe3 GPU support."""
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        self.fused_moe_impl: XpuFusedMoe | None = None
        is_xe2_or_xe3 = torch.ops._xpu_C.is_xe2_arch() or torch.ops._xpu_C.is_xe3_arch()
        if not is_xe2_or_xe3:
            raise NotImplementedError(
                "XPUExperts is only supported on Intel Xe2/Xe3 GPUs"
            )
        self._expects_unquantized_inputs = is_xe2_or_xe3

    @property
    def expects_unquantized_inputs(self) -> bool:
        """Whether this expert implementation expects unquantized inputs."""
        return self._expects_unquantized_inputs

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        """Return the activation tensor layout expected by this backend."""
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        """Return whether the current platform is XPU."""
        return current_platform.is_xpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        """Return whether activation-without-multiply variants are supported."""
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        """Return whether ``activation`` is supported by the XPU kernel."""
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SITU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        """Return whether ``moe_parallel_config`` is supported (always True)."""
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Return whether the given weight/activation quantization is supported."""
        SUPPORTED_W_A = [
            (None, None),
            (kFp8StaticTensorSym, None),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        """Return the no-op finalizer, since weighting/reduce happens in-kernel."""
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Return workspace and output shapes; no extra workspace is needed."""
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        """Run the fused MoE forward pass, writing results into ``output``.

        Lazily constructs the underlying ``XpuFusedMoe`` kernel wrapper on
        first call, then dispatches tokens to experts and writes the
        combined result into ``output`` in place.
        """
        if expert_map is not None:
            _validate_expert_map(
                expert_map,
                global_num_experts,
                self.moe_config.num_local_experts,
                hidden_states.device,
            )
        if self.fused_moe_impl is None:
            topk = topk_ids.size(-1)
            if (
                self.quant_config is not None
                and self.quant_config.weight_quant_dtype == "mxfp4"
            ):
                w1 = w1.view(torch.float4_e2m1fn_x2)
                w2 = w2.view(torch.float4_e2m1fn_x2)
            self.fused_moe_impl = XpuFusedMoe(
                w13=w1,
                w13_scales=self.w1_scale,
                w13_bias=self.w1_bias,
                w2=w2,
                w2_scales=self.w2_scale,
                w2_bias=self.w2_bias,
                n_experts_per_token=topk,
                activation=activation.value,
                num_experts=self.moe_config.num_local_experts,
                ep_rank=self.moe_config.ep_rank,
                ep_size=self.moe_config.ep_size,
                expert_map=expert_map,
                gemm1_clamp_limit=self.gemm1_clamp_limit,
                activation_situ_beta=self.moe_config.activation_situ_beta,
                activation_situ_linear_beta=(
                    self.moe_config.activation_situ_linear_beta
                ),
            )
        assert self.fused_moe_impl is not None
        if expert_map is not None:
            self.fused_moe_impl.expert_map = expert_map
        if envs.VLLM_XPU_MOE_GATHER_OFFLOAD:
            _run_gather_offload_pipe(
                self.fused_moe_impl,
                output,
                hidden_states,
                topk_weights,
                topk_ids,
                expert_map,
                a1q_scale,
            )
            return
        self.fused_moe_impl.apply(
            output=output,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            a1q_scale=a1q_scale,
        )


class XPUExpertsFp8(XPUExperts):
    """XPU MoE experts for static-tensor fp8 weight quantization."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        """Initialize the fp8 expert module."""
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Return whether the given weight/activation quantization is supported."""
        SUPPORTED_W_A = [
            (kFp8StaticTensorSym, None),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class XPUExpertsMxFp8(XPUExpertsFp8):
    """XPU MoE experts for static mxfp8 weight quantization."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        """Initialize the mxfp8 expert module and assert the quant dtype."""
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        assert quant_config.quant_dtype == "mxfp8"

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Return whether the given weight/activation quantization is supported."""
        SUPPORTED_W_A = [
            (kMxfp8Static, None),
            (kMxfp8Static, kMxfp8Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class XPUExpertsBlockFp8(XPUExperts):
    """XPU MoE experts for block-wise fp8 weight quantization."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        """Initialize the block-fp8 expert module."""
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Return whether the given weight/activation quantization is supported."""
        SUPPORTED_W_A = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class XPUExpertsWNA16(XPUExperts):
    """W4A16 INT4-symmetric MoE backed by `xpu_fused_moe(is_int4=True)`.

    Weight layout when `is_int4=True` (per `xpu_fused_moe` docstring):
        w13: [num_experts, 2*inter_size, hidden_size]   contiguous int4-packed
        w13_scales: [num_experts, 2*inter_size, hidden_size // group_size]
        w2:  [num_experts, hidden_size, inter_size]     contiguous int4-packed
        w2_scales:  [num_experts, hidden_size, inter_size // group_size]

    Pairs with `INCXPULinearMethod` for the linear layers; together they
    cover full-attn + MoE on Intel XPU end-to-end without IPEX.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        """Initialize the INT4-symmetric expert module."""
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Return whether the given weight/activation quantization is supported."""
        return (weight_key, activation_key) in (
            (kInt4Static, None),
            (kInt4Static32, None),
        )


class XPUExpertsMxFp4(XPUExperts):
    """XPU MoE experts for static mxfp4 weight quantization."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        """Initialize the mxfp4 expert module."""
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Return workspace and output shapes, accounting for mxfp4 packing."""
        # K = a1q.size(-1). When activations are pre-quantized packed mxfp4,
        # K is the packed hidden_size (= logical / 2); the kernel output is at
        # logical hidden_size (2 * K). When unquantized (bf16), K is already
        # the logical size.
        logical_K = K if self.expects_unquantized_inputs else 2 * K
        return (0,), (0,), (M, logical_K)

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Return whether the given weight/activation quantization is supported."""
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
            (kMxfp4Static, kMxfp4Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A
