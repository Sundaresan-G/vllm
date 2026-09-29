# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVA-based CPU offloading using Unified Virtual Addressing."""

import os
from collections.abc import Generator

import torch
import torch.nn as nn
from torch.func import functional_call

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader, should_pin_memory
from vllm.utils.mem_utils import format_gib
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)

_GIB = 1024**3
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def _read_rss_bytes() -> int:
    """Resident set size of this process in bytes."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * _PAGE_SIZE
    except Exception:
        return 0


def _read_numa_free_bytes() -> dict[int, int]:
    """System-wide free memory per NUMA node in bytes."""
    free: dict[int, int] = {}
    base = "/sys/devices/system/node"
    try:
        for entry in os.listdir(base):
            if not entry.startswith("node") or not entry[4:].isdigit():
                continue
            node = int(entry[4:])
            try:
                with open(f"{base}/{entry}/meminfo") as f:
                    for line in f:
                        if "MemFree:" in line:
                            free[node] = int(line.split()[3]) * 1024
                            break
            except Exception:
                free[node] = 0
    except Exception:
        pass
    return free


def _read_self_numa_bytes() -> dict[int, int]:
    """Bytes this process has mapped on each NUMA node (from numa_maps)."""
    mapped: dict[int, int] = {}
    try:
        with open("/proc/self/numa_maps") as f:
            for line in f:
                for tok in line.split():
                    if tok.startswith("N") and "=" in tok:
                        node_str, _, pages_str = tok[1:].partition("=")
                        if node_str.isdigit() and pages_str.isdigit():
                            node = int(node_str)
                            mapped[node] = (
                                mapped.get(node, 0) + int(pages_str) * _PAGE_SIZE
                            )
    except Exception:
        pass
    return mapped


def _fmt_node_deltas(before: dict[int, int], after: dict[int, int]) -> str:
    """Format per-node *consumed* bytes (free decrease) as ``n0=+X n1=+Y``."""
    nodes = sorted(set(before) | set(after))
    parts = []
    for n in nodes:
        consumed = before.get(n, 0) - after.get(n, 0)
        parts.append(f"n{n}={'+' if consumed >= 0 else '-'}{abs(consumed) / _GIB:.2f}GiB")
    return " ".join(parts)


def _fmt_node_abs(vals: dict[int, int]) -> str:
    """Format absolute per-node bytes as ``n0=X n1=Y``."""
    return " ".join(f"n{n}={vals.get(n, 0) / _GIB:.2f}GiB" for n in sorted(vals))


class UVAOffloader(BaseOffloader):
    """Offloader using Unified Virtual Addressing (UVA) for zero-copy access.

    This offloader moves parameters to pinned CPU memory and creates CUDA views
    using UVA. The GPU can then directly access the CPU memory without explicit
    transfers, at the cost of PCIe bandwidth (slower than GPU memory).

    When UVA is disabled via env var, falls back to a functional_call-based
    approach that moves parameters on-demand.

    Args:
        cpu_offload_max_bytes: Maximum bytes to offload to CPU.
        cpu_offload_params: Set of parameter name segments to selectively
            offload. If empty, all parameters are eligible up to the byte limit.
    """

    supports_tower_offload = True

    def __init__(
        self,
        cpu_offload_max_bytes: int,
        cpu_offload_params: set[str] | None = None,
    ):
        self.cpu_offload_max_bytes = cpu_offload_max_bytes
        self.cpu_offload_bytes = 0
        self.cpu_offload_params = cpu_offload_params or set()

        self.pin_memory = should_pin_memory()
        self.uva_offloading = (
            is_uva_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA
        )

        # Diagnostics: per-parameter offload/resident tracking so it's easy
        # to see exactly which parameters ended up offloaded vs. left on
        # device, and how large each one is. Each entry is
        # (name, size_bytes, shape, dtype, element_size).
        self.offloaded_param_info: list[
            tuple[str, int, torch.Size, torch.dtype, int]
        ] = []
        self.resident_param_info: list[
            tuple[str, int, torch.Size, torch.dtype, int]
        ] = []

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
        prefix: str = "",
    ) -> list[nn.Module]:
        """Wrap modules with UVA offloading."""
        if prefix:
            prefix = f"{prefix}."
        modules = [
            self._maybe_offload_to_cpu(module, prefix) for module in modules_generator
        ]
        if self.cpu_offload_bytes > 0:
            logger.info(
                "Total CPU offloaded parameters: %s",
                format_gib(self.cpu_offload_bytes),
            )

        for name, size, shape, dtype, elsize in self.offloaded_param_info:
            logger.debug(
                "  [OFFLOADED] %s: shape=%s dtype=%s elsize=%dB %s GiB",
                name,
                tuple(shape),
                dtype,
                elsize,
                format_gib(size),
            )
        for name, size, shape, dtype, elsize in self.resident_param_info:
            logger.debug(
                "  [RESIDENT]  %s: shape=%s dtype=%s elsize=%dB %s GiB",
                name,
                tuple(shape),
                dtype,
                elsize,
                format_gib(size),
            )

        total_offloaded = sum(size for _, size, *_ in self.offloaded_param_info)
        total_resident = sum(size for _, size, *_ in self.resident_param_info)
        logger.debug(
            "  [TOTAL] offloaded=%s GiB (%d params), resident=%s GiB (%d params)",
            format_gib(total_offloaded),
            len(self.offloaded_param_info),
            format_gib(total_resident),
            len(self.resident_param_info),
        )

        return modules

    def _maybe_offload_to_cpu(self, module: nn.Module, prefix: str = "") -> nn.Module:
        """Offload module parameters to CPU using UVA if budget allows."""
        if (params := next(module.parameters(), None)) is None:
            return module

        device = params.device

        if device == torch.device("cpu"):
            return module

        module_tag = type(module).__name__
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is not None:
            module_tag = f"{module_tag}[L{layer_idx}]"

        if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
            for name, p in module.named_parameters():
                self.resident_param_info.append(
                    (
                        f"{module_tag}.{name}",
                        p.numel() * p.element_size(),
                        p.shape,
                        p.dtype,
                        p.element_size(),
                    )
                )
            return module

        # offload parameters to CPU
        # use pin_memory if possible, which helps cudagraph capture speed
        offloaded_parameters = False
        for name, p in module.named_parameters():
            if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
                # we use per-parameter offloading
                # one module might have some parameters offloaded and some not
                self.resident_param_info.append(
                    (
                        f"{module_tag}.{name}",
                        p.numel() * p.element_size(),
                        p.shape,
                        p.dtype,
                        p.element_size(),
                    )
                )
                break

            # Skip parameters an earlier wrap_modules call already offloaded.
            # The UVA path leaves p.device as the accelerator (a view of CPU
            # memory), so the marker is the only way to recognize those.
            if p.device.type == "cpu" or getattr(p, "_vllm_is_uva_offloaded", False):
                continue

            if self.cpu_offload_params:
                # Check if parameter belongs to the offloading set
                # Add dots here to ensure we match full segments only
                # e.g., "experts.w2_weight" matches "mlp.experts.w2_weight"
                # but not "mlp.experts.w2_weight_scale"
                should_offload = any(
                    f".{param}." in f".{prefix}{name}."
                    for param in self.cpu_offload_params
                )
                if not should_offload:
                    self.resident_param_info.append(
                        (
                            f"{module_tag}.{name}",
                            p.numel() * p.element_size(),
                            p.shape,
                            p.dtype,
                            p.element_size(),
                        )
                    )
                    continue

            param_bytes = p.numel() * p.element_size()
            logger.debug(
                "  [UVA-ALLOC] dev=%s %s.%s size=%s GiB cum_pinned=%s GiB",
                device,
                module_tag,
                name,
                format_gib(param_bytes),
                format_gib(self.cpu_offload_bytes),
            )

            rss0 = _read_rss_bytes()
            free0 = _read_numa_free_bytes()

            cpu_data = p.data.to(device="cpu")
            rss1 = _read_rss_bytes()
            free1 = _read_numa_free_bytes()
            logger.debug(
                "    d(copy) : rss=%+.2fGiB consumed[%s]",
                (rss1 - rss0) / _GIB,
                _fmt_node_deltas(free0, free1),
            )

            if self.pin_memory:
                cpu_data = cpu_data.pin_memory()
            rss2 = _read_rss_bytes()
            free2 = _read_numa_free_bytes()
            logger.debug(
                "    d(pin)  : rss=%+.2fGiB consumed[%s]",
                (rss2 - rss1) / _GIB,
                _fmt_node_deltas(free1, free2),
            )

            if not self.uva_offloading:
                p.data = cpu_data
            else:
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                p._vllm_is_uva_offloaded = True
            rss3 = _read_rss_bytes()
            free3 = _read_numa_free_bytes()
            logger.debug(
                "    d(view) : rss=%+.2fGiB consumed[%s]",
                (rss3 - rss2) / _GIB,
                _fmt_node_deltas(free2, free3),
            )

            logger.debug(
                "    d(total): rss=%+.2fGiB consumed[%s]",
                (rss3 - rss0) / _GIB,
                _fmt_node_deltas(free0, free3),
            )
            logger.debug(
                "    self_numa (this worker mapped)=[%s]",
                _fmt_node_abs(_read_self_numa_bytes()),
            )
            logger.debug(
                "    numa_free (system total)=[%s]",
                _fmt_node_abs(free3),
            )

            offloaded_bytes = p.data.numel() * p.data.element_size()
            self.cpu_offload_bytes += offloaded_bytes
            self.offloaded_param_info.append(
                (
                    f"{module_tag}.{name}",
                    offloaded_bytes,
                    p.data.shape,
                    p.data.dtype,
                    p.data.element_size(),
                )
            )
            offloaded_parameters = True

        if offloaded_parameters and not self.uva_offloading:
            original_forward = module.forward

            def forward(*args, **kwargs):
                module.forward = original_forward
                device_state = {
                    # here we blindly call `to(device)`
                    # if the parameter is already on the device,
                    # it will be a no-op
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }

                # set `tie_weights=False` as tied weights in original model
                # become untied when calling .to(device) individually
                output = functional_call(
                    module,
                    device_state,
                    args=args,
                    kwargs=kwargs,
                    tie_weights=False,
                )
                module.forward = forward
                return output

            module.forward = forward

        return module
