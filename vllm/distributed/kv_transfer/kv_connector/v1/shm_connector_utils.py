import torch
from torch.utils.cpp_extension import load_inline

source = """
#ifndef XPU_PIN_MEMORY_HPP
#define XPU_PIN_MEMORY_HPP

// This header provides utilities for pinning host memory for XPU (Intel GPU) devices
// using SYCL and Level Zero APIs for efficient memory transfer operations
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <iostream>
#include <map>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <functional>

#if defined(_MSVC_LANG)
#define XPU_CPP_STANDARD _MSVC_LANG
#else
#define XPU_CPP_STANDARD __cplusplus
#endif

#define XPU_ENSURE_CPP17_OR_LATER() \
    static_assert(XPU_CPP_STANDARD >= 201703L, "xpu_pin_memory.hpp requires C++17 or later")

XPU_ENSURE_CPP17_OR_LATER();

#define XPU_LOG_ERROR(message)                                                         \
    (std::cerr << __FILE__ << ":" << __LINE__ << " in " << __func__ << ": " << message \
               << std::endl)

// c++17 and beyond ensures inline has external linkage for variables, so we can define this in a header without violating the one definition rule
inline std::map<uintptr_t, ze_context_handle_t> ptrToZeContextMap;

inline int xpu_pin_host_memory(void *ptr, size_t size, const sycl::queue* syclQueuePtr) {
    // if (ptr == nullptr) {
    //     XPU_LOG_ERROR("Error: Invalid pointer provided.");
    //     return -1; // Invalid pointer
    // }
    auto* aligned_ptr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ptr) & ~(4096 - 1));
    if (ptrToZeContextMap.find(reinterpret_cast<uintptr_t>(ptr)) != ptrToZeContextMap.end()) {
        XPU_LOG_ERROR("Warning: Pointer is already pinned.");
        return 0; // Pointer is already pinned
    }
    size += reinterpret_cast<uintptr_t>(ptr) & (4096 - 1); // Adjust size to account for alignment offset

    auto syclContext = syclQueuePtr->get_context();

    ze_result_t status;
    ze_context_handle_t zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(syclContext);

    ze_external_memmap_sysmem_ext_desc_t memmap_desc{ZE_STRUCTURE_TYPE_EXTERNAL_MEMMAP_SYSMEM_EXT_DESC, nullptr, aligned_ptr, size};
    ze_host_mem_alloc_desc_t host_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, &memmap_desc, 0};

    status = zeMemAllocHost(zeContext, &host_desc, size, 0, &aligned_ptr);

    if (status != ZE_RESULT_SUCCESS) {
        XPU_LOG_ERROR("Error: zeMemAllocHost failed with status " << status);
        return -1; // Memory allocation failed
    }
    if (aligned_ptr != reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ptr) & ~(4096 - 1))) {
        XPU_LOG_ERROR("Error: zeMemAllocHost did not return the expected aligned pointer.");
        return -1; // Memory allocation failed
    }

    ptrToZeContextMap.emplace(reinterpret_cast<uintptr_t>(ptr), zeContext);

    return 0;
}

inline int xpu_unpin_host_memory(void *ptr) {
    // if (ptr == nullptr) {
    //     XPU_LOG_ERROR("Warning: Null pointer provided for freeing.");
    //     return 0; // Invalid pointer
    // }
    auto it = ptrToZeContextMap.find(reinterpret_cast<uintptr_t>(ptr));
    if (it == ptrToZeContextMap.end()) {
        XPU_LOG_ERROR("Warning: Pointer is not pinned.");
        return 0; // Pointer is not pinned
    }

    ze_context_handle_t zeContext = it->second;

    ze_result_t status = zeMemFree(zeContext, reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ptr) & ~(4096 - 1)));

    if (status != ZE_RESULT_SUCCESS) {
        XPU_LOG_ERROR("Error: zeMemFree failed with status " << status);
        return -1; // Memory free failed
    }

    ptrToZeContextMap.erase(it);

    return 0;
}

#endif // XPU_PIN_MEMORY_HPP

// Start of CPP file for pybind11 bindings

namespace {

    // Pin the memory backing `source`, then return a new tensor that shares the
    // same storage.  When the returned tensor's storage is released, the custom
    // deleter (1) unpins the memory, then (2) drops the captured reference to
    // `source`, allowing PyTorch to free the original storage if no other
    // references remain.
    torch::Tensor pin_and_wrap(torch::Tensor source, uintptr_t syclQueueAddr) {
        if (!source.device().is_cpu()) {
            throw std::invalid_argument("source tensor must be a CPU tensor");
        }

        // Use the storage base pointer and full storage size so that stride gaps
        // in non-contiguous tensors are also pinned, preventing partial pinning.
        void* ptr   = const_cast<void*>(source.storage().data());
        size_t size = static_cast<size_t>(source.storage().nbytes());

        if (ptr == nullptr) {
            throw std::invalid_argument("source tensor has a null data pointer");
        }
        if (size == 0) {
            throw std::invalid_argument("source tensor has zero bytes");
        }
        if (syclQueueAddr == 0) {
            throw std::invalid_argument("syclQueueAddr must be non-zero");
        }

        const sycl::queue* syclQueuePtr =
            reinterpret_cast<const sycl::queue*>(syclQueueAddr);

        if (xpu_pin_host_memory(ptr, size, syclQueuePtr) != 0) {
            throw std::runtime_error("xpu_pin_host_memory failed");
        }

        // Capture `source` by value to keep the storage alive for at least as
        // long as the returned tensor is alive.  Unpin before releasing the
        // reference so the pointer is still valid during zeMemFree.
        return torch::from_blob(
            source.data_ptr(),
            source.sizes(),
            source.strides(),
            [source](void* p) {
                // Unpin the storage base (not the tensor's data_ptr which may
                // be offset into the storage for non-contiguous tensors).
                xpu_unpin_host_memory(const_cast<void*>(source.storage().data()));
                // `source` goes out of scope here; if this was the last
                // reference its storage is freed by PyTorch normally.
            },
            source.options()
        );
    }

} // namespace

namespace py = pybind11;

PYBIND11_MODULE(xpu_pin_memory_ext, m) {
    m.doc() = "pybind wrapper for xpu_pin_memory.hpp";

    m.def("pin_and_wrap", &pin_and_wrap,
          py::arg("source"), py::arg("queue_addr"),
          "Pin a contiguous CPU tensor and return a new tensor backed by the "
          "same memory; the returned tensor's destructor unpins the memory and "
          "releases the reference to the original tensor");
}

// End of CPP file for pybind11 bindings
"""

xpu_pin_memory_ext = None

def load_xpu_pin_memory_extension():
    global xpu_pin_memory_ext
    if xpu_pin_memory_ext is None and hasattr(torch, "xpu") and torch.xpu.is_available():
        xpu_pin_memory_ext = load_inline(
            name="xpu_pin_memory_ext",
            cpp_sources=[],
            sycl_sources=source,
            extra_sycl_cflags=["-Wall", "-O3", "-std=c++17"],
            with_sycl=True,
            extra_ldflags=["-lze_loader"],
            verbose=True,
            keep_intermediates=False,
        )
    return xpu_pin_memory_ext