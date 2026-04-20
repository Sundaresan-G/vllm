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

#include <cstddef>
#include <cstdint>
#include <stdexcept>

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

    void pin_existing(uintptr_t ptr, size_t size, const uintptr_t syclQueueAddr) {
        if (ptr == 0) {
            throw std::invalid_argument("ptr must be non-zero");
        }
        if (size == 0) {
            throw std::invalid_argument("size must be > 0");
        }

        void* raw_ptr = reinterpret_cast<void*>(ptr);
        const sycl::queue* syclQueuePtr = reinterpret_cast<const sycl::queue*>(syclQueueAddr);
        if (xpu_pin_host_memory(raw_ptr, size, syclQueuePtr) != 0) {
            throw std::runtime_error("xpu_pin_host_memory failed");
        }
    }

    void unpin_existing(uintptr_t ptr) {
        if (ptr == 0) {
            return;
        }

        void* raw_ptr = reinterpret_cast<void*>(ptr);
        if (xpu_unpin_host_memory(raw_ptr) != 0) {
            throw std::runtime_error("xpu_unpin_host_memory failed");
        }
    }

} // namespace

namespace py = pybind11;

PYBIND11_MODULE(xpu_pin_memory_ext, m) {
    m.doc() = "pybind wrapper for xpu_pin_memory.hpp";

    // Pin/unpin functions that accept sycl::queue address
    m.def("pin_existing", &pin_existing,
          py::arg("ptr_addr"), py::arg("size"), py::arg("queue_addr"),
          "Pin existing page-aligned host memory by address and queue address");

    m.def("unpin_existing", &unpin_existing,
          py::arg("ptr_addr"),
          "Unpin previously pinned memory by address");
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