# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import queue
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from multiprocessing import shared_memory

from vllm import envs
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.selector import get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import TpKVTopology
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.block_table import BlockTable

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

Transfer = tuple[int, float]  # (xfer_handle, start_time)
EngineId = str
ReqId = str

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    # To be used when logical block size does not match the kernel block size
    local_physical_block_ids: list[int]
    remote_block_ids: list[int]
    remote_filename: str
    remote_engine_id: str
    tp_size: int
    remote_kv_cache_shape: list[int]
    remote_kv_cache_stride: list[int]
    remote_kv_cache_layout: str
    remote_block_size: int


class ShmConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}
        self.reqs_in_batch: set[ReqId] = set()
        self.reqs_not_processed: set[ReqId] = set()

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
        save_to_host: bool = False,
    ):
        # save and load are mutually exclusive
        assert load_remote_cache ^ save_to_host
        _req = ReqMeta(
            local_block_ids=local_block_ids,
            local_physical_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_filename=kv_transfer_params["remote_filename"],
            # P workers don't need to receive tp_size from proxy here.
            tp_size=kv_transfer_params.get("tp_size", 1),
            remote_kv_cache_shape=kv_transfer_params.get("remote_kv_cache_shape"),
            remote_kv_cache_stride=kv_transfer_params.get("remote_kv_cache_stride"),
            remote_kv_cache_layout=kv_transfer_params.get("remote_kv_cache_layout"),
            remote_block_size=kv_transfer_params.get("remote_block_size"),
        )
        if load_remote_cache:
            self.reqs_to_recv[request_id] = _req

# TODO: make it per-instance
_send_ReqId2BlockIds: dict[ReqId, list[int]] = {}
global_kv_cache_shape: list[int] = []
global_kv_cache_stride: list[int] = []

class ShmConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        self._send_ReqId2BlockIds: dict[ReqId, list[int]] = _send_ReqId2BlockIds

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: Optional[ShmConnectorScheduler] = (
                ShmConnectorScheduler(vllm_config, self.engine_id, self._send_ReqId2BlockIds)
            )
            self.connector_worker: Optional[ShmConnectorWorker] = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = ShmConnectorWorker(vllm_config, self.engine_id, self._send_ReqId2BlockIds)

    ############################################################
    # Class Methods
    ############################################################
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        if vllm_config.model_config is None:
            logger.warning_once(
                "Unable to detect current VLLM config. "
                "Fallback to default kv cache layout."
            )
            return None
        use_mla = vllm_config.model_config.use_mla
        if use_mla:
            # return None when we have mla
            # as the layout should not matter in that case,
            # which fallback to the default behavior.
            return None
        logger.info_once(
            "ShmConnector setting KV cache layout to HND for better xfer performance."
        )
        return "HND"

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Get block IDs that failed to load via NIXL."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()
    
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, ShmConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """ShmConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        """ShmConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, ShmConnectorMetadata)
        self.connector_worker.wait_for_save(self._connector_metadata)

    def shutdown(self):
        if self.connector_worker is not None:
            self.connector_worker.shutdown()


class ShmConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str, reqId2BlockIds: dict[ReqId, list[int]]):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.side_channel_filename = engine_id
        self._send_ReqId2BlockIds = reqId2BlockIds
        logger.info("Initializing SHM Scheduler %s", engine_id)

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        # Reqs to send and their expiration time
        self._reqs_need_send: dict[ReqId, float] = {}
        self._reqs_in_batch: set[ReqId] = set()
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[ReqId] = set()

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "SHMConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            token_ids = request.prompt_token_ids or []
            count = len(token_ids) - num_computed_tokens
            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "SHMConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_decode"):
            self._reqs_in_batch.add(request.request_id)
            local_block_ids = blocks.get_unhashed_block_ids()
            self._send_ReqId2BlockIds[request.request_id] = local_block_ids
        if params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                if all(
                    p in params
                    for p in ("remote_engine_id", "remote_filename")
                ):
                    # If remote_blocks and num_external_tokens = 0, we have
                    # a full prefix cache hit on the D worker. We need to call
                    # send_notif in _read_blocks to free the memory on the P.
                    local_block_ids = (
                        blocks.get_unhashed_block_ids()
                        if num_external_tokens > 0
                        else []
                    )
                    # Get unhashed blocks to pull from remote.
                    self._reqs_need_recv[request.request_id] = (
                        request,
                        local_block_ids,
                    )

                else:
                    logger.warning(
                        "Got invalid KVTransferParams: %s. This "
                        "request will not utilize KVTransfer",
                        params,
                    )
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = ShmConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                load_remote_cache=True,
                save_to_host=False,
            )

        meta.reqs_to_send = self._reqs_need_send
        meta.reqs_in_batch = self._reqs_in_batch
        meta.reqs_not_processed = self._reqs_not_processed

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        self._reqs_in_batch = set()
        self._reqs_not_processed = set()
        self._reqs_need_send = {}

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        from vllm.v1.request import RequestStatus

        params = request.kv_transfer_params
        logger.debug(
            "SHMConnector request_finished(%s), request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params:
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if not params.get("do_remote_decode"):
            return False, None
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(request.request_id)
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = len(block_ids) > 0

        if delay_free_blocks:
            # Prefill request on remote. It will be read from D upon completion
            logger.debug(
                "SHMConnector request_finished(%s) waiting for %d seconds "
                "for remote decode to fetch blocks",
                request.request_id,
                envs.VLLM_SHM_ABORT_REQUEST_TIMEOUT,
            )
            self._reqs_need_send[request.request_id] = (
                time.perf_counter() + envs.VLLM_SHM_ABORT_REQUEST_TIMEOUT
            )

        # self._send_ReqId2BlockIds[request.request_id] = block_ids
        # logger.debug(f"self._send_ReqId2BlockIds = {self._send_ReqId2BlockIds}")
        global global_kv_cache_shape, global_kv_cache_stride

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_filename=self.side_channel_filename,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            remote_kv_cache_shape=global_kv_cache_shape,
            remote_kv_cache_stride=global_kv_cache_stride,
            remote_kv_cache_layout=get_kv_cache_layout(),
            remote_block_size=self.block_size,
        )


class ShmConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str, reqId2BlockIds: dict[ReqId, list[int]]):
        logger.info("Initializing SHM wrapper")
        logger.info("Initializing SHM worker %s", engine_id)

        # Config.
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        
        self._send_ReqId2BlockIds = reqId2BlockIds
        self._pending_send_reqs: set[ReqId] = set()

        self.side_channel_filename = engine_id
        # Metadata.
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.num_blocks = 0
        self.enable_permute_local_kv = False

        # KV Caches and nixl tracking data.
        self.device_type = current_platform.device_type
        self.kv_buffer_device: str = vllm_config.kv_transfer_config.kv_buffer_device
        self.device_kv_caches: dict[str, torch.Tensor] = {}

        # Map of engine_id -> kv_caches_base_addr. For TP case, each local
        # rank will still only pull from a single remote TP worker.
        self.kv_caches_base_addr: dict[EngineId, list[int]] = {}
        self.device_id: int = 0

        # Number of SHM regions. Currently one region per cache
        # (so 1 per layer for MLA, otherwise 2 per layer)
        self.num_regions = 0
        self.num_layers = 0

        # Map of engine_id -> num_blocks. All ranks in the same deployment will
        # have the same number of blocks.
        self.dst_num_blocks: dict[EngineId, int] = {}

        # In progress transfers.
        # [req_id -> list[handle]]
        self._recving_metadata: dict[ReqId, ReqMeta] = {}
        self._recving_transfers = defaultdict[ReqId, list[Transfer]](list)
        # Track the expiration time of requests that are waiting to be sent.
        self._reqs_to_send: dict[ReqId, float] = {}
        # Set of requests that have been part of a batch, regardless of status.
        self._reqs_to_process: set[ReqId] = set()

        # invalid blocks from failed NIXL operations
        self._invalid_block_ids: set[int] = set()
        # requests that skipped transfer (handshake or transfer failures)
        self._failed_recv_reqs: set[ReqId] = set()
        self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        # TODO(mgoin): remove this once we have hybrid memory allocator
        # Optimization for models with local attention (Llama 4)
        # List of block window sizes for each layer for local attention
        self.block_window_per_layer: list[int | None] = []
        self.use_mla = self.model_config.use_mla

        backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.cache_config.cache_dtype,
            self.block_size,
            use_mla=self.use_mla,
        )
        self.backend_name = backend.get_name()
        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.world_size}
        self._block_size: dict[EngineId, int] = {self.engine_id: self.block_size}
        # With heterogeneous TP, P must wait for all assigned D TP workers to
        # finish reading before safely freeing the blocks.
        self.consumer_notification_counts_by_req = defaultdict[ReqId, int](int)

        self.kv_topo = TpKVTopology(
            tp_rank=self.tp_rank,
            engine_id=self.engine_id,
            remote_tp_size=self._tp_size,  # shared state
            remote_block_size=self._block_size,  # shared state
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backend=backend,
        )
        self._use_pallas = self.kv_topo._use_pallas
        self._physical_blocks_per_logical_kv_block = 1
        
        self.shm_kv_caches: dict[str, shared_memory.SharedMemory] = {}
        # Flags to indicate whether a layer's shared memory is free to use.
        # True means busy, False means free to use
        self.shm_kv_caches_completion_flags: Optional[shared_memory.SharedMemory] = None
        self.remote_shm_objects: dict[EngineId, dict[str, shared_memory.SharedMemory]] = {}
        self.remote_cached_kv_caches: dict[EngineId, dict[str, torch.Tensor]] = {}
        self.remote_cached_shm_kv_caches_completion_flags: dict[EngineId, shared_memory.SharedMemory] = {}

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in nixl."""

        logger.info(
            "Registering KV_Caches. use_mla: %s",
            self.use_mla,
        )

        caches_data = []
        # With hybrid allocator, layers can share a kv cache tensor
        seen_base_addresses = []

        # Note(tms): I modified this from the original region setup code.
        # K and V are now in different regions. Advantage is that we can
        # elegantly support MLA and any cases where the K and V tensors
        # are non-contiguous (it's not locally guaranteed that they will be)
        # Disadvantage is that the encoded ShmAgentMetadata is now larger
        # (roughly 8KB vs 5KB).
        # Conversely for FlashInfer, K and V are registered in the same region
        # to better exploit the memory layout (ie num_blocks is the first dim).
        split_k_and_v = self.kv_topo.split_k_and_v
        tensor_size_bytes = None

        # TODO (NickLucche): Get kernel_block_size in a cleaner way
        # NHD default "view" for non-MLA cache
        if self.device_type == "cpu":
            block_size_position = -2
        else:
            block_size_position = -2 if self.use_mla else -3

        # logger.debug(f"block_size_position = {block_size_position}")

        # Enable different block lengths for different layers when MLA is used.
        self.block_len_per_layer = list[int]()
        self.slot_size_per_layer = list[int]()  # HD bytes in kv terms
        self.device_id = self.tp_rank
        for layer_name, cache_or_caches in kv_caches.items():
            logger.debug("Registering layer %s with cache shape %s and stride %s", layer_name, cache_or_caches.shape, cache_or_caches.stride())
            assert cache_or_caches.device.type == "cpu", (
                f"KV cache tensor {layer_name} must be on CPU for SHM xfer, "
                f"but got {cache_or_caches.device}"
            )

            # Capture all tensor properties before processing
            original_cache_size = cache_or_caches.numel() * cache_or_caches.element_size()
            original_cache_dtype = cache_or_caches.dtype
            original_cache_shape = cache_or_caches.shape
            original_cache_stride = cache_or_caches.stride()
            original_cache_requires_grad = cache_or_caches.requires_grad
            original_cache_is_contiguous = cache_or_caches.is_contiguous()

            # TODO: improve it
            global global_kv_cache_shape, global_kv_cache_stride
            global_kv_cache_shape = list(original_cache_shape)
            global_kv_cache_stride = list(original_cache_stride)

            cache_or_caches.data = torch.empty(0)  # release tensor from PyTorch management

            # First try to clean up any existing shared memory with the same name
            shm_name = f"{self.side_channel_filename}_{layer_name}"
            try:
                existing_shm = shared_memory.SharedMemory(name=shm_name)
                existing_shm.close()
                existing_shm.unlink()
                logger.debug(f"Deleted existing shared memory for {layer_name}")
            except FileNotFoundError:
                # No existing shared memory, which is fine
                logger.debug(f"No existing shared memory found for {layer_name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup existing shared memory {shm_name}: {e}")

            # Now create new shared memory
            self.shm_kv_caches[layer_name] = shared_memory.SharedMemory(
                name=shm_name, 
                create=True, 
                size=original_cache_size
            )
            logger.debug(f"Created new shared memory for {layer_name} with name {shm_name} and size {original_cache_size} bytes")

            # Recreate the tensor from shared memory
            # Convert shared memory buffer to numpy array with correct dtype
            np_array = np.frombuffer(
                buffer=self.shm_kv_caches[layer_name].buf,
                dtype=np.uint8,  # 1 byte per element
                count=original_cache_size  # Total size in bytes
            )
            
            # Create PyTorch tensor from numpy array (shares memory)
            cache_or_caches.data = torch.from_numpy(np_array).view(dtype=original_cache_dtype).reshape(original_cache_shape)
            
            # Restore tensor properties if the original had specific stride/layout
            if not original_cache_is_contiguous:
                # Use as_strided to restore the exact memory layout
                cache_or_caches.data = torch.as_strided(
                    cache_or_caches,
                    size=original_cache_shape,
                    stride=original_cache_stride,
                    storage_offset=0  # Assuming shared memory starts at offset 0
                )
            
            # Restore requires_grad property
            cache_or_caches.requires_grad_(original_cache_requires_grad)
            
            # Verify the recreated tensor has the same properties
            assert cache_or_caches.shape == original_cache_shape, f"Shape mismatch: {cache_or_caches.shape} vs {original_cache_shape}"
            assert cache_or_caches.dtype == original_cache_dtype, f"Dtype mismatch: {cache_or_caches.dtype} vs {original_cache_dtype}"
            assert cache_or_caches.stride() == original_cache_stride, f"Stride mismatch: {cache_or_caches.stride()} vs {original_cache_stride}"
            assert cache_or_caches.requires_grad == original_cache_requires_grad, f"Requires_grad mismatch"
            
            logger.debug(f"Successfully recreated tensor from shared memory for {layer_name} with shape {cache_or_caches.shape}")

            # Pin the memory
            if self.device_type == "cuda":
                torch.cuda.cudart().cudaHostRegister(cache_or_caches.data_ptr(), original_cache_size, 0)

            # cache_or_caches[0, 0, 0, 0, :4] = torch.tensor([1, 2, 3, 4], dtype=cache_or_caches.dtype)

            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]

            for cache in cache_list:
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue

                kernel_block_size = cache.shape[block_size_position]

                if self.block_size != kernel_block_size:
                    logger.info_once(
                        "User-specified logical block size (%s) does not match"
                        " physical kernel block size (%s). Using the latter. ",
                        self.block_size,
                        kernel_block_size,
                    )
                    self._physical_blocks_per_logical_kv_block = (
                        self.block_size // kernel_block_size
                    )
                    self.block_size = kernel_block_size
                    self._block_size[self.engine_id] = kernel_block_size

                seen_base_addresses.append(base_addr)
                curr_tensor_size_bytes = cache.numel() * cache.element_size()

                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_tensor_size_bytes
                    self.num_blocks = cache.shape[0]

                assert cache.shape[0] == self.num_blocks, (
                    "All kv cache tensors must have the same number of blocks"
                )

                self.block_len_per_layer.append(
                    curr_tensor_size_bytes // self.num_blocks
                )
                self.slot_size_per_layer.append(
                    self.block_len_per_layer[-1] // self.block_size
                )

                if not self.use_mla:
                    # Different kv cache shape is not supported by HeteroTP
                    assert tensor_size_bytes == curr_tensor_size_bytes, (
                        "All kv cache tensors must have the same size"
                    )
                # Need to make sure the device ID is non-negative for NIXL,
                # Torch uses -1 to indicate CPU tensors.
                self.device_id = max(cache.get_device(), 0)
                caches_data.append(
                    (base_addr, curr_tensor_size_bytes, self.device_id, "")
                )

        shm_name = f"{self.side_channel_filename}_flags"
        try:
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            existing_shm.close()
            existing_shm.unlink()
            logger.debug(f"Deleted existing shared memory for completion flags of {layer_name}")
        except FileNotFoundError:
            # No existing shared memory, which is fine
            logger.debug(f"No existing shared memory found for completion flags of {layer_name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup existing shared memory {shm_name}: {e}")

        # Now create new shared memory for completion flags
        self.shm_kv_caches_completion_flags = shared_memory.SharedMemory(
            name=shm_name, 
            create=True, 
            size=self.num_blocks + 1  # single byte for boolean flag
        )

        logger.debug(f"Created new shared memory for completion flags with name {shm_name} and size {self.num_blocks + 1} bytes")

        # Initialize the completion flag to False (free)
        for i in range(self.num_blocks):
            self.shm_kv_caches_completion_flags.buf[i] = False  # False

        logger.debug(
            "Different block lengths collected: %s", set(self.block_len_per_layer)
        )
        assert len(self.block_len_per_layer) == len(seen_base_addresses)
        assert self.num_blocks != 0

        self.kv_caches_base_addr[self.engine_id] = seen_base_addresses
        self.num_regions = len(caches_data)
        self.num_layers = len(kv_caches.keys())

        self.device_kv_caches = kv_caches
        self.dst_num_blocks[self.engine_id] = self.num_blocks
        if self.kv_topo.is_kv_layout_blocks_first:
            for i in range(len(self.slot_size_per_layer)):
                assert self.slot_size_per_layer[i] % 2 == 0
                self.slot_size_per_layer[i] //= 2

            # NOTE (NickLucche) When FlashInfer is used, memory is registered
            # with joint KV for each block. This minimizes the overhead in
            # registerMem allowing faster descs queries. In order to be able to
            # split on kv_heads dim as required by heterogeneous TP, one must
            # be able to index K/V separately. Hence we double the number
            # of 'virtual' regions here and halve `block_len` below.
            self.num_regions *= 2

        # Register local/src descr for SHM xfer.
        # blocks_data = []
        # for i, base_addr in enumerate(seen_base_addresses):
        #     kv_block_len = self.get_backend_aware_kv_block_len(layer_idx=i)
        #     # NOTE With heter-TP, more blocks are prepared than what are
        #     # needed as self.num_blocks >= nixl_agent_meta.num_blocks. We
        #     # could create fewer, but then _get_block_descs_ids needs to
        #     # select agent_meta.num_blocks instead of self.num_blocks for
        #     # local descr, and that makes handling regular flow less clean.
        #     for block_id in range(self.num_blocks):
        #         block_offset = block_id * self.block_len_per_layer[i]
        #         addr = base_addr + block_offset
        #         # (addr, len, device id)
        #         blocks_data.append((addr, kv_block_len, self.tp_rank))

        #     if self._use_flashinfer:
        #         # Separate and interleave K/V regions to maintain the same
        #         # descs ordering. This is needed for selecting contiguous heads
        #         # when split across TP ranks.
        #         for block_id in range(self.num_blocks):
        #             block_offset = block_id * self.block_len_per_layer[i]
        #             addr = base_addr + block_offset
        #             # Register addresses for V cache (K registered first).
        #             v_addr = addr + kv_block_len
        #             blocks_data.append((v_addr, kv_block_len, self.tp_rank))
        # logger.debug(
        #     "Created %s blocks for src engine %s and rank %s",
        #     len(blocks_data),
        #     self.engine_id,
        #     self.tp_rank,
        # )

        # TODO(mgoin): Hybrid memory allocator is currently disabled for
        # models with local attention (Llama 4). Can remove this once enabled.
        if self.model_config.hf_config.model_type == "llama4":
            from transformers import Llama4TextConfig

            assert isinstance(self.model_config.hf_text_config, Llama4TextConfig)
            llama4_config = self.model_config.hf_text_config
            no_rope_layers = llama4_config.no_rope_layers
            chunk_size = llama4_config.attention_chunk_size
            chunk_block_size = math.ceil(chunk_size / self.block_size)
            for layer_idx in range(self.num_layers):
                # no_rope_layers[layer_idx] == 0 means NoPE (global)
                # Any other value means RoPE (local chunked)
                is_local_attention = no_rope_layers[layer_idx] != 0
                block_window = chunk_block_size if is_local_attention else None
                self.block_window_per_layer.append(block_window)
            logger.debug(
                "Llama 4 block window per layer mapping: %s",
                self.block_window_per_layer,
            )
            assert len(self.block_window_per_layer) == self.num_layers

    def permute_device_kv(self, block_ids: list[int]):
        """Transforms the layout of received KV cache blocks to the local format.

        This method corrects layout mismatches from direct memory copies by
        permuting the tensor dimensions.

        - **Source Layout:** `[num_blocks, n_kv_head, block_size, head_dim]`
        - **Target Layout:** `[num_blocks, block_size, n_kv_head, head_dim]`

        Args:
            block_ids: A list of block IDs to update and permute.

        Implementation:
        - x = blocks_to_update.reshape(src_shape) # view local kv with sender layout
        - permuted_blocks = x.permute(*inv_order) # transpose n_kv_heads, block_size
        - cache.index_copy_(0, indices, permuted_blocks) # copy permuted kv back

        """
        split_k_and_v = self.kv_topo.split_k_and_v
        inv_order = [0, 2, 1, 3]
        sample_cache = list(self.device_kv_caches.values())[0][0]
        target_shape = list(sample_cache.shape)
        target_shape[0] = -1
        src_shape = tuple(target_shape[i] for i in inv_order)
        indices = torch.tensor(block_ids, device=sample_cache.device)

        for _, cache_or_caches in self.device_kv_caches.items():
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
            for cache in cache_list:
                blocks_to_update = cache.index_select(0, indices)
                permuted_blocks = blocks_to_update.reshape(src_shape).permute(
                    *inv_order
                )
                cache.index_copy_(0, indices, permuted_blocks)

    def blocksize_post_process(self, block_ids_per_ratio: dict[float, list[list[int]]]):
        def _process_local_gt_remote(blocks_to_update, block_size_ratio):
            n_kv_heads, block_size, head_size = blocks_to_update.shape[1:]
            remote_block_size = block_size // block_size_ratio
            n_blocks = block_size_ratio
            # actual permute is to convert
            # for local blocksize > remote blocksize
            # ex: local blocksize = 16 tokens, remote blocksize = 4 tokens
            # local block[0] = remote block[0, 1, 2, 3]
            # remote is |h0-b0|h1-b0|h2-b0|h3-b0|h0-b1|h1-b1|h2-b1|h3-b1|...
            # local is  |h0-b0..................|h1-b0..................|...
            # permute is to:
            # 1. view => view remote as n_blocks * remote_shape(H,remoteN,D)
            # 2. permute => (H, nblocks, remoteN, D)
            # 3. flatten => (H, localN, D)
            permuted_blocks = (
                blocks_to_update.reshape(
                    -1, n_blocks, n_kv_heads, remote_block_size, head_size
                )
                .permute(0, 2, 1, 3, 4)
                .flatten(2, 3)
            )
            return permuted_blocks

        if len(self.device_kv_caches) == 0:
            return
        split_k_and_v = not (
            self.use_mla or self._use_pallas or self.kv_topo.is_kv_layout_blocks_first
        )
        sample_cache = list(self.device_kv_caches.values())[0][0]
        for block_size_ratio, block_ids_list in block_ids_per_ratio.items():
            assert block_size_ratio > 1, "Only nP < nD supported currently."
            block_ids_list = [[item for sublist in block_ids_list for item in sublist]]

            for block_ids in block_ids_list:
                indices = torch.tensor(block_ids, device=sample_cache.device)

                for _, cache_or_caches in self.device_kv_caches.items():
                    cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
                    for cache in cache_list:
                        blocks_to_update = cache.index_select(0, indices)
                        # because kv_cache is always using original layout NHD as
                        # virtual shape while stride can be either HND / NHD at
                        # initialization.
                        # we need to firstly get physical view of the tensor
                        permuted_blocks = _process_local_gt_remote(
                            blocks_to_update.permute(0, 2, 1, 3), block_size_ratio
                        ).permute(0, 2, 1, 3)
                        cache.index_copy_(0, indices, permuted_blocks)

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        done_sending = self._get_new_notifs()
        done_recving = self._pop_done_transfers(self._recving_transfers)

        # add requests that skipped transfer to done_recving
        done_recving.update(self._failed_recv_reqs)
        self._failed_recv_reqs.clear()

        if len(done_sending) > 0 or len(done_recving) > 0:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(done_sending),
                len(done_recving),
            )

        block_ids_to_permute = []
        block_ids_for_blocksize_post_process = defaultdict(list)
        for req_id in done_recving:
            # clean up metadata for completed requests
            meta = self._recving_metadata.pop(req_id, None)
            assert meta is not None, f"{req_id} not found in recving_metadata list"
            if self.enable_permute_local_kv:
                block_ids_to_permute += meta.local_physical_block_ids

            # post processing for heteroblocksize
            # TODO: fix this when heteroTP is supported
            # block_size_ratio = self.kv_topo.block_size_ratio_from_engine_id(
            #     meta.remote_engine_id
            # )
            block_size_ratio = 1
            if (
                not self.use_mla
                and block_size_ratio > 1
                and self.kv_cache_layout == "HND"
            ):
                block_ids_for_blocksize_post_process[block_size_ratio].append(
                    meta.local_block_ids
                )
        self.blocksize_post_process(block_ids_for_blocksize_post_process)
        if len(block_ids_to_permute) > 0:
            self.permute_device_kv(block_ids_to_permute)

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()
        while self._reqs_to_send:
            req_id, expires = next(iter(self._reqs_to_send.items()))
            # Sorted dict, oldest requests are put first so we can exit early.
            if now < expires:
                break
            count = self.consumer_notification_counts_by_req.pop(req_id, 0)
            logger.warning(
                "Releasing expired KV blocks for request %s which were "
                "retrieved by %d decode worker(s) within %d seconds.",
                req_id,
                count,
                envs.VLLM_SHM_ABORT_REQUEST_TIMEOUT,
            )
            self._reqs_to_process.remove(req_id)
            del self._reqs_to_send[req_id]
            del self._send_ReqId2BlockIds[req_id]
            self._pending_send_reqs.discard(req_id)
            done_sending.add(req_id)

        return done_sending, done_recving

    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP scenario), wait
        for all consumers to be done pulling.
        """
        notified_req_ids: set[str] = set()
        logger.debug(f"self._send_ReqId2BlockIds = {self._send_ReqId2BlockIds}")
        logger.debug(f"self._pending_send_reqs = {self._pending_send_reqs}")
        for req_id in list(self._reqs_to_send.keys()):
            assert req_id in self._send_ReqId2BlockIds
            block_id = self._send_ReqId2BlockIds[req_id][-1]  # Check the last block's flag
            logger.debug(f"self.shm_kv_caches_completion_flags.buf[{block_id}] = {self.shm_kv_caches_completion_flags.buf[block_id]}")
            if self.shm_kv_caches_completion_flags.buf[block_id] == False:  # False means free
                self._send_ReqId2BlockIds.pop(req_id, None)
                self._pending_send_reqs.discard(req_id)
                # TODO: support heterogeneous TP
                tp_ratio = 1
                self.consumer_notification_counts_by_req[req_id] += 1
                # Wait all consumers (D) to be done reading before freeing.
                if self.consumer_notification_counts_by_req[req_id] == int(tp_ratio):
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    self._reqs_to_process.remove(req_id)
                    self._reqs_to_send.pop(req_id, None)
        return notified_req_ids

    def _pop_done_transfers(self, transfers: dict[str, list[int]]) -> set[str]:
        """
        Pop completed xfers by checking for DONE state.
        Args:
            transfers: dict of req_id -> list[running_xfer]
        Returns:
            set of req_ids that have all done xfers
        """
        done_req_ids: set[str] = set()
        for req_id, _ in list(transfers.items()):
            in_progress = False
            if not in_progress:
                done_req_ids.add(req_id)
                del transfers[req_id]
        return done_req_ids

    def start_load_kv(self, metadata: ShmConnectorMetadata):
        """
        Start loading by triggering blocking xfer.
        We check for these trnxs to complete in each step().
        """
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids
            )
            meta.remote_block_ids = self._logical_to_kernel_block_ids(
                meta.remote_block_ids
            )
            remote_engine_id = meta.remote_engine_id
            logger.debug(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote_block_ids),
            )
            # always store metadata for failure recovery
            self._recving_metadata[req_id] = meta
            if remote_engine_id not in self.remote_cached_kv_caches:
                logger.debug(
                    "Remote engine id %s not seen before. Existing keys are %s",
                    remote_engine_id,
                    list(self.remote_cached_kv_caches.keys()),
                )
                self.remote_shm_objects[remote_engine_id] = {} 
                self.remote_cached_kv_caches[remote_engine_id] = {}
                assert remote_engine_id not in self.remote_cached_shm_kv_caches_completion_flags   
                try: 
                    self.remote_cached_shm_kv_caches_completion_flags[remote_engine_id] = shared_memory.SharedMemory(
                        name=f"{meta.remote_filename}_flags",
                        create=False,
                    )
                except FileNotFoundError:
                    logger.error(
                        "Shared memory for remote engine %s completion flags "
                        "not found. Make sure the remote process has created "
                        "the shared memory segment.",
                        remote_engine_id,
                    )
                    raise RuntimeError(
                        f"Failed to access shared memory for remote engine {remote_engine_id} "
                        f"completion flags. Ensure the remote process has created the "
                        f"shared memory segment and it is accessible."
                    ) 
                for layer_name, kv_cache in self.device_kv_caches.items():
                    try:
                        remote_shm_name = f"{meta.remote_filename}_{layer_name}"
                        logger.debug(
                            "Accessing shared memory for remote engine %s and layer %s with name %s",
                            remote_engine_id,
                            layer_name,
                            remote_shm_name,
                        )
                        remote_shm_buffer = shared_memory.SharedMemory(
                            name=remote_shm_name,
                            create=False,
                        )
                        # Store it to prevent cleanup
                        self.remote_shm_objects[remote_engine_id][layer_name] = remote_shm_buffer
                    except FileNotFoundError:
                        logger.error(
                            "Shared memory for remote engine %s and layer %s "
                            "not found. Make sure the remote process has created "
                            "the shared memory segment.",
                            remote_engine_id,
                            layer_name,
                        )
                        raise RuntimeError(
                            f"Failed to access shared memory for remote engine {remote_engine_id} "
                            f"and layer {layer_name}. Ensure the remote process has created the "
                            f"shared memory segment and it is accessible."
                        )
                    # Recreate the tensor from shared memory
                    np_array = np.frombuffer(
                        buffer=remote_shm_buffer.buf,
                        dtype=np.uint8,  # 1 byte per element
                    )
                    self.remote_cached_kv_caches[remote_engine_id][layer_name] = torch.from_numpy(np_array).view(
                        dtype=kv_cache.dtype
                        ).as_strided_(
                            size=meta.remote_kv_cache_shape,
                            stride=meta.remote_kv_cache_stride,
                        )

                    # For difference in the block size position between GPU and CPU layouts
                    self.remote_cached_kv_caches[remote_engine_id][layer_name].transpose_(-2, -3)

                    logger.debug(f"Recreated remote KV cache tensor for engine {remote_engine_id} and layer {layer_name} with shape {self.remote_cached_kv_caches[remote_engine_id][layer_name].shape} and stride {self.remote_cached_kv_caches[remote_engine_id][layer_name].stride()}")

                    # logger.debug(f"Remote cached KV cache[{layer_name}][0, 0, 0, 0, :4] = {self.remote_cached_kv_caches[remote_engine_id][layer_name][0, 0, 0, 0, :4]}")

            from vllm._custom_ops import cpu_attn_reshape_and_cache
            from vllm.v1.attention.backends.cpu_attn import _get_attn_isa

            assert self.remote_cached_shm_kv_caches_completion_flags[remote_engine_id].buf[meta.remote_block_ids[-1]] == True, (
                f"Remote block id {meta.remote_block_ids[-1]} for engine {remote_engine_id} is not marked busy as expected.")
            
            # Do not use the assert below. In the case of prefix caching, len(meta.remote_block_ids) can be greater than len(meta.local_block_ids)
            # assert len(meta.remote_block_ids) <= len(meta.local_block_ids), \
            #     (f"Number of remote block ids {len(meta.remote_block_ids)} should be less than or equal to number of local block ids {len(meta.local_block_ids)}"
            #      f"meta.remote_block_ids = {meta.remote_block_ids}, meta.local_block_ids = {meta.local_block_ids}")

            logger.debug(f"meta.local_block_ids = {meta.local_block_ids}")
            logger.debug(f"meta.remote_block_ids = {meta.remote_block_ids}")

            curr_local_block_ids = meta.local_block_ids[:len(meta.remote_block_ids)] if len(meta.remote_block_ids) <= len(meta.local_block_ids) else meta.local_block_ids
            # If there are more remote blocks than local blocks, we skip the initial remote blocks since likely they are from prefix caching
            curr_remote_block_ids = meta.remote_block_ids if len(meta.remote_block_ids) <= len(curr_local_block_ids) else meta.remote_block_ids[-len(curr_local_block_ids):]

            logger.debug(f"curr_local_block_ids = {curr_local_block_ids}")
            logger.debug(f"curr_remote_block_ids = {curr_remote_block_ids}")
            
            isa = _get_attn_isa(next(iter(self.device_kv_caches.values())).dtype, self.block_size)
            slot_mapping = torch.tensor([curr_local_block_ids[i] * self.block_size + offset for i in range(len(curr_remote_block_ids)) for offset in range(self.block_size)])

            # TODO: make it general for different block sizes
            assert self.block_size == meta.remote_block_size, (
                f"Block size mismatch: local block size {self.block_size} vs remote block size {meta.remote_block_size}"
            )
            
            copy_start_time = time.perf_counter()

            for layer_name, remote_kv_cache in self.remote_cached_kv_caches[remote_engine_id].items():
                # logger.debug(
                #     "Accessed remote KV cache for engine %s, layer %s with shape %s",
                #     remote_engine_id,
                #     layer_name,
                #     remote_kv_cache.shape,
                # )
                # logger.debug(f"self.device_kv_caches[{layer_name}] shape = {self.device_kv_caches[layer_name].shape} and stride = {self.device_kv_caches[layer_name].stride()}")
                # logger.debug(f"remote_kv_cache[{layer_name}] shape = {remote_kv_cache.shape} and stride = {remote_kv_cache.stride()}")

                if slot_mapping.numel() > 0:
                    keys, values = remote_kv_cache.unbind(0)
                    key = keys[meta.remote_block_ids].transpose(-2, -3)
                    key = key.reshape(-1, key.shape[-2], key.shape[-1])
                    value = values[meta.remote_block_ids].transpose(-2, -3)
                    value = value.reshape(-1, value.shape[-2], value.shape[-1])
                    cpu_attn_reshape_and_cache(
                        key,
                        value,
                        self.device_kv_caches[layer_name][0],
                        self.device_kv_caches[layer_name][1],
                        slot_mapping,
                        isa
                    )

                # keys, values = remote_kv_cache.unbind(0)
                # for i in range(len(curr_remote_block_ids)):
                #     remote_block_id = curr_remote_block_ids[i]
                #     key = keys[remote_block_id].transpose(0, 1)
                #     value = values[remote_block_id].transpose(0, 1)
                #     cpu_attn_reshape_and_cache(
                #         key,
                #         value,
                #         self.device_kv_caches[layer_name][0],
                #         self.device_kv_caches[layer_name][1],
                #         slot_mapping[i * self.block_size : (i + 1) * self.block_size],
                #         isa
                #     )

                
                # for i in range(len(meta.remote_block_ids)):
                #     remote_block_id = meta.remote_block_ids[i]
                #     local_block_id = meta.local_block_ids[i]
                #     # Copy data from remote shared memory to local device cache
                #     for j in range(2):  # For K and V
                #         # assert self.device_kv_caches[layer_name][j][local_block_id].is_contiguous()
                #         self.device_kv_caches[layer_name][j][local_block_id].copy_(
                #             remote_kv_cache[j][remote_block_id]
                #         )   
                # 

                    
                # logger.debug(f"Remote cache[{layer_name}] data_ptr() = {remote_kv_cache.data_ptr()}")
                # remote_k_cache = remote_kv_cache[0]
                # # for remote_block_id in meta.remote_block_ids[:1]:
                # #     logger.debug(f"{req_id}, {remote_block_id}, {remote_k_cache[remote_block_id][0][1][:20]}")
                # for slot in [0, self.block_size, 55, 56, 57, 58]:
                #     block_id = slot // self.block_size
                #     offset = slot % self.block_size
                #     logger.debug(f"Remote cache slot key_cache[{block_id} , 1, {offset}, :20] {slot}: {remote_k_cache[block_id, 1, offset, :20]}")


                # logger.debug(f"local cache[{layer_name}] data_ptr() = {self.device_kv_caches[layer_name].data_ptr()}")
                # local_k_cache = self.device_kv_caches[layer_name][0]
                # # for local_block_id in meta.local_block_ids[:1]:
                # #     logger.debug(f"{req_id}, {local_block_id}, {local_k_cache[local_block_id][0][1][:20]}")
                # for slot in [self.block_size, 55, 56, 57, 58]:
                #     block_id = slot // self.block_size
                #     offset = slot % self.block_size
                #     logger.debug(f"Local cache slot key_cache[{block_id} , 1, {offset}, :20] {slot}: {local_k_cache[block_id, 1, offset, :20]}") 

            copy_end_time = time.perf_counter()
            
            logger.debug(f"meta.remote_block_ids = {meta.remote_block_ids}")
            # Mark the last remote block as free
            self.remote_cached_shm_kv_caches_completion_flags[remote_engine_id].buf[meta.remote_block_ids[-1]] = False  # False means free


            logger.debug(
                "Completed copying KV blocks for request %s from remote engine %s in %.4f seconds.",
                req_id,
                remote_engine_id,
                copy_end_time - copy_start_time,
            )         

            self._recving_transfers[req_id].append((None, time.perf_counter()))

        # Start transfers for requests whose handshakes have now finished.
        # while not self._ready_requests.empty():
        #     self._read_blocks_for_req(*self._ready_requests.get_nowait())

        # Keep around the requests that have been part of a batch. This is
        # needed because async scheduling pushes the misalignment between the
        # moment in which requests expiration is set (P side) and the moment in
        # which blocks are read from D. As P can now more easily lag behind D
        # while processing the next batch, we make sure to only set an
        # expiration for requests that have not been read from D yet.
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)

        # Remove all requests that are not to be processed (eg aborted).
        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            # We should never get an abort after setting an expiry timer
            assert req_id not in self._reqs_to_send

        # Add to requests that are waiting to be read and track expiration.
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                self._reqs_to_send[req_id] = expiration_time

    def get_mapped_blocks(self, block_ids, block_size_ratio):
        """
          Calculates the new set of block IDs by mapping every element
          in the (potentially sparse) input array.
          Example: block_ids=[0, 2], block_size_ratio=2
        get_mapped_blocks    0     1     [2     3]     4     5
              # remote is |h0-b0|h1-b0||h0-b1|h1-b1||h0-b1|h1-b1||
              # local is  |h0-b0......||h1-b0......||h2-b0........
        local_block_ids         0           [1]           2
        """
        if block_ids.size == 0:
            return np.array([], dtype=np.int64)

        start_ids = block_ids * block_size_ratio
        offsets = np.arange(block_size_ratio)
        mapped_2d = start_ids[:, None] + offsets[None, :]

        return mapped_2d.flatten().astype(np.int64)

    def wait_for_save(self, metadata: ShmConnectorMetadata):
        # Get the most recently added item (dictionaries maintain insertion order)
        if self._send_ReqId2BlockIds:
            for req_id in self._send_ReqId2BlockIds.keys():
                if req_id not in self._pending_send_reqs:
                    self._pending_send_reqs.add(req_id)
                    i = self._send_ReqId2BlockIds[req_id][-1]
                    self.shm_kv_caches_completion_flags.buf[i] = True  # Mark as busy
                    logger.debug(f"self.shm_kv_caches_completion_flags.buf[{i}] = {self.shm_kv_caches_completion_flags.buf[i]}")
        logger.debug(f"self._send_ReqId2BlockIds = {self._send_ReqId2BlockIds}")
        # for reqId, blockIdList in self._send_ReqId2BlockIds.items():
        #     k_cache = list(self.device_kv_caches.values())[0][0]
        #     for blockId in blockIdList:
        #         logger.debug(f"{reqId}, {blockId}, {k_cache[blockId].permute(1,0,2).view(-1)}")

    def _logical_to_kernel_block_ids(self, block_ids: list[int]) -> list[int]:
        """
        Convert logical block ids to kernel physical block ids.
        This is required when the logical block size (the one set by the user)
        does not match the one required by the attn backend.
        """
        if self._physical_blocks_per_logical_kv_block == 1:
            # Noop when physical and logical block sizes are the same
            return block_ids
        block_ids_np = np.array(block_ids)
        block_arange = np.arange(0, self._physical_blocks_per_logical_kv_block).reshape(
            1, -1
        )
        return BlockTable.map_to_kernel_blocks(
            block_ids_np, self._physical_blocks_per_logical_kv_block, block_arange
        ).tolist()

    def get_backend_aware_kv_block_len(self, layer_idx: int):
        """
        Get the block length for one K/V element (K and V have the same size).

        For FA and other backends, this is equal to the length of the whole
        block, as K and V are in separate regions.
        For FlashInfer, this is half the length of the whole block, as K and V
        share the same region.
        """
        if self.kv_topo.is_kv_layout_blocks_first:
            # For indexing only half (either just the K or V part).
            block_len = self.block_len_per_layer[layer_idx] // 2
        else:
            block_len = self.block_len_per_layer[layer_idx]
        return block_len

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Return and clear the set of block IDs that failed to load.

        This is called by the scheduler to identify blocks that need
        to be retried after a NIXL transfer failure.
        """
        result = self._invalid_block_ids
        self._invalid_block_ids = set()
        return result

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Shutdown the connector worker."""
        # First clear tensor references to shared memory
        if hasattr(self, 'device_kv_caches'):
            for layer_name, tensor in self.device_kv_caches.items():
                try:
                    # Release tensor reference to shared memory
                    tensor.data = torch.empty(0)
                except Exception as e:
                    logger.warning(f"Error releasing tensor reference for {layer_name}: {e}")
            self.device_kv_caches.clear()

        # Clear remote cached tensors
        if hasattr(self, 'remote_cached_kv_caches'):
            for remote_engine_id, cache_dict in self.remote_cached_kv_caches.items():
                for layer_name, tensor in cache_dict.items():
                    try:
                        # Release tensor reference to remote shared memory
                        tensor.data = torch.empty(0)
                    except Exception as e:
                        logger.warning(f"Error releasing remote tensor reference for {layer_name}: {e}")
            self.remote_cached_kv_caches.clear()

        # Clean up remote shared memory references
        if hasattr(self, 'remote_shm_objects'):
            for remote_engine_id, shm_dict in self.remote_shm_objects.items():
                for layer_name, shm_obj in shm_dict.items():
                    try:
                        shm_obj.close()
                    except Exception as e:
                        logger.warning(f"Error closing remote shared memory: {e}")
                logger.debug(f"Cleaned up remote shared memory references for engine {remote_engine_id}")
            self.remote_shm_objects.clear()

        if hasattr(self, 'remote_cached_shm_kv_caches_completion_flags'):
            for remote_engine_id, shm_obj in self.remote_cached_shm_kv_caches_completion_flags.items():
                try:
                    shm_obj.close()
                except Exception as e:
                    logger.warning(f"Error closing remote completion flags shared memory: {e}")
                logger.debug(f"Cleaned up remote completion flags shared memory for engine {remote_engine_id}")
            self.remote_cached_shm_kv_caches_completion_flags.clear()

        # Clean up shared memory objects
        if hasattr(self, 'shm_kv_caches'):
            for layer_name, shm_obj in self.shm_kv_caches.items():
                try:
                    shm_obj.close()
                    shm_obj.unlink()
                    logger.debug(f"Cleaned up shared memory for {layer_name}")
                except FileNotFoundError:
                    # Already cleaned up
                    logger.debug(f"Shared memory for {layer_name} already cleaned up")
                except Exception as e:
                    logger.warning(f"Failed to cleanup shared memory for {layer_name}: {e}")
            self.shm_kv_caches.clear()
        if hasattr(self, 'shm_kv_caches_completion_flags'):
            try:
                self.shm_kv_caches_completion_flags.close()
                self.shm_kv_caches_completion_flags.unlink()
                logger.debug("Cleaned up shared memory for completion flags")
            except FileNotFoundError:
                # Already cleaned up
                logger.debug("Shared memory for completion flags already cleaned up")
            except Exception as e:
                logger.warning(f"Failed to cleanup shared memory for completion flags: {e}")
