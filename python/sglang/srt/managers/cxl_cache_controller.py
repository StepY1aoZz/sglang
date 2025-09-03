import logging
import os
import threading
from queue import Empty, Full, PriorityQueue, Queue
from typing import TYPE_CHECKING, List, Optional
from sglang.srt.managers.cache_controller import StorageOperation, PrefetchOperation
from sglang.srt.managers.cxl_controller import CXLClient

import torch
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    get_hash_str_mooncake,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator


logger = logging.getLogger(__name__)


class CXLPrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        done_event: threading.Event = None,
        last_hash: Optional[str] = None,
        batch_id: int = 0,
    ):
        self.request_id = request_id
        self.done_event = done_event  # NOTE: should be set in revoke and prefetch done
        self.data = None  # save the return value of cxl manager, including the tensor loc and offsets

        self._done_flag = False
        self.batch_id = batch_id
        self._lock = threading.Lock()

        super().__init__(host_indices, token_ids, last_hash)

    def increment(self, num_tokens: int):
        with self._lock:
            if self._done_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_done(self):
        with self._lock:
            self._done_flag = True

    def is_done(self) -> bool:
        return self._done_flag


class CXLCacheController:
    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        cxl_client: CXLClient,
        io_backend: str = "",
        prefetch_threshold: int = 16,
    ):
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        self.mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        self.page_size = page_size
        self.io_backend = io_backend
        self.cxl_client = cxl_client
        self.get_hash_str = get_hash_str_mooncake
        self.prefetch_threshold = max(prefetch_threshold, self.page_size)
        # create a new communication group for synchronizing storage operations across TP workers
        self.tp_world_size = torch.distributed.get_world_size(group=tp_group)
        if self.tp_world_size > 1:
            group_ranks = torch.distributed.get_process_group_ranks(tp_group)
            self.prefetch_tp_group = torch.distributed.new_group(
                group_ranks, backend="gloo"
            )
            self.backup_tp_group = torch.distributed.new_group(
                group_ranks, backend="gloo"
            )
        self.layer_done_counter = None
        self.stop_event = threading.Event()

        self.write_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

        self.prefetch_thread = threading.Thread(
            target=self.prefetch_thread_func, daemon=True
        )
        self.backup_thread = threading.Thread(
            target=self.backup_thread_func, daemon=True
        )
        self.prefetch_queue = Queue()
        self.backup_queue = Queue()

        self.prefetch_revoke_queue = Queue()
        self.ack_backup_queue = Queue()

        self.prefetch_thread.start()
        self.backup_thread.start()

    def reset(self):
        self.stop_event.set()

        self.prefetch_thread.join()
        self.backup_thread.join()
        self.prefetch_queue.queue.clear()
        self.backup_queue.queue.clear()
        self.prefetch_revoke_queue.queue.clear()
        self.ack_backup_queue.queue.clear()

        self.stop_event.clear()

        self.prefetch_thread = threading.Thread(
            target=self.prefetch_thread_func, daemon=True
        )
        self.backup_thread = threading.Thread(
            target=self.backup_thread_func, daemon=True
        )
        self.prefetch_thread.start()
        self.backup_thread.start()

    def prefetch(
        self,
        request_id: str,
        device_indices: torch.Tensor,
        new_input_tokens: List[int],
        done_event: threading.Event = None,
        last_hash: Optional[str] = None,
    ) -> int:
        """
        Prefetch KV caches from storage backend to device memory.
        """
        operation = CXLPrefetchOperation(
            request_id, device_indices, new_input_tokens, done_event, last_hash
        )
        torch.cuda.current_stream().synchronize()
        self.prefetch_queue.put(operation)
        return operation

    def get_prefetch_result(self, operation: CXLPrefetchOperation) -> int:
        operation.mark_done()
        return operation.completed_tokens

    def prefetch_thread_func(self):
        """
        Manage prefetching operations from storage backend to device memory.
        """
        self.prefetch_buffer = Queue()
        aux_thread = threading.Thread(target=self.prefetch_io_aux_func, daemon=True)
        aux_thread.start()
        while (not self.stop_event.is_set()) or not self.prefetch_queue.empty():
            try:
                operation = self.prefetch_queue.get(block=True, timeout=1)
                if operation is None:
                    continue
                last_hash = operation.last_hash
                tokens_to_fetch = operation.token_ids

                storage_hit_count = 0
                remaining_tokens = len(tokens_to_fetch)
                hash_value = []
                while remaining_tokens >= self.page_size:
                    last_hash = self.get_hash_str(
                        tokens_to_fetch[
                            storage_hit_count : storage_hit_count + self.page_size
                        ],
                        last_hash,
                    )
                    hash_value.append(last_hash)
                    storage_hit_count += self.page_size
                    remaining_tokens -= self.page_size
                batch_id, exist_result = self.cxl_client.get(hash_value)
                operation.batch_id = batch_id
                storage_hit_count = len(exist_result) * self.page_size
                if self.tp_world_size > 1:
                    storage_hit_count_tensor = torch.tensor(
                        storage_hit_count, dtype=torch.int
                    )
                    torch.distributed.all_reduce(
                        storage_hit_count_tensor,
                        op=torch.distributed.ReduceOp.MIN,
                        group=self.prefetch_tp_group,
                    )
                    storage_hit_count = storage_hit_count_tensor.item()

                if storage_hit_count < self.prefetch_threshold:
                    self.prefetch_revoke_queue.put(
                        (operation.request_id, operation.batch_id)
                    )
                    operation.done_event.set()
                else:
                    operation.hash_value = hash_value[
                        : (storage_hit_count // self.page_size)
                    ]
                    # self.mem_pool_device_allocator.free(
                    #     operation.host_indices[storage_hit_count:]
                    # )
                    operation.host_indices = operation.host_indices[:storage_hit_count]
                    operation.data = exist_result
                    logger.debug(
                        f"Prefetching {len(operation.hash_value)} pages for request {operation.request_id}."
                    )
                    self.prefetch_buffer.put(operation)
            except Empty:
                continue

    def prefetch_io_aux_func(self):
        """
        Auxiliary function conducting IO operations for prefetching.
        """
        torch.cuda.set_stream(self.load_stream)
        while not self.stop_event.is_set():
            try:
                operation = self.prefetch_buffer.get(block=True, timeout=1)
                self.load_cxl_page(operation)
                self.load_stream.synchronize()
            except Empty:
                continue

    def load_cxl_page(self, operation: CXLPrefetchOperation):
        """
        Conducting CXL page transfer for prefetching data into device.
        """
        if not self.cxl_client:
            logger.error("CXL client is not initialized.")
            return

        logger.debug(f"Starting CXL page reading for request {operation.request_id}.")
        flat_pages = self.cxl_client.get_tensor_to_read(
            operation, self._get_shape_helper(), self.mem_pool_device.dtype
        )
        logger.debug(
            f"prefetch data: {flat_pages[0]}, data length: {len(flat_pages)}, data[0] shape: {flat_pages[0].shape}"
        )
        self._load_from_cxl_all_layer(flat_pages, operation)

        self.cxl_client.get_end(operation.batch_id)
        operation.done_event.set()

    def _get_shape_helper(self) -> torch.Size:
        if isinstance(self.mem_pool_device, MHATokenToKVPool):
            shape = self.mem_pool_device.k_buffer[0].shape  # (size,head_num,head_dim)
            return (
                torch.Size([2, self.mem_pool_device.layer_num, self.page_size])
                + shape[1:]
            )  # (2, layer_num, page_size, head_num, head_dim)

        elif isinstance(self.mem_pool_device, MLATokenToKVPool):
            shape = self.mem_pool_device.kv_buffer[0].shape  # (size,1,lora+rope)
            return (
                torch.Size([self.mem_pool_device.layer_num, self.page_size]) + shape[1:]
            )  # (layer_num, page_size, 1, lora+rope)
        else:
            raise ValueError("Unsupported memory pool device type")

    def _load_from_cxl_all_layer(
        self, flat_pages: List[torch.Tensor], operation: CXLPrefetchOperation
    ):
        indices = operation.host_indices
        logger.debug(f"trying to load into mem pool: {indices.tolist()}")
        logger.debug(f"flat pages: {len(flat_pages)}")
        logger.debug(f"page shape: {flat_pages[0].shape}")
        assert (
            indices.dim() == 1
        ), f"Indices should be a 1D tensor, but recieved {indices.shape}"
        if isinstance(self.mem_pool_device, MHATokenToKVPool):
            for i in range(0, len(indices), self.page_size):
                index = indices[i]
                for j in range(self.mem_pool_device.layer_num):
                    self.mem_pool_device.k_buffer[j][
                        index : index + self.page_size, :, :
                    ].copy_(
                        flat_pages[i // self.page_size][0, j, :, :, :],
                        non_blocking=True,
                    )
                    self.mem_pool_device.v_buffer[j][
                        index : index + self.page_size, :, :
                    ].copy_(
                        flat_pages[i // self.page_size][1, j, :, :, :],
                        non_blocking=True,
                    )
                if not operation.increment(self.page_size):
                    # terminated
                    # self.mem_pool_device_allocator.free(
                    #     indices[operation.completed_tokens :]
                    # )
                    break
        elif isinstance(self.mem_pool_device, MLATokenToKVPool):
            for i in range(0, len(indices), self.page_size):
                index = indices[i]
                for j in range(self.mem_pool_device.layer_num):
                    self.mem_pool_device.kv_buffer[j][
                        index : index + self.page_size, :, :
                    ].copy_(
                        flat_pages[i // self.page_size][j, :, :, :],
                        non_blocking=True,
                    )
                if not operation.increment(self.page_size):
                    # terminated
                    # self.mem_pool_device_allocator.free(
                    #     indices[operation.completed_tokens :]
                    # )
                    break
        else:
            raise ValueError("Unsupported memory pool device type")

    def write_backup_cxl(self, value: torch.Tensor, hash_value: List[str]):
        operation = PrefetchOperation(None, value, None)
        operation.hash_value = hash_value
        torch.cuda.current_stream().synchronize()
        self.backup_queue.put(operation)
        return operation.id

    def backup_thread_func(self):
        torch.cuda.set_stream(self.write_stream)
        while not self.stop_event.is_set():
            try:
                operation = self.backup_queue.get(block=True, timeout=1)
                if operation is None:
                    continue

                self.write_cxl_page(operation)

                self.write_stream.synchronize()
                min_completed_tokens = operation.completed_tokens
                if self.tp_world_size > 1:
                    completed_tokens_tensor = torch.tensor(
                        min_completed_tokens, dtype=torch.int
                    )
                    torch.distributed.all_reduce(
                        completed_tokens_tensor,
                        op=torch.distributed.ReduceOp.MIN,
                        group=self.backup_tp_group,
                    )
                    min_completed_tokens = completed_tokens_tensor.item()

                self.ack_backup_queue.put(
                    (
                        operation.id,
                        operation.hash_value[: min_completed_tokens // self.page_size],
                        min_completed_tokens,
                    )
                )

            except Empty:
                continue

    def write_cxl_page(self, operation: PrefetchOperation):
        hash_value = operation.hash_value
        indices = operation.host_indices
        size_per_page = self._get_shape_helper()[-4:]  # (layer_num, page_size, .. , ..)

        if isinstance(self.mem_pool_device, MHATokenToKVPool):
            size_per_page = (
                torch.Size([2]) + size_per_page
            )  # (2, layer_num, page_size, .. , ..)

        raw_length_per_page = (
            size_per_page.numel() * self.mem_pool_device.dtype.itemsize
        )

        # 1. try alloc len(hash_value) spaces
        # NOTE: the length of offsets may not equal to length of hash_value, only success prefix returns
        # when alloc failed because of no more free space, the manager just stopped and return RPC resp.
        batch_id, offsets, exists = self.cxl_client.put(
            hash_value, [raw_length_per_page] * len(hash_value)
        )
        operation.batch_id = batch_id

        # 2. write data to cxl
        for i in range(len(exists)):
            if exists[i]:
                continue
            index = indices[i * self.page_size]
            data = self._get_device_data_pages(index)
            logger.info(
                f"Writing to CXL: {hash_value[i]}, progress:{i}/{len(exists)}, data length {len(data)}, data[0] shape {data[0].shape}"
            )
            self.cxl_client.write_to_cxl(
                offsets[i],
                raw_length_per_page,
                data,
                isinstance(self.mem_pool_device, MHATokenToKVPool),
            )  # layer first (2, layer_num, page_size, ..,..)
        operation.completed_tokens += self.page_size * len(offsets)
        self.cxl_client.put_end(operation.batch_id)

    def _get_device_data_pages(self, start_index: int) -> List[torch.Tensor]:
        # NOTE: I choose not to use torch.concat here for reducing data copy, but
        # it adds more complexity to the source code. Be careful when dealing with
        # MHA and MLA, since they are not in the same length.
        data = []
        if isinstance(self.mem_pool_device, MHATokenToKVPool):
            for j in range(self.mem_pool_device.layer_num):
                data.extend(
                    [
                        self.mem_pool_device.k_buffer[j][
                            start_index : start_index + self.page_size, :, :
                        ],
                        self.mem_pool_device.v_buffer[j][
                            start_index : start_index + self.page_size, :, :
                        ],
                    ]
                )
        elif isinstance(self.mem_pool_device, MLATokenToKVPool):
            for j in range(self.mem_pool_device.layer_num):
                data.append(
                    self.mem_pool_device.kv_buffer[j][
                        start_index : start_index + self.page_size, :, :
                    ]
                )
        else:
            raise ValueError("Unsupported memory pool device type")
        return data


from collections import namedtuple

if __name__ == "__main__":
    cxl_client = CXLClient("localhost:50051", "/dev/dax0.0", 4 * (1024 * 1024))
    layer_num = 2
    page_size = 1
    pool_size = 16
    head_num = 8
    head_dim = 16
    dtype = torch.bfloat16
    fake_mha_mem_pool_k = [
        torch.randn(pool_size, head_num, head_dim, dtype=dtype)
    ] * layer_num
    fake_mha_mem_pool_v = [
        torch.randn(pool_size, head_num, head_dim, dtype=dtype)
    ] * layer_num
    hash_values = ["key1", "key2", "key3"]
    start_index = torch.tensor([0])
    data = []
    for j in range(layer_num):
        data.extend(
            [
                fake_mha_mem_pool_k[j][start_index : start_index + page_size, :, :],
                fake_mha_mem_pool_v[j][start_index : start_index + page_size, :, :],
            ]
        )
    raw_length_per_page = (
        fake_mha_mem_pool_k[0].shape[1:].numel() * layer_num * dtype.itemsize
    )
    cxl_client.write_to_cxl(
        0,
        raw_length_per_page,
        data,
        True,
    )

    operation = StorageOperation(None, None)
    handle = namedtuple("handle", ["offset", "len"])
    operation.data = [handle(0, raw_length_per_page * 2)]
    flat_pages = cxl_client.get_tensor_to_read(
        operation,
        shape=torch.Size([2, layer_num, page_size, head_num, head_dim]),
        target_dtype=dtype,
    )

    new_slot_k = [torch.zeros_like(fake_mha_mem_pool_k[0])] * layer_num
    new_slot_v = [torch.zeros_like(fake_mha_mem_pool_v[0])] * layer_num
    indices = [1]
    for i in range(0, len(indices), page_size):
        index = indices[i]
        for j in range(layer_num):
            new_slot_k[j][index : index + page_size, :, :].copy_(
                flat_pages[i // page_size][0, j, :, :, :],
                non_blocking=False,
            )
            new_slot_v[j][index : index + page_size, :, :].copy_(
                flat_pages[i // page_size][1, j, :, :, :],
                non_blocking=False,
            )

    print(
        new_slot_v[0],
        fake_mha_mem_pool_v[i][start_index : start_index + page_size, :, :],
    )

    for i in range(len(new_slot_v)):
        assert torch.equal(
            new_slot_v[i],
            fake_mha_mem_pool_v[i][start_index : start_index + page_size, :, :],
        )
        assert torch.equal(
            new_slot_k[i],
            fake_mha_mem_pool_k[i][start_index : start_index + page_size, :, :],
        )
