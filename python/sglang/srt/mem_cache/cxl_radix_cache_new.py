import logging
import torch
import threading
import time
import heapq

from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional, Tuple
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
from sglang.srt.managers.cxl_cache_controller import CXLCacheController
from python.sglang.srt.managers.cxl_controller import CXLClient

logger = logging.getLogger(__name__)


# When prefetch: get all the available data from cxl, alloc the data from device pool,
# lock the data in cxl manager, then fetch the data and put them in tree.
# When check_prefetch: wait for all prefetching process done, then return.
# When load_back: PANIC! after check_prefetch there should not be any host node.
# When evict: as RadixCache did.
# When write back: write to cxl async.
class CXLRadixCache(RadixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        tp_cache_group: torch.distributed.ProcessGroup,
        cxl_rpc_addr: str,
        io_backend: str,
        disable=False,
        enable_kv_cache_events=False,
    ):

        self.page_size = page_size
        assert self.page_size == 1, "page token hash is not supported now"

        self.tp_group = tp_cache_group
        self.tp_world_size = (
            torch.distributed.get_world_size(group=self.tp_group)
            if self.tp_group
            else 1
        )

        self.io_backend = io_backend
        self.write_threshold = (
            4  # NOTE: when this is small, may add duplicate write backup, locking too many TreeNode.
        )
        self.prefetch_threshold = 8
        self.cxl_rpc_addr = cxl_rpc_addr
        self.cxl_client = CXLClient(
            cxl_rpc_addr=cxl_rpc_addr,
        )

        self.cache_controller = CXLCacheController(
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            page_size=self.page_size,
            tp_group=self.tp_group,
            cxl_client=self.cxl_client,
            io_backend=self.io_backend,
            prefetch_threshold=self.prefetch_threshold,
        )

        self.ongoing_backup = {}
        self.ongoing_prefetch = {}

        self.load_cache_event = threading.Event()
        self.prefetch_done_events = {}

        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()
        if not (
            isinstance(self.kv_cache, MHATokenToKVPool)
            or isinstance(self.kv_cache, MLATokenToKVPool)
        ):
            raise ValueError(f"CXLRadixCache only supports MHA and MLA yet")

        super().__init__(
            req_to_token_pool,
            token_to_kv_pool_allocator,
            page_size,
            disable,
            enable_kv_cache_events,
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        super().reset()

    def match_prefix(self, key: List[int], **kwargs):
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            assert False, "node with no value is found, which should not happen."
            # host_hit_length += len(last_node.host_value)
            # last_node = last_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def init_load_back(
        self,
        last_node: TreeNode,
        host_hit_length: int,
        mem_quota: Optional[int] = None,
    ):
        raise RuntimeError("CXL Radix Cache should not load back.")

    def check_hicache_events(self):
        # Basically, we have two data transfer methods:
        #   - prefetch, load data from cxl to dram
        #   - write, send data from dram to cxl
        # We check revoked prefetch and write here, those success
        # prefetch will be checked in check_prefetch_progress.

        self.check_revoked_prefetch()
        self.check_backup_progress()

    def check_backup_progress(self):
        queue_size = torch.tensor(
            self.cache_controller.ack_backup_queue.qsize(), dtype=torch.int
        )
        if self.tp_world_size > 1:
            # synchrnoize TP workers to make the same update to hiradix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        for _ in range(queue_size.item()):
            ack_id, hash_value, completed_tokens = (
                self.cache_controller.ack_backup_queue.get()
            )
            node = self.ongoing_backup[ack_id]
            # FIXME: node lock and release, node writed or not? double check here.
            if completed_tokens == 0:
                node.backuped_cxl = False
            elif completed_tokens < len(node.key):
                # backup is only partially successful, split the node
                new_node = self._split_node(node.key, node, completed_tokens)
                new_node.backuped_cxl = True
                self.release(new_node)
            else:
                # TODO: Is there any chance that a node is splited when being writing to cxl?
                # if so, the completied tokens can be bigger than len(key)
                node.backuped_cxl = True
            self.release(node)
            del self.ongoing_backup[ack_id]

    def check_revoked_prefetch(self):
        queue_size = torch.tensor(
            self.cache_controller.prefetch_revoke_queue.qsize(), dtype=torch.int
        )
        if self.tp_world_size > 1:
            # synchrnoize TP workers to make the same update to hiradix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        for _ in range(queue_size.item()):
            req_id, batch_id = self.cache_controller.prefetch_revoke_queue.get()
            if req_id in self.ongoing_prefetch:
                last_host_node, _, device_indices, _ = self.ongoing_prefetch[req_id]
                self.token_to_kv_pool_allocator.free(device_indices)
                logger.info(
                        f"Revoking prefetch for request {req_id}, "
                        f"pool indices: {device_indices[:min(len(device_indices),10)]}... with length {len(device_indices)} due to insufficient hits."
                    )
                self.cxl_client.get_end(batch_id)
                self.dec_lock_ref(last_host_node)
                del self.ongoing_prefetch[req_id]
            else:
                # the revoked operation already got terminated
                pass

    # DONE
    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
    ):
        prefetch_length = len(new_input_tokens) - (
            len(new_input_tokens) % self.page_size
        )
        new_input_tokens = new_input_tokens[:prefetch_length]
        if prefetch_length < self.prefetch_threshold:
            return
        self.inc_lock_ref(last_host_node)
        device_indices = self.token_to_kv_pool_allocator.alloc(
            prefetch_length
        )  # HACK: Alloc enough slots, free them later
        if device_indices is None:
            self.evict(prefetch_length)
            device_indices = self.token_to_kv_pool_allocator.alloc(prefetch_length)
        if device_indices is None:
            self.dec_lock_ref(last_host_node)
            return
        logging.info(f"prefetch allocated {device_indices[:min(len(device_indices),10)]} with length {len(device_indices)} for {req_id}")
        self.prefetch_done_events[req_id] = threading.Event()
        operation = self.cache_controller.prefetch(
            req_id,
            device_indices,
            new_input_tokens,
            self.prefetch_done_events[req_id],
            last_hash,
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            new_input_tokens,
            device_indices,
            operation,
        )

    def check_prefetch_progress(self, req_id: str):
        if req_id not in self.ongoing_prefetch:
            # there is no ongoing prefetch for this request or it has been revoked
            return
        
        self.prefetch_done_events[req_id].wait()  # wait for prefetch done or revoked

        if req_id not in self.ongoing_prefetch:
            return

        # todo: more policies for prefetch progress such as timeout
        # the current policy is to prefetch with best effort and terminate when queuing is over
        last_host_node, token_ids, device_indices, operation = self.ongoing_prefetch[
            req_id
        ]

        completed_tokens = self.cache_controller.get_prefetch_result(operation)
        logger.debug(f"Prefetch {req_id} completed with {completed_tokens} tokens")

        min_completed_tokens = completed_tokens
        if self.tp_world_size > 1:
            # synchrnoize TP workers to make the same update to hiradix cache
            completed_tokens_tensor = torch.tensor(
                min_completed_tokens, dtype=torch.int
            )
            torch.distributed.all_reduce(
                completed_tokens_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            min_completed_tokens = completed_tokens_tensor.item()
        fetched_token_ids = token_ids[:min_completed_tokens]
        written_indices = device_indices[:min_completed_tokens]
        matched_length = self._insert_helper(
            last_host_node,
            fetched_token_ids,
            written_indices,
        )
        self.token_to_kv_pool_allocator.free(
            device_indices[:matched_length]
        ) 
        self.token_to_kv_pool_allocator.free(
            device_indices[min_completed_tokens:]
        )
        # NOTE: the matched_length to prefetch_length is freed in controller, dont
        # need to free here.
        self.dec_lock_ref(last_host_node)
        del self.ongoing_prefetch[req_id]

    def inc_hit_count(self, node: TreeNode):
        node.hit_count += 1

        if not node.backuped_cxl:
            if node.hit_count >= self.write_threshold:
                # write to host if the node is not backuped
                self.write_backup_cxl(node)

    def write_backup_cxl(self, node: TreeNode):
        assert (
            node.hash_value is not None and len(node.hash_value) != 0
        ), "every node should have a hash value"
        operation_id = self.cache_controller.write_backup_cxl(
            node.value, node.hash_value
        )
        self.ongoing_backup[operation_id] = node
        self.protect(node)  # HACK remember to release this

    def _insert_helper(self, node: TreeNode, key: List, value):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                self.inc_hit_count(node)
                total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                self.inc_hit_count(new_node)
                total_prefix_length += prefix_len

                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            new_node.host_value = None
            new_node.hash_value = self._get_node_hash_fn(
                key, node.get_last_hash_value()
            )
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self.inc_hit_count(new_node)

        return total_prefix_length

    # NOTE: directly copy from Hiradix cache
    def _split_node(self, key, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.loading = child.loading
        new_node.hit_count = child.hit_count
        new_node.backuped_cxl = child.backuped_cxl

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]
        if child.backuped:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]

        if child.hash_value:
            new_node.hash_value = child.hash_value[: split_len // self.page_size]
            child.hash_value = child.hash_value[split_len // self.page_size :]
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    def _get_node_hash_fn(self, key: List[int], last_hash: str) -> List[str]:
        # calc hash key for each page
        hash_value = []
        calc_count = 0
        remaining_tokens = len(key)
        while remaining_tokens >= self.page_size:
            last_hash = self.cache_controller.get_hash_str(
                key[calc_count : calc_count + self.page_size], last_hash
            )
            hash_value.append(last_hash)
            calc_count += self.page_size
            remaining_tokens -= self.page_size

        # if remaining_tokens > 0:
        #     last_hash = self.cache_controller.get_hash_str(
        #         key[calc_count : ], last_hash
        #     )
        #     hash_value.append(last_hash)
        return hash_value
