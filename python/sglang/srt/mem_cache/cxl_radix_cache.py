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

logger = logging.getLogger(__name__)


# prefetch: get all available cache and there position on CXL
# when load back,  CXL -> VRAM transfer
# when hit many times, VRAM -> CXL transfer
# when evicting, do nothing
class CXLRadixCache(RadixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        tp_cache_group: torch.distributed.ProcessGroup,
        cxl_rpc_addr: str,
        disable=False,
        enable_kv_cache_events=False,
    ):
        self.tp_group = tp_cache_group
        self.tp_world_size = (
            torch.distributed.get_world_size(group=self.tp_group)
            if self.tp_group
            else 1
        )
        self.cache_controller = None
        self.write_threshold = 3
        self.prefetch_threshold = 64  # TODO: validate the value here
        self.load_back_threshold = 10
        self.ongoing_write_through = {}
        self.ongoing_prefetch = {}
        self.load_cache_event = threading.Event()
        self.cxl_rpc_addr = cxl_rpc_addr
        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()
        # TODO: set up CXL RPC client
        super().__init__(
            req_to_token_pool,
            token_to_kv_pool_allocator,
            page_size,
            disable,
            enable_kv_cache_events,
        )

    def check_hicache_events(self):
        self.loading_check()  # check all tensor have been loaded to device from CXL
        self.check_revoked_prefetch()  # check all failed CXL read operations are revoked
        self.writing_check()  # check all tensor are writed to CXL

    def loading_check(self):
        while not self.cache_controller.ack_load_queue.empty():
            try:
                ack_id = self.cache_controller.ack_load_queue.get_nowait()
                start_node, end_node = self.ongoing_load_back[ack_id]
                self.dec_lock_ref(end_node)
                while end_node != start_node:
                    assert end_node.loading
                    end_node.loading = False
                    end_node = end_node.parent
                # clear the reference
                del self.ongoing_load_back[ack_id]
            except Exception:
                break

    def writing_check(self):
        """check if all writing operation is finishied

        Args:
            write_back (bool, optional): _description_. Defaults to False.
        """
        queue_size = torch.tensor(
            self.cache_controller.ack_write_queue.qsize(), dtype=torch.int  # TODO
        )
        if self.tp_world_size > 1:
            # synchrnoize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        for _ in range(queue_size.item()):
            ack_id = self.cache_controller.ack_write_queue.get()
            self.dec_lock_ref(self.ongoing_write_through[ack_id])
            del self.ongoing_write_through[ack_id]

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
            req_id = self.cache_controller.prefetch_revoke_queue.get()
            if req_id in self.ongoing_prefetch:
                del self.ongoing_prefetch[req_id]
            else:
                # the revoked operation already got terminated
                pass

    def check_prefetch_progress(self, req_id: str):
        if req_id not in self.ongoing_prefetch:
            # there is no ongoing prefetch for this request or it has been revoked
            return

        last_host_node, token_ids, operation = self.ongoing_prefetch[req_id]

        # TODO: terminate_prefetch groundbreaking change, be careful
        completed_tokens, hash_value, cxl_value = (
            self.cache_controller.terminate_prefetch(operation)
        )
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

        self._insert_helper_cxl(
            last_host_node,
            fetched_token_ids,
            cxl_value,
            hash_value[: min_completed_tokens // self.page_size],
        )
        del self.ongoing_prefetch[req_id]

    def _insert_helper_cxl(
        self, node: TreeNode, key: List, cxl_value: List, hash_value: List[str]
    ):

        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            cxl_value = cxl_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = None
            new_node.cxl_value = cxl_value
            new_node.hash_value = hash_value
            node.children[child_key] = new_node

        return total_prefix_length

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
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            self.inc_hit_count(node)

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            new_node.cxl_value = None
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._record_store_event(new_node)
            self.inc_hit_count(new_node)
        return total_prefix_length

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
    ):
        # align the number of fetching tokens to the page size
        prefetch_length = len(new_input_tokens) - (
            len(new_input_tokens) % self.page_size
        )
        if prefetch_length < self.prefetch_threshold:
            return
        new_input_tokens = new_input_tokens[:prefetch_length]
        last_host_node.protect_host()  # NOTE: Protect the node from eviction, unlock until load back finished.
        operation = self.cache_controller.prefetch(req_id, new_input_tokens, last_hash)
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            new_input_tokens,
            operation,
        )

    def write_backup(self, node: TreeNode):
        """Writes a backup copy of the given tree node.

        Args:
            node (TreeNode): The tree node to back up.
        """
        # TODO: implement backup logic, should looks like write_storage

        # 1: get_hash(hash_str,last_hash)
        pass

    def inc_hit_count(self, node: TreeNode):
        node.hit_count += 1
        if not node.cxl_backuped:
            if node.hit_count >= self.write_threshold:
                # write to host if the node is not backuped
                self.write_backup(node)

    def load_back(
        self,
        last_node: TreeNode,
        host_hit_length: int,
        mem_quota: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        # TODO: use host_hit_length to find the last_node, then release_host() it

        last_hit_node = last_node
        nodes_to_load = []
        while last_node.evicted:
            assert (
                last_node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, last_node)
            last_node = last_node.parent
        else:
            ancester_node = last_node

        # protect the ancestor nodes from eviction
        delta = self.inc_lock_ref(ancester_node)

        # load it all or not at all
        data_on_cxl = [].extend(n.cxl_value for n in nodes_to_load)

        # TODO: check the data_len correctness here
        data_len = (
            sum(d.numel() for d in data_on_cxl)
            // 2
            // (
                self.kv_cache.layer_num
                * self.kv_cache.head_num
                * self.kv_cache.head_dim
            )
        )
        if data_len < self.load_back_threshold or (
            data_len > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=data_on_cxl, node_id=last_hit_node.id
        )
        if device_indices is None:
            self.evict(data_len)
            device_indices = self.cache_controller.load(
                host_indices=data_on_cxl, node_id=last_hit_node.id
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None

        self.ongoing_load_back[last_hit_node.id] = (
            ancester_node,
            last_hit_node,
            host_hit_length,
        )
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)]
            offset += len(node.host_value)
            node.loading = True
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        return device_indices

    def init_load_back(
        self,
        last_node: TreeNode,
        host_hit_length: int,
        mem_quota: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Preparing KV cache loading from CXL host to device.
        """
        while host_hit_length > 0:
            t_node = last_node.parent
            host_hit_length -= 1

        if last_node.evicted:
            loading_values = self.load_back(last_node, host_hit_length, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                t_node.release_host()  # release the host reference after loading back
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        t_node.release_host()
        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def ready_to_load_host_cache(self):
        producer_index = self.cache_controller.layer_done_counter.next_producer()
        self.load_cache_event.set()
        return producer_index

    def evict(self, num_tokens: int):
        leaves = self._collect_leaves_device()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            # node is protected from eviction as it has ongoing prefetch
            if x.host_ref_counter > 0:
                continue

            # is it possible evict and insert happen simultaneously?
            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            for child in x.parent.children.values():
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                heapq.heappush(leaves, x.parent)

            self._record_remove_event(x)

    def _collect_leaves_device(self):
        def is_leaf(node):
            if node.evicted:
                return False
            if node == self.root_node:
                return False
            if len(node.children) == 0:
                return True
            for child in node.children.values():
                if not child.evicted:
                    return False
            return True

        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            if is_leaf(cur_node):
                ret_list.append(cur_node)
            else:
                for cur_child in cur_node.children.values():
                    if not cur_child.evicted:
                        stack.append(cur_child)
        return ret_list

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
            host_hit_length += 1  # NOTE: ground breaking change, use node depth instead of value length
            last_node = last_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def _match_prefix_helper(self, node: TreeNode, key: List):
        node.last_access_time = time.monotonic()
        child_key = self.get_child_key_fn(key)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.loading = child.loading
        new_node.hit_count = child.hit_count

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]
        if child.cxl_backuped:
            new_node.cxl_value = child.cxl_value[:split_len]
            child.cxl_value = child.cxl_value[split_len:]

        if child.hash_value:
            new_node.hash_value = child.hash_value[: split_len // self.page_size]
            child.hash_value = child.hash_value[split_len // self.page_size :]
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key[:10],
                current_node.cxl_value[:10] if current_node.cxl_value else "",
                current_node.hash_value[:10] if current_node.hash_value else "",
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"


if __name__ == "__main__":
    tree = CXLRadixCache(None, None, 1, None, "", disable=False)

    tree.insert("Hello")
    tree.insert("Hello")
    tree.insert("Hello_L.A.!")
    tree.insert("Hello_world! Happy")
    tree.pretty_print()
    leaves = tree._collect_leaves_device()
    import random

    leaf = random.choice(leaves)
    print(f"Selected leaf: {leaf.key}")
    tree._insert_helper_cxl(leaf, "Lucky", [1] * 5, [1] * 5)
    tree._insert_helper_cxl(leaf, "Smile", [1] * 5, [1] * 5)
    tree._insert_helper_cxl(leaf, "LuckySmile", [1] * 10, [1] * 10)
    tree.pretty_print()
    print(tree.match_prefix("Hello_world! Happy"))
