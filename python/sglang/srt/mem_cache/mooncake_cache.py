from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.memory_pool import (
    ReqToTokenPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.hicache_storage import get_hash_str

DEBUG = True

if DEBUG:
    from sglang.srt.mem_cache.storage.mooncake_store.mock_mooncake_store import MooncakeStore
else:
    from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore


if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class MooncakeRadixCache(RadixCache):
    """RadixCache + LMCache IO.

    This subclass adds:
      - LMCache connector setup (device/host buffers, TP rank/size)
      - Two CUDA streams for async load/store
      - Layer-wise transfer executor wiring to the KV cache
      - Overridden `match_prefix` to fetch missing prefix chunks from LMCache
      - Extended cache_finalization paths to store back into LMCache
      - Eviction barrier that respects any in-flight host->device stores
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        disable: bool = False,
        enable_kv_cache_events: bool = False,
        model_config: Optional["ModelConfig"] = None,
        tp_size: int = 1,
        rank: int = 0,
        model_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            page_size=page_size,
            disable=disable,
            enable_kv_cache_events=enable_kv_cache_events,
        )

        self.kvcache = self.token_to_kv_pool_allocator.get_kvcache()

        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        if not (
            isinstance(self.kvcache, MHATokenToKVPool)
            or isinstance(self.kvcache, MLATokenToKVPool)
        ):
            raise ValueError(
                f"MooncakeRadixCache only supports MHATokenToKVPool or MLATokenToKVPool, got {type(self.kvcache)}"
            )

        self.is_mla = isinstance(self.kvcache, MLATokenToKVPool)

        config = HiCacheStorageConfig(
            tp_rank=rank,
            tp_size=tp_size,
            is_mla_model=self.is_mla,
            is_page_first_layout=False,
            model_name=model_name,
        )
        self.storage = MooncakeStore(config)

        self._in_flight_nodes: list[TreeNode] = []
        self._node_lock = threading.Lock()

    def reset(self):  # type: ignore[override]
        super().reset()
        if hasattr(self, "_in_flight_nodes"):
            with self._node_lock:
                self._in_flight_nodes.clear()

    def match_prefix(self, key: List[int], **kwargs) -> MatchResult:  # type: ignore[override]
        """Match cached prefix; if there's a tail miss, prefetch from LMCache.

        Reuses the base matching logic to obtain (value, last_node). If there
        remains a *page-aligned* uncached suffix and there is room (or after
        eviction), we allocate token slots and trigger an async LMCache load
        into those slots, then materialize a new child node for the retrieved
        chunk.
        """
        if self.disable or not key:
            return super().match_prefix(key, **kwargs)

        if self.page_size != 1:
            aligned_len = len(key) // self.page_size * self.page_size
            key = key[:aligned_len]

        base_res = super().match_prefix(key, **kwargs)
        value: torch.Tensor = base_res.device_indices
        last_node: TreeNode = base_res.last_device_node

        uncached_len = len(key) - value.numel()
        if uncached_len == 0:
            return base_res

        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.inc_lock_ref(last_node)
            self.evict(uncached_len)
            self.dec_lock_ref(last_node)

        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            return base_res

        last_hash = last_node.get_last_hash_value()

        hashes = self._calc_keys_hash(key[len(value) :], last_hash)

        num_exists = self.storage.batch_exists(hashes)
        logger.debug("num_retrieved_tokens: %s", num_exists)
        if num_exists == 0:
            self.token_to_kv_pool_allocator.free(token_slots)
            return base_res

        self.token_to_kv_pool_allocator.free(token_slots[num_exists:])
        hashes = hashes[:num_exists]
        token_slots = token_slots[:num_exists]
        data_ptr, lengths = self.kvcache.get_buf_infos_per_layer(token_slots)
        fetched = self._load_kv_from_mooncake(token_slots, lengths, data_ptr, hashes)
        if fetched % self.kvcache.layer_num != 0:
            logger.warning(
                "Fetched tokens %d is not aligned with layer num %d, aborted",
                fetched,
                self.kvcache.layer_num,
            )
            fetched = 0
        if fetched == 0:
            self.token_to_kv_pool_allocator.free(token_slots)
            return base_res
        if fetched != num_exists:
            self.token_to_kv_pool_allocator.free(token_slots[fetched:num_exists])

        new_node = TreeNode()
        start = value.numel()
        end = start + fetched
        new_node.key = key[start:end]
        new_node.value = token_slots[:fetched]
        new_node.parent = last_node
        new_node.hash_value = hashes[:fetched]
        last_node.children[self.get_child_key_fn(new_node.key)] = new_node
        last_node = new_node

        value = torch.cat([value, token_slots[:fetched]])
        self.evictable_size_ += fetched

        self._record_store_event(new_node.parent)
        self._record_store_event(new_node)

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def _calc_keys_hash(self, uncached_key: List[int], last_hash: str) -> List[str]:
        hashes = []
        for i in range(0, len(uncached_key), self.page_size):
            chunk = uncached_key[i : i + self.page_size]
            last_hash = get_hash_str(chunk, last_hash)
            hashes.append(last_hash)
        return hashes

    def cache_finished_req(self, req: "Req") -> None:  # type: ignore[override]
        """On request completion, insert device KV into radix and store to LMCache."""

        super().cache_finished_req(req)


        _, new_last_node, _, _ = self.match_prefix(req.origin_input_ids) # store the prefill tokens
        assert new_last_node is not None

        self.inc_lock_ref(new_last_node)
        slots = new_last_node.value
        keys = new_last_node.hash_value
        data_ptr, lengths = self.kvcache.get_buf_infos_per_layer(slots)
        with torch.cuda.stream(self.store_stream):
            self._backup_kv_to_mooncake(lengths, data_ptr, keys)
        with self._node_lock:
            self._in_flight_nodes.append(new_last_node)

    def evict(self, num_tokens: int) -> None:  # type: ignore[override]
        """Before base eviction, wait for any outstanding stores and release locks."""
        if self.disable:
            return

        self.store_stream.synchronize()
        with self._node_lock:
            for node in self._in_flight_nodes:
                self.dec_lock_ref(node)
            self._in_flight_nodes.clear()

        super().evict(num_tokens)

    def _load_kv_from_mooncake(
        self,
        lengths: List[int],
        data_ptr: List[int],
        hashes: List[str],
    ) -> int:
        hashes = [
            f"{h}_{l}"
            for h in hashes
            for l in range(
                self.kvcache.start_layer,
                self.kvcache.start_layer + self.kvcache.layer_num,
            )
        ]
        if not self.is_mla:
            hashes = [f"{h}_k" for h in hashes] + [
                f"{h}_v" for h in hashes
            ]  # length is 2*num_exists
        else:
            hashes = [
                f"{h}_k" for h in hashes
            ]  # for compatibility with current mooncake store implementation

        num_copied = self.storage.batch_get(
            hashes,
            data_ptr,
            lengths,
        )
        return num_copied

    def _backup_kv_to_mooncake(
        self,
        lengths: List[int],
        data_ptr: List[int],
        hashes: List[str],
    ) -> int:
        hashes = [
            f"{h}_{l}"
            for h in hashes
            for l in range(
                self.kvcache.start_layer,
                self.kvcache.start_layer + self.kvcache.layer_num,
            )
        ]
        if not self.is_mla:
            hashes = [f"{h}_k" for h in hashes] + [
                f"{h}_v" for h in hashes
            ]  # length is 2*num_exists
        else:
            hashes = [
                f"{h}_k" for h in hashes
            ]

        num_copied = self.storage.batch_set(
            hashes,
            target_locations=data_ptr,
            target_sizes=lengths,
        )
        return num_copied

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

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            new_node.hash_value = self._calc_keys_hash(key, node.get_last_hash_value())
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._record_store_event(new_node)
        return total_prefix_length

    # NOTE: directly copy from Hiradix cache, need to split node hashes when split them.
    def _split_node(self, key, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

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

    def pretty_print(self):  # type: ignore[override]
        super().pretty_print()
        try:
            logger.debug(
                "evictable=%d protected=%d", self.evictable_size_, self.protected_size_
            )
        except Exception:  # pragma: no cover
            pass


if __name__ == "__main__":
    cache = MooncakeRadixCache(
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        disable=False,
        enable_kv_cache_events=False,
        model_config=None,
        tp_size=1,
        rank=0,
        tp_group=None,
    )
    cache.insert([1, 2, 3], torch.tensor([10, 11, 12], dtype=torch.int64))
    cache.insert([1, 2, 3, 4], torch.tensor([10, 11, 12, 13], dtype=torch.int64))
    cache.pretty_print()
