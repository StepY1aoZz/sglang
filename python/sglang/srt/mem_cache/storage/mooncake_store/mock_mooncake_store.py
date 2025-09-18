import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig

DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024  # 16 MB

logger = logging.getLogger(__name__)

class MooncakeStore(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig = None):
        self.hash_map = {}
        
    def close(self):
        # MooncakeDistributedStore will automatically call the destructor, so
        # it is unnecessary to close it manually.
        pass

    def clear(self) -> None:
        self.hash_map.clear()

    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        return self.hash_map.get(key) is not None
    
    def batch_exists(self, keys) -> int:
        return 0
    
    def set(self, key, value = None, target_location = None, target_sizes = None):
        if self.hash_map.get(key) is not None:
            return False
        self.hash_map[key] = value if value is not None else target_location
        return True
    
    def get(self, key, target_location = None, target_sizes = None):
        return None
    
    def batch_set(self, keys, values = None, target_locations = None, target_sizes = None):
        assert len(keys) == len(target_locations) == len(target_sizes), f"Length mismatch: {len(keys)}, {len(target_locations)}, {len(target_sizes)}"
        for i in range(len(keys)):
            res = self.set(keys[i], None, target_locations[i], target_sizes[i])
            if not res:
                return False
        return True

    def batch_get(self, keys, target_locations = None, target_sizes = None):
        assert len(keys) == len(target_locations) == len(target_sizes), f"Length mismatch: {len(keys)}, {len(target_locations)}, {len(target_sizes)}"
        return 0