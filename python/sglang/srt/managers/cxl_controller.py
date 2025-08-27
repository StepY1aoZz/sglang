import os
import mmap
import torch
import logging
import grpc

from typing import List, Tuple, TYPE_CHECKING
from sglang.srt.managers.proto import cache_pb2_grpc, cache_pb2

if TYPE_CHECKING:
    from sglang.srt.managers.cxl_cache_controller import CXLPrefetchOperation

logger = logging.getLogger(__name__)

debug_mode = True


class CXLManager:
    def __init__(self, cxl_dev_path: str = None, cxl_dev_size: int = None):
        if cxl_dev_path is None:
            cxl_dev_path = os.getenv("SGL_CXL_DEV_PATH")
            if cxl_dev_path is None:
                raise ValueError("CXL device path must be specified.")
        self.cxl_dev_path = cxl_dev_path

        if cxl_dev_size is None:
            cxl_dev_size = int(os.getenv("SGL_CXL_DEV_SIZE"))
            if cxl_dev_size is None:
                raise ValueError("CXL device size must be specified.")

        self.cxl_dev_size = cxl_dev_size

        self.f = open(self.cxl_dev_path, "w+b")
        self.mm = mmap.mmap(
            self.f.fileno(),
            self.cxl_dev_size,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        self.mv = memoryview(self.mm)

    def get_from(
        self,
        offset: int,
        raw_size: int = None,
        size: int = None,
        dtype: torch.dtype = torch.uint8,
    ) -> torch.Tensor:
        if raw_size is not None and size is not None:
            raise ValueError("Only one of raw_size or size should be specified.")

        if raw_size is None and size is None:
            raise ValueError("Either raw_size or size must be specified.")

        # dtype not specified, use uint8 by default.
        # Note that in this scene the raw_size and size are the same.

        if raw_size is not None:
            m = self.mm[offset : offset + raw_size]
        else:
            raw_size = size * dtype.itemsize
            m = self.mm[offset : offset + raw_size]

        ts = torch.frombuffer(m, dtype=torch.uint8).view(dtype)
        return ts

    def set_to(self, offset: int, data: torch.Tensor):
        # data = data.contiguous() not sure if we need this
        raw_size = data.numel() * data.element_size()
        ts = torch.frombuffer(self.mv[offset : offset + raw_size], dtype=torch.uint8)
        ts.copy_(data.flatten().view(dtype=torch.uint8),non_blocking=True)


# manage the cxl mmap object and rpc call to manager
class CXLClient:
    def __init__(self, cxl_rpc_addr: str, cxl_path: str = None, cxl_size: int = None):
        self.cxl_rpc_addr = cxl_rpc_addr
        self.channel = grpc.insecure_channel(self.cxl_rpc_addr)
        self.client = cache_pb2_grpc.CacheStub(self.channel)
        self.manager = CXLManager(cxl_dev_path=cxl_path, cxl_dev_size=cxl_size)

    def get(self, hash_strs: List[str]) -> List:
        hash_bytes = [s.encode("utf-8") for s in hash_strs]
        req = cache_pb2.GetStartRequest(keys=hash_bytes)
        res = self.client.GetStart(req)
        return res.batchID, res.handles

    def get_tensor_to_read(
        self,
        operation: "CXLPrefetchOperation",
        shape: torch.Size = None,
        target_dtype: torch.dtype = None,
    ):
        """
        reshape cxl data to tensor for reading data into dram
        """
        flat_pages = []
        for handle in operation.data:  # handle:(offset int, len int)
            offset = handle.offset
            length = handle.len  # count in bytes
            tensor = self.manager.get_from(offset, raw_size=length, dtype=target_dtype)
            if shape is not None:
                tensor = tensor.view(shape)
            flat_pages.append(tensor)
        return flat_pages

    def write_to_cxl(
        self, offset: int, raw_length: int, data: List[torch.Tensor], is_mha: bool
    ):
        """Write data to CXL memory.

        Args:
            offset (int): the beginning of CXL memory offset to write, count in byte
            raw_length (int): the TOTAL RAW length of data
            data (List[torch.Tensor]): the list of tensors to write
            is_mha (bool): whether the data is for MHA or MLA
        """
        if is_mha:
            stop = len(data) // 2
        else:
            stop = len(data)
        size_per_layer = raw_length // len(data)
        assert (
            size_per_layer == data[0].numel() * data[0].element_size()
        ), "Size mismatch"
        for i in range(stop):
            if is_mha:
                self.manager.set_to(offset + i * size_per_layer, data[2 * i])  # k cache
                self.manager.set_to(
                    offset + raw_length // 2 + i * size_per_layer, data[2 * i + 1]
                )  # v cache
            else:
                self.manager.set_to(offset + i * size_per_layer, data[i])  # kv cache

    def get_end(self, batch_id: int):
        self.client.GetEnd(cache_pb2.GetEndRequest(batchID=batch_id, revoked=False))

    def put(
        self, hash_values: list[str], lengths: List[int]
    ) -> Tuple[int, List[int], List[bool]]:
        if debug_mode:
            assert len(hash_values) == len(
                lengths
            ), "hash_values and lengths must match in length."
        hash_bytes = [s.encode("utf-8") for s in hash_values]
        req = cache_pb2.PutStartRequest(keys=hash_bytes, lens=lengths)
        resp = self.client.PutStart(req)
        return resp.batchID, resp.offsets, resp.exists

    def put_end(self, batch_id: int):
        self.client.PutEnd(cache_pb2.PutEndRequest(batchID=batch_id))


if __name__ == "__main__":
    cxl_client = CXLClient("localhost:50051", "/dev/dax0.0", 4 * (1024 * 1024))
    # Test 1
    # =========================
    # put_batch_id, offsets, exists = cxl_client.put(["hash2"], [1024])
    # cxl_client.put_end(put_batch_id)
    # batchID, handles = cxl_client.get(["hash2", "hash3"])
    # print(batchID, handles, len(handles))
    # assert len(handles) == 1, "Should only have one handle for hash2"
    # assert offsets[0] == handles[0].offset, "Offsets should match"
    # cxl_client.get_end(batchID)

    # Test 2
    # ==========================
    layer_num = 6
    dtype = torch.bfloat16
    data = [
        torch.randn(2, 4, 8, dtype=dtype),
    ] * layer_num

    raw_size = data[0].numel() * dtype.itemsize * layer_num

    layer_first = torch.stack(data, dim=0).contiguous().flatten()
    page_first = torch.stack(data, dim=1).contiguous().flatten()

    cxl_client.write_to_cxl(0, raw_size, data, is_mha=False)
    new_data = cxl_client.manager.get_from(0, raw_size=raw_size, dtype=dtype)
    if torch.equal(new_data, layer_first):
        print("is layer first")
    if torch.equal(new_data, page_first):
        print("is page first")

    # Test 3
    # ==========================
    # layer_num = 6
    # dtype = torch.bfloat16
    # page_size = [1, 1, 1]
    # data = [
    #     torch.randn(page_size, dtype=dtype),
    # ] * layer_num
    # raw_size = data[0].numel() * dtype.itemsize * layer_num
    # k_layer_first = torch.stack([data[0], data[2], data[4]], dim=0)
    # k_page_first = torch.stack([data[0], data[2], data[4]], dim=0).view([1,3,1,1])
    # v_layer_first = torch.stack([data[1], data[3], data[5]], dim=0)
    # cxl_client.write_to_cxl(0, raw_size, data, is_mha=True)
    # new_data = cxl_client.manager.get_from(0, raw_size=raw_size, dtype=dtype).view([2, 3] + page_size)
    # assert torch.equal(new_data[0, :, :, :, :], k_layer_first)
    # assert torch.equal(new_data[1, :, :, :, :], v_layer_first)
    # assert torch.equal(new_data[0, :, :, :, :].view([1,3,1,1]), k_page_first)
