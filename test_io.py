import mmap
import ctypes
import torch

if __name__ == "__main__":
    # 以读写模式打开
    with open("/dev/dax0.0", "w+b") as f:
        # 指定映射长度（如 4096 字节）
        mm = mmap.mmap(f.fileno(), 4*(2**30),flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        mv = memoryview(mm)
        ts2 = torch.frombuffer(mv[16:32], dtype=torch.uint8)
        print(f"origin ts2: {ts2}")
        data_to_wrote = torch.randn([2,4],dtype=torch.float16,device="cuda:0").view(dtype=torch.uint8).flatten()
        ts2.copy_(data_to_wrote)

        del ts2
        del mv
        mm.close()