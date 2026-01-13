
import triton,os
from triton import language as tl

from triton.backends.triton_shared.driver import TCUDriver
triton.runtime.driver.set_active(TCUDriver())

import torch

@triton.jit
def sum_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    # if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
    #     in_ptr.dtype.element_ty == tl.bfloat16
    # ):
    #     cdtype = tl.float32
    # else:
    #     cdtype = in_ptr.dtype.element_ty

    cdtype = in_ptr.dtype.element_ty
    tl.static_print(f"BLOCK MN: {BLOCK_M} {BLOCK_N} STATE: {STAGE}")
    # 如果 program_id(0) = 1且 BLOCK_M = 128，则 row_ids代表第 128 到 255 行（具体取决于索引从0还是1开始）
    row_ids = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    # 生成一个布尔掩码 row_mask（长度为 BLOCK_M），用于标记 row_ids中哪些行是有效的（小于总行数 M），防止对输入矩阵范围之外的内存进行读写
    row_mask = row_ids < M

    # 这个累加器通常位于 GPU 的快速共享内存（SRAM）或寄存器中，用于临时存储中间计算结果，减少对全局内存（DRAM）的访问
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    # off是当前列的起始索引，从0开始，每次步进 BLOCK_N，直到 N。
    # STAGE参数可能用于控制​​循环展开​​或​​预取​​的阶段数，以优化指令级并行和内存访问延迟（例如，在一次循环迭代中预加载下一个块的数据） pipeline loading 
    for off in tl.range(0, N, BLOCK_N, STAGE):
        col_ids = off + tl.arange(0, BLOCK_N)
        col_mask = col_ids < N
        mask = row_mask[:, None] & col_mask[None, :]

        a = tl.load(in_ptr + row_ids[:, None] * N + col_ids, mask, other=0).to(cdtype)
        acc += a
    out = tl.sum(acc, axis=1)
    tl.store(out_ptr + row_ids, out, row_mask)


if __name__ == "__main__":
    import torch

    # os.environ["LLVM_BINARY_DIR"] = "/home/leo/.triton/llvm/llvm-064f02da-ubuntu-x64/bin/"
    os.environ["LLVM_BINARY_DIR"] = "/home/leo/Downloads/dev/LLVM/install/bin/"
    os.environ["TRITON_SHARED_OPT_PATH"] = "/home/leo/Downloads/dev/triton_shared/triton/build/cmake.linux-x86_64-cpython-3.13/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
    
    # m = 1024
    # n = 1024 * 16
    m = 4
    n = 3 * 4
    x = torch.ones((m, n), device="cpu")
    print("x: {}".format(x))
    out = torch.empty((m,1), device="cpu")
    BLOCK_M = 1
    BLOCK_N = 1024
    grid = (triton.cdiv(m, BLOCK_M), 1, 1)
    sum_kernel[grid](x, out, m, n, BLOCK_M, BLOCK_N, STAGE=2, num_warps=4)
    print(out)
    print(x.sum(1))

