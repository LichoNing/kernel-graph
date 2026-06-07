/**
 * cute GEMM Kernel 示例
 * 
 * 使用NVIDIA CUTLASS cute库实现的矩阵乘法kernel
 * 展示了cute的核心概念：Tensor, Layout, TiledMMA, CopyAtom
 */

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

using namespace cute;

/**
 * cute GEMM Kernel
 * 
 * 计算: C = A * B
 * 其中 A: [M, K], B: [K, N], C: [M, N]
 */
__global__ void cute_gemm(
    half_t const* A, half_t const* B, half_t* C,
    int M, int N, int K
) {
    // ============================================
    // 1. 定义Tensor (Global Memory)
    // ============================================
    
    // A矩阵: [M, K] 列主序
    Tensor gA = make_tensor(
        make_gmem_ptr(A),
        make_shape(M, K),
        make_stride(Int<1>{}, M)
    );
    
    // B矩阵: [K, N] 列主序
    Tensor gB = make_tensor(
        make_gmem_ptr(B),
        make_shape(K, N),
        make_stride(Int<1>{}, K)
    );
    
    // C矩阵: [M, N] 列主序
    Tensor gC = make_tensor(
        make_gmem_ptr(C),
        make_shape(M, N),
        make_stride(Int<1>{}, M)
    );
    
    // ============================================
    // 2. 定义Tile形状
    // ============================================
    
    // Block-level tile: 128x128x32
    using bM = Int<128>;
    using bN = Int<128>;
    using bK = Int<32>;
    
    // ============================================
    // 3. 定义TiledMMA
    // ============================================
    
    // 使用Tensor Core MMA: m16n8k16
    using TiledMMA = decltype(make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{},
        Layout<Shape<_2, _2, _1>>{},     // 2x2 warps
        Tile<_32, _32, _16>{}            // 每个warp处理32x32x16
    ));
    TiledMMA tiled_mma;
    
    // ============================================
    // 4. 定义Shared Memory Tensor
    // ============================================
    
    // Shared memory layout
    using sA_layout = decltype(make_layout(
        make_shape(bM{}, bK{}),
        make_stride(Int<1>{}, bM{} + Int<8>{})  // 加上padding避免bank conflict
    ));
    
    using sB_layout = decltype(make_layout(
        make_shape(bN{}, bK{}),
        make_stride(Int<1>{}, bN{} + Int<8>{})  // 加上padding避免bank conflict
    ));
    
    // 分配shared memory
    __shared__ half_t smemA[cosize(sA_layout{})];
    __shared__ half_t smemB[cosize(sB_layout{})];
    
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout{});
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout{});
    
    // ============================================
    // 5. 定义CopyAtom (异步拷贝)
    // ============================================
    
    using CopyA = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
        Layout<Shape<_32, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{}
    ));
    
    using CopyB = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
        Layout<Shape<_32, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{}
    ));
    
    CopyA copy_a;
    CopyB copy_b;
    
    // ============================================
    // 6. 获取线程坐标
    // ============================================
    
    int idx = threadIdx.x;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    
    // ============================================
    // 7. 划分Global Tensor为Tiles
    // ============================================
    
    // 当前block处理的tile
    Tensor gA_block = local_tile(gA, make_tile(bM{}, bK{}), make_coord(bidy, _));
    Tensor gB_block = local_tile(gB, make_tile(bN{}, bK{}), make_coord(_, bidx));
    Tensor gC_block = local_tile(gC, make_tile(bM{}, bN{}), make_coord(bidy, bidx));
    
    // ============================================
    // 8. 创建Register Tensor (Accumulator)
    // ============================================
    
    ThrMMA thr_mma = tiled_mma.get_slice(idx);
    
    // A和B的fragment
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC_block);
    
    // Accumulator
    Tensor accum = make_tensor_like(tCgC);
    clear(accum);
    
    // ============================================
    // 9. 主循环 (Pipeline)
    // ============================================
    
    // 划分A和B的K维度
    int ntile = K / size<1>(gA_block);
    
    // Pipeline: 双缓冲
    for (int itile = 0; itile < ntile; ++itile) {
        // ---- Stage 1: 从Global拷贝到Shared ----
        Tensor gA_tile = gA_block(_, _, itile);
        Tensor gB_tile = gB_block(_, _, itile);
        
        // 异步拷贝 A
        copy(copy_a, gA_tile, sA);
        // 异步拷贝 B
        copy(copy_b, gB_tile, sB);
        
        // 等待拷贝完成
        cp_async_wait<0>();
        __syncthreads();
        
        // ---- Stage 2: MMA计算 ----
        // 从Shared加载到Register并执行MMA
        gemm(tiled_mma, tCsA, tCsB, accum);
        
        __syncthreads();
    }
    
    // ============================================
    // 10. Epilogue: 写回Global Memory
    // ============================================
    
    // 将accumulator写回C
    copy(tCgC, accum);
}

/**
 * 更高级的cute GEMM (带Pipeline优化)
 */
template <int NumStages = 3>
__global__ void cute_gemm_pipeline(
    half_t const* A, half_t const* B, half_t* C,
    int M, int N, int K
) {
    // 使用Pipeline进行多级缓冲
    using Pipeline = MainloopPipeline<NumStages>;
    
    // ... (更复杂的pipeline实现)
    
    // Producer-Consumer模式
    // Producer: 负责从Global拷贝到Shared
    // Consumer: 负责从Shared执行MMA
}

/**
 * cute GEMM with TMA (Hopper架构)
 */
__global__ void cute_gemm_tma(
    half_t const* A, half_t const* B, half_t* C,
    int M, int N, int K
) {
    // 使用Tensor Memory Accelerator
    // TMA可以自动处理内存拷贝，无需显式cp.async
    
    Tensor gA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(Int<1>{}, M));
    Tensor gB = make_tensor(make_gmem_ptr(B), make_shape(K, N), make_stride(Int<1>{}, K));
    Tensor gC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(Int<1>{}, M));
    
    // TMA descriptor
    auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, gA);
    auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, gB);
    auto tma_c = make_tma_copy(SM90_TMA_STORE{}, gC);
    
    // ... (TMA实现)
}
