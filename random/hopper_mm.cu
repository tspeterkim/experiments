#include <cuda.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cstdio>
#include <random>

typedef __nv_bfloat16 floatX;
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
                cudaGetErrorString(error));
        exit(1);
    }
}
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

bool verify(floatX* C, floatX* Cref, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; i < N; i++) {
        diff = std::fabs(__bfloat162float(Cref[i]) - __bfloat162float(C[i]));
        if (diff > 0.1) {
            printf("Divergence at %d: expected %.2f, actual %.2f (Diff %.2f)\n", i, __bfloat162float(Cref[i]), __bfloat162float(C[i]), diff);
            return false;
        }
    }
    return true;
}

__host__ __device__ void print_matrix(floatX* A, int M, int N, bool row_major = true, const char* name = "") {
    printf("%s:\n", name);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%5.2f ", __bfloat162float(A[row_major ? i*N + j : j*M + i]));
        }
        printf("\n");
    }
}

cublasHandle_t cublas_handle;
void runCublasGemmBF16(int M, int N, int K, floatX *A, floatX *B, floatX *C) {
    float alpha = 1, beta = 0;
    // C(column major) = A(row major) * B(column major)
    cublasStatus_t status = cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_16BF,
        K, B, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // cublasStatus_t status = cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_32F,
    //     N, B, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
    // TODO: both of these work?

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS error: %d\n", status);
        exit(1);
    }
}

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }
__device__ uint64_t make_smem_desc(floatX* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16; 
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32; 
    desc |= 1llu << 62; // 128B swizzle
    return desc;
}

__device__ inline uint64_t timestamp() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(ret) :: "memory");
    return ret;
}

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait_group() {
    // static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

__device__ void wgmma_m64n128k16(float d[8][8], floatX* sA, floatX* sB) {
    uint64_t a_desc = make_smem_desc(&sA[0]); uint64_t b_desc = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66,    %67,  %68,  %69,  %70;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
            "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
            "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
            "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
            "+f"(d[3][6]), "+f"(d[3][7]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
            "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]), "+f"(d[5][0]), "+f"(d[5][1]),
            "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]),
            "+f"(d[6][6]), "+f"(d[6][7]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
            "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(a_desc), "l"(b_desc), "n"(1), "n"(1), "n"(1), "n"(0), "n"(0));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma_m64n256k16(float d[16][8], floatX* sA, floatX* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]); uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
        " %104, %105, %106, %107, %108, %109, %110, %111,  "
        " %112, %113, %114, %115, %116, %117, %118, %119,  "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130,    %131,  %132,  %133,  %134;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
            "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
            "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
            "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
            "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
            "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
            "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
            "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
            "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]), "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

__device__ __forceinline__ void init_semaphore(uint64_t* sema, int thread_count, int transaction_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sema)); 
    asm volatile ("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar_ptr), "r"(thread_count+transaction_count) : "memory");
}

__device__ __forceinline__ void expect_bytes(uint64_t* sema, uint32_t bytes) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sema)); 
    asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar_ptr), "r"(bytes));
}

__device__ __forceinline__ void load_async(floatX* dst, void const* src_tma_map, uint64_t* sema, int global_col_idx, int global_row_idx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sema));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile ("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n" :: "r"(dst_ptr), "l"(tma_ptr), "r"(global_col_idx), "r"(global_row_idx), "r"(mbar_ptr) : "memory");
}

__device__ __forceinline__ void wait(uint64_t* sema, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sema));
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

__device__ __forceinline__ void arrive(uint64_t* sema, uint32_t count=1) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sema)); 
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}

__device__ void arrive_cluster(uint64_t* bar, uint32_t cta_id, uint32_t count=1) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n\t"
        ".reg .b32 remAddr32;\n\t" // declare 32b reg
        "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t" // get smem addr of other cta's barrier
        "mbarrier.arrive.shared::cluster.b64  _, [remAddr32], %2;\n\t" // arrive on other cta's barrier
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(count));
}

__device__ static inline void load_async_multicast(floatX *dst, void const* const src_tma_map, uint64_t* bar, int global_col_idx, int global_row_idx, uint16_t cluster_mask) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    asm volatile (
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%3, %4}], [%2], %5;"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
        "r"(global_col_idx), "r"(global_row_idx), "h"(cluster_mask)
        : "memory"
    );
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <int BM, int BN, int BK, int QSIZE>
struct SMem {
    alignas(128) floatX A[QSIZE][BM*BK];
    alignas(128) floatX B[QSIZE][BK*BN];
    alignas(128) floatX C[BM*BN];
};

template<int BM, int BN, int BK, int NUM_SMS>
struct NaiveSchedule {
    int start, end;

    __device__ __forceinline__ NaiveSchedule(int M, int N, int sm_id) { // # blocks = # SMs
        int total_blocks = (M/BM) * (N/BN);
        int blocks_per_sm = total_blocks / NUM_SMS;
        int remaining_blocks = total_blocks - blocks_per_sm * NUM_SMS;
        if (sm_id < remaining_blocks) {
            start = sm_id * (blocks_per_sm + 1);
            end = start + blocks_per_sm + 1;
        } else {
            start = remaining_blocks * (blocks_per_sm + 1) + (sm_id - remaining_blocks) * blocks_per_sm;
            end = start + blocks_per_sm;
        }
    }

    __device__ __forceinline__ int next() { return start == end ? -1 : start++; }
    __device__ __forceinline__ int peek_next() { return start == end ? -1 : start;}
};

template<int BM, int BN, int TM, int TN, int NUM_SMS>
struct L2ReuseSchedule {
    int block;
    int it;
    int total_blocks_m;
    int total_blocks_n;

    __device__ __forceinline__ L2ReuseSchedule(int M, int N, int _block) {
        block = _block;
        it = 0;
        total_blocks_m = M/BM;
        total_blocks_n = N/BN;
        assert(total_blocks_m%TM == 0 && total_blocks_n%TN == 0);
    }

    __device__ __forceinline__ int next() {
        int num = it*NUM_SMS + block;
        if (num >= total_blocks_m*total_blocks_n) return -1;
        
        int cur_tile = num / (TM*TN);
        int cur_tile_pos = num % (TM*TN);
        int m = TM*(cur_tile / (total_blocks_n/TN));
        int n = TN*(cur_tile % (total_blocks_n/TN));
        m += cur_tile_pos / TN;
        n += cur_tile_pos % TN;
        ++it;
        return m*total_blocks_n + n;
    }
};

template<int BM, int BN, int BK, int WGMMA_M, int WGMMA_N, int WGMMA_K, int NUM_THREADS, int QSIZE, int NUM_SMS, int CTAS_PER_CLUSTER>
__global__ __launch_bounds__(NUM_THREADS) void __cluster_dims__(CTAS_PER_CLUSTER,1,1) tc(floatX* C, const __grid_constant__ CUtensorMap tensorMapA, const __grid_constant__ CUtensorMap tensorMapB, const __grid_constant__ CUtensorMap tensorMapC, int M, int N, int K) {
    extern __shared__ __align__(128) floatX smem[];
    SMem<BM, BN, BK, QSIZE> &s = *reinterpret_cast<SMem<BM, BN, BK, QSIZE> *>(smem);

    int ki_max = K / BK;
    int wg = threadIdx.x / 128;
    int tid = threadIdx.x % 128;

    uint32_t cluster_id; // cluster id out of all clusters
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(cluster_id) :);

    // NaiveSchedule<BM, BN, BK, NUM_SMS> schedule(M, N, blockIdx.x);
    L2ReuseSchedule<BM*2, BN, 8, 8, NUM_SMS/2> schedule(M, N, cluster_id);

    uint32_t cluster_ctarank; // in-cluster cta rank
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(cluster_ctarank) :);

    // First init barriers
    __shared__ __align__(8) uint64_t full[QSIZE], empty[QSIZE];
    if (threadIdx.x == 0) {
        for (int qi = 0; qi < QSIZE; qi++) {
            init_semaphore(&full[qi], 0, 1); 
            init_semaphore(&empty[qi], 4, 0); // 2 warpgroups * 2 per cluster. Cluste-wide sync
        }
    }
    // __syncthreads(); // TODO REVIEW for the semaphore
    asm volatile("barrier.cluster.arrive;\n" : :);
    asm volatile("barrier.cluster.wait;\n" : :);

    if (wg == 0) { // producer (wg=0)
        warpgroup_reg_dealloc<24>();
        if (tid == 0) {
            int p = 0, qi = 0, ni, mi;
            for (int bi = schedule.next(); bi >= 0; bi = schedule.next()) {
                ni = bi % (N/BN); 
                mi = 2*(bi / (N/BN)) + cluster_ctarank;
                for (int ki = 0; ki < ki_max; ki++, qi++) {
                    if (qi == QSIZE) { qi = 0; p ^= 1; }
                    wait(&empty[qi], p);

                    expect_bytes(&full[qi], sizeof(s.A[qi]) + sizeof(s.B[qi]));
                    load_async(&s.A[qi][0], &tensorMapA, &full[qi], ki*BK, mi*BM);

                    if (cluster_ctarank == 0)
                        load_async_multicast(&s.B[qi][0], &tensorMapB, &full[qi], ki*BK, ni*BN, 0b11);
                }
            }
        }
    } else { // consumer (wg=1,2)
        warpgroup_reg_alloc<240>();
        wg--;

        float d[WGMMA_N/16][8];
        int p = 0, qi = 0, ni, mi;
        if (tid < CTAS_PER_CLUSTER) { // indicate queue is empty
            for (int qi = 0; qi < QSIZE; qi++) arrive_cluster(&empty[qi], tid);
        }
        for (int bi = schedule.next(); bi >= 0; bi = schedule.next()) {
            ni = bi % (N/BN); 
            mi = 2*(bi / (N/BN)) + cluster_ctarank;

            { // first tile, use scaleD = 0 (to avoid clearing d every block)
                if (qi == QSIZE) { qi = 0; p ^= 1; }
                wait(&full[qi], p);
                warpgroup_arrive();
                floatX* wgmma_sA = s.A[qi] + wg*WGMMA_M*BK;
                floatX* wgmma_sB = s.B[qi];
                wgmma_m64n256k16<0, 1, 1, 0, 0>(d, &wgmma_sA[0], &wgmma_sB[0]); // only first one needs 0 zscale

                #pragma unroll
                for (int k_it = 1; k_it < BK/WGMMA_K; k_it++)
                    wgmma_m64n256k16<1, 1, 1, 0, 0>(d, &wgmma_sA[k_it*WGMMA_K], &wgmma_sB[k_it*WGMMA_K]);
                warpgroup_commit_group();
                warpgroup_wait_group<0>();
                if (tid < CTAS_PER_CLUSTER) arrive_cluster(&empty[qi], tid);
                qi++;
            }

            for (int ki = 1; ki < ki_max; ki++, qi++) {
                if (qi == QSIZE) { qi = 0; p ^= 1; }
                wait(&full[qi], p);

                // Compute
                warpgroup_arrive();
                floatX* wgmma_sA = s.A[qi] + wg*WGMMA_M*BK;
                floatX* wgmma_sB = s.B[qi];
                #pragma unroll
                for (int k_it = 0; k_it < BK/WGMMA_K; k_it++) {
                    wgmma_m64n256k16<1, 1, 1, 0, 0>(d, &wgmma_sA[k_it*WGMMA_K], &wgmma_sB[k_it*WGMMA_K]);
                }
                warpgroup_commit_group();
                warpgroup_wait_group<0>();

                // Signal that qi is processed so can be filled again
                if (tid < CTAS_PER_CLUSTER) arrive_cluster(&empty[qi], tid);
            }

            asm volatile("cp.async.bulk.wait_group 0;"); // wait for previous tile's store to complete

            // 3. Store
            int warp = tid / 32;
            int lane = tid % 32;
            int row = warp * 16 + lane / 4;
            // floatX* blockC = C + mi*BM + ni*BN*M; // offset into C block
            for (int w = 0; w < WGMMA_N/16; w++) {
                int col = 16*w + 2*(tid % 4);
                #define C(r, c) s.C[(r) + wg*WGMMA_M + (c)*BM] // Putting paranthesis around r and c important for operator precedence
                C(row, col) = d[w][0];
                C(row, col+1) = d[w][1];
                C(row+8, col) = d[w][2];
                C(row+8, col+1) = d[w][3];
                C(row, col+8) = d[w][4];
                C(row, col+9) = d[w][5];
                C(row+8, col+8) = d[w][6];
                C(row+8, col+9) = d[w][7];
                #undef C
            }

            // TMA Async Smem -> Gmem
            asm volatile ("bar.sync 1, 256;\n");
            if (threadIdx.x == 128) { // only want one thread in 2 warpgroups init TMA store
                cde::cp_async_bulk_tensor_2d_shared_to_global(&tensorMapC, mi*BM, ni*BN, &s.C);
                cde::cp_async_bulk_commit_group(); // start off TMA write
            }
        }
    }
}

template<int BlockMajorSize, int BlockMinorSize, bool swizzle=true>
__host__ static inline CUtensorMap allocate_tensorMap(floatX* src, int blocks_height, int blocks_width) {
    CUtensorMap d_tma_map;
    // cudaMalloc(&d_tma_map, sizeof(CUtensorMap));

    // allocate on host first
    // CUtensorMap h_tmp_map;
    void* gmem_addr = (void*) src;
    constexpr int rank = 2;
    const uint64_t GMEM_HEIGHT = BlockMajorSize * blocks_height;
    const uint64_t GMEM_WIDTH = BlockMinorSize * blocks_width;
    uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT}; // width comes first?
    uint64_t stride[rank-1] = {GMEM_WIDTH * sizeof(floatX)};
    uint32_t box_size[rank] = {(uint32_t)BlockMinorSize, (uint32_t)BlockMajorSize};
    uint32_t elem_stride[rank] = {1, 1}; // unit in elems, not bytes

    CUresult result = cuTensorMapEncodeTiled(
        &d_tma_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        gmem_addr,
        size,
        stride,
        box_size,        
        elem_stride,         
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        (swizzle) ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    assert(result == CUDA_SUCCESS);
    // cudaMemcpy(d_tma_map, &h_tmp_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return d_tma_map;
}

__global__ void warmupKernel() {__shared__ int s[100]; s[0] += s[1];}

CUtensorMap d_tma_map_A, d_tma_map_B, d_tma_map_C;
int main() {
    warmupKernel<<<1024, 1024>>>();

    float elapsed_time, ref_elapsed_time;
    cudaEvent_t start, stop, ref_start, ref_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&ref_start);
    cudaEventCreate(&ref_stop);

    const int M = 4096, N = 4096, K = 4096;
    constexpr int BM = 128, BN = 256, BK = 64;
    constexpr int WGMMA_M = 64, WGMMA_N = 256, WGMMA_K = 16;
    constexpr int NUM_SMS = 128;
    constexpr int NUM_THREADS = 128 * 3;
    constexpr int QSIZE = 3;
    constexpr int CTAS_PER_CLUSTER = 2;

    floatX *hA, *hB, *hC, *hCref;
    floatX *dA, *dB, *dC, *dC_ref;

    hA = (floatX *)malloc(M*K*sizeof(floatX));
    hB = (floatX *)malloc(K*N*sizeof(floatX));
    hC = (floatX *)malloc(M*N*sizeof(floatX));
    hCref = (floatX *)malloc(M*N*sizeof(floatX));

    std::mt19937 gen(42);
    std::normal_distribution<float> distribution(0, 1);
    for (int i = 0; i < M*K; i++) hA[i] = distribution(gen);
    for (int i = 0; i < K*N; i++) hB[i] = distribution(gen);
    for (int i = 0; i < M*N; i++) hC[i] = 0.0f;
    for (int i = 0; i < M*N; i++) hCref[i] = 0.0f;

    // cpu_gemm(hA, hB, hCref, M, N, K);
    // print_matrix(hA, M, K, true, "A");
    // print_matrix(hB, K, N, false, "B");
    // print_matrix(hCref, M, N, false, "Cref");

    cudaCheck(cudaMalloc((void**)&dA, M*K*sizeof(floatX)));
    cudaCheck(cudaMalloc((void**)&dB, K*N*sizeof(floatX)));
    cudaCheck(cudaMalloc((void**)&dC, M*N*sizeof(floatX)));
    cudaCheck(cudaMalloc((void**)&dC_ref, M*N*sizeof(floatX)));

    cudaMemcpy(dA, hA, M*K*sizeof(floatX), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K*N*sizeof(floatX), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, M*N*sizeof(floatX), cudaMemcpyHostToDevice);
    cudaMemcpy(dC_ref, hCref, M*N*sizeof(floatX), cudaMemcpyHostToDevice);

    int repeat_times = 10;
    long flops = 2LL * M * N * K;

    // Get cuBLAS reference
    cublasCreate(&cublas_handle);
    runCublasGemmBF16(M, N, K, dA, dB, dC_ref); // warmup is super important
    cudaEventRecord(ref_start);
    for (int i = 0; i < repeat_times; i++) {
        runCublasGemmBF16(M, N, K, dA, dB, dC_ref);
    }
    cudaEventRecord(ref_stop);
    cudaEventSynchronize(ref_start);
    cudaEventSynchronize(ref_stop);
    cudaEventElapsedTime(&ref_elapsed_time, ref_start, ref_stop);
    printf("cuBLAS avg time: %f s, perf: %f TFLOPS\n", 
        ref_elapsed_time / 1000.0 / repeat_times, 
        (repeat_times * flops * 1e-9) / ref_elapsed_time);
    cudaMemcpy(hCref, dC_ref, M*N*sizeof(floatX), cudaMemcpyDeviceToHost);

    // Allocate tensorMaps
    d_tma_map_A = allocate_tensorMap<BM, BK>(dA, M/BM, K/BK);
    d_tma_map_B = allocate_tensorMap<BN, BK>(dB, N/BN, K/BK);
    d_tma_map_C = allocate_tensorMap<BN, BM, false>(dC, N/BN, M/BM);

    auto* kernel = tc<BM, BN, BK, WGMMA_M, WGMMA_N, WGMMA_K, NUM_THREADS, QSIZE, NUM_SMS, CTAS_PER_CLUSTER>;
    constexpr size_t sMemSize = sizeof(SMem<BM, BN, BK, QSIZE>);
    cudaCheck(cudaFuncSetAttribute(kernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    kernel<<<NUM_SMS, NUM_THREADS, sMemSize>>>(dC, d_tma_map_A, d_tma_map_B, d_tma_map_C, M, N, K);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError()); // Check for async errors during kernel run

    cudaMemcpy(hC, dC, M*N*sizeof(floatX), cudaMemcpyDeviceToHost);

    // print_matrix(hC, M, N, false, "C"); print_matrix(hCref, M, N, false, "Cref");

    if (!verify(hC, hCref, M*N)) return -1;

    cudaEventRecord(start);
    for (int i = 0; i < repeat_times; i++) {
        kernel<<<NUM_SMS, NUM_THREADS, sMemSize>>>(dC, d_tma_map_A, d_tma_map_B, d_tma_map_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("avg time: %f s, perf: %f TFLOPS\n", 
        elapsed_time / 1000.0 / repeat_times, (repeat_times * flops * 1e-9) / elapsed_time);
    
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hCref);
    return 0;
}