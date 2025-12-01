#include <cuda.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cub/cub.cuh>
#include <cfloat>
#include <cstdio>
#include <random>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>

/*
* Pragma unroll is breaking the kernel?
* Can't compare this to torch topk yet. Need to return both logits and indices & in sorted order.
    * Still, for the usual multinomial sampling that comes after topk, being sorted doesn't matter.
*/

#define SAFE_UPPER_BOUND 3072 // TODO REVIEW empirically observed heavy tail count in a bin for LLM inference

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

template<typename T>
void print_matrix(char* name, T *matrix, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (std::is_same<T, float>::value) {
                printf("%f ", static_cast<float>(matrix[i*cols + j]));
            } else {
                printf("%d ", matrix[i*cols + j]);
            }
        }
        printf("\n");
    }
}

bool verify(int* C, int* Cref, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; i < N; i++) {
        if (Cref[i] != C[i]) {
            printf("Divergence at %d: expected %d, actual %d\n", i, Cref[i], C[i]);
            return false;
        }
    }
    return true;
}

static inline __device__ uint16_t extract_bin_idx(float x) {
    union { __half h; uint16_t u16; } tmp;
    tmp.h = __float2half_rn(x);
    tmp.u16 = (x < 0.f) ? (~tmp.u16 & 0xffff) : (tmp.u16 | 0x8000);
    return tmp.u16 >> 7; // use most significant 9 bits to bin into 512 (2^9) bins
}

template<int NUM_THREADS>
__global__ void topk(float *logits, int *indices, int B, int V, int K) {
    constexpr int num_bins = NUM_THREADS; // key assumption
    constexpr int num_final_per_thread = SAFE_UPPER_BOUND / NUM_THREADS;

    using FinalSort = cub::BlockRadixSort<float, NUM_THREADS, num_final_per_thread, int>;

    __shared__ int smem_hist[num_bins]; // will also house the prefix sum result
    __shared__ int smem_threshold_bin_idx;
    __shared__ int smem_ki, smem_fi;

    __shared__ struct {
        float logits[SAFE_UPPER_BOUND];
        int indices[SAFE_UPPER_BOUND];
    } smem_final_items;
    __shared__ typename cub::BlockScan<int, NUM_THREADS>::TempStorage smem_scan;
    __shared__ typename FinalSort::TempStorage smem_final_sort;

    float* cur_logits = logits + blockIdx.x * V;
    int* cur_indices = indices + blockIdx.x * K;

    // 1. Create histogram of 512 bins
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        uint16_t bin_idx = extract_bin_idx(cur_logits[i]);
        atomicAdd(&smem_hist[bin_idx], 1);
    }
    __syncthreads();

    // 2. Exclusive prefix sum to get start index of each bin
    using Scan = cub::BlockScan<int, NUM_THREADS>;
    int bin_count = smem_hist[threadIdx.x], prefix_sum = 0, total_sum = 0;
    __syncthreads();
    
    Scan(smem_scan).ExclusiveSum(bin_count, prefix_sum, total_sum); // ty cub

    // 3. Find threshold bin index
    int threshold_val_idx = total_sum - K;
    int low = prefix_sum, high = prefix_sum + bin_count;
    if (low <= threshold_val_idx && threshold_val_idx < high)
        smem_threshold_bin_idx = threadIdx.x;

    if (threadIdx.x == 0) { smem_fi = 0; smem_ki = 0; } // TODO REVIEW can i get rid of this?
    __syncthreads();

    // 4. Collect top K indices of bins greater than threshold bin
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        uint16_t bin_idx = extract_bin_idx(cur_logits[i]);
        if (bin_idx > smem_threshold_bin_idx) { // guaranteed that it's in top K
            int ki = atomicAdd(&smem_ki, 1);
            cur_indices[ki] = i;
        } else if (bin_idx == smem_threshold_bin_idx) { // ambiguous bin. will sort these later in step 5
            int fi = atomicAdd(&smem_fi, 1);
            smem_final_items.logits[fi] = cur_logits[i];
            smem_final_items.indices[fi] = i;
        }
    }

    // 5. Collect top K indices from threshold bin
    int k_remaining = K - smem_ki;
    if (k_remaining == 0) return; // lucky!

    float final_logits[num_final_per_thread];
    int final_indices[num_final_per_thread];

    for (int i = threadIdx.x; i < SAFE_UPPER_BOUND; i += blockDim.x) {
        if (i < smem_fi) {
            final_logits[i] = smem_final_items.logits[i];
            final_indices[i] = smem_final_items.indices[i];
        } else {
            final_logits[i] = -FLT_MAX;
            final_indices[i] = -1; // TODO REVIEW necessary?
        }
    }

    FinalSort(smem_final_sort).SortDescendingBlockedToStriped(final_logits, final_indices);

    for (int i = 0; i < num_final_per_thread; i++) {
        int k_offset = threadIdx.x + i * blockDim.x;
        if (k_offset < k_remaining) {
            cur_indices[smem_ki + k_offset] = final_indices[i];
        }
    }
}

void cpu_topk(float *logits, int *indices, int B, int V, int K) {
    for (int b = 0; b < B; b++) {
        const float* row = logits + b * V;
        auto* out = indices + b * K;

        // Min-heap that stores (value, index)
        using Pair = std::pair<float, int>;
        auto cmp = [](const Pair& a, const Pair& b) { return a.first > b.first; };
        std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)> heap(cmp);

        // Build a K-sized min-heap for the row
        for (int i = 0; i < V; i++) {
            float v = row[i];

            if ((int)heap.size() < K) {
                heap.emplace(v, i);
            } else if (v > heap.top().first) {
                heap.pop();
                heap.emplace(v, i);
            }
        }

        // Extract K items from heap → unsorted top-K
        for (int i = 0; i < K; i++) {
            out[i] = heap.top().second;
            heap.pop();
        }

        // Optional: sort final indices by descending logit value
        std::sort(out, out + K, [&](int a, int b) {
            return row[a] > row[b];
        });
    }
}

int main() {
    float elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int B = 1024, V = 50000;
    constexpr int K = 2048;
    constexpr int NUM_THREADS = 512;
    constexpr int repeat_times = 50;

    float *hlogits, *dlogits;
    int *hindices_ref, *hindices, *dindices;

    hlogits = (float*)malloc(B*V*sizeof(float));
    hindices = (int*)malloc(B*K*sizeof(int));
    hindices_ref = (int*)malloc(B*K*sizeof(int));

    std::mt19937 gen(42);
    std::normal_distribution<float> distribution(0, 1);
    for (int i = 0; i < B*V; i++) hlogits[i] = distribution(gen);
    // print_matrix("logits", hlogits, B, V);

    cudaCheck(cudaMalloc((void**)&dlogits, B*V*sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dindices, B*K*sizeof(int)));

    cudaMemcpy(dlogits, hlogits, B*V*sizeof(float), cudaMemcpyHostToDevice);

    auto kernel = topk<NUM_THREADS>;

    kernel<<<B, NUM_THREADS>>>(dlogits, dindices, B, V, K);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());

    cudaMemcpy(hindices, dindices, B*K*sizeof(int), cudaMemcpyDeviceToHost);

    cpu_topk(hlogits, hindices_ref, B, V, K);
    // print_matrix("indices_ref", hindices_ref, B, K);
    // print_matrix("indices", hindices, B, K);
    // if (!verify(hindices, hindices_ref, B*K)) return -1;

    cudaEventRecord(start);
    for (int i = 0; i < repeat_times; i++) kernel<<<B, NUM_THREADS>>>(dlogits, dindices, B, V, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(start); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("avg time: %f ms\n", elapsed_time / repeat_times);
    
    cudaFree(dlogits); cudaFree(dindices); free(hlogits); free(hindices);
    return 0;
}