#include <cuda.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cstdio>
#include <random>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>

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

__global__ void topk(float *logits, int *indices, int B, int V, int K) {
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

    const int B = 2, V = 5;
    constexpr int K = 2;
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
    print_matrix("logits", hlogits, B, V);

    cudaCheck(cudaMalloc((void**)&dlogits, B*V*sizeof(float)));
    cudaCheck(cudaMalloc((void**)&dindices, B*K*sizeof(int)));

    cudaMemcpy(dlogits, hlogits, B*V*sizeof(float), cudaMemcpyHostToDevice);

    topk<<<B, NUM_THREADS>>>(dlogits, dindices, B, V, K);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());

    cudaMemcpy(hindices, dindices, B*K*sizeof(int), cudaMemcpyDeviceToHost);

    cpu_topk(hlogits, hindices_ref, B, V, K);
    print_matrix("indices_ref", hindices_ref, B, K);
    if (!verify(hindices, hindices_ref, B*K)) return -1;

    cudaEventRecord(start);
    for (int i = 0; i < repeat_times; i++) topk<<<B, NUM_THREADS>>>(dlogits, dindices, B, V, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(start); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("avg time: %f ms\n", elapsed_time / repeat_times);
    
    cudaFree(dlogits); cudaFree(dindices); free(hlogits); free(hindices);
    return 0;
}