#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <vector>
#include <random>

#define NNZ_PER_ROW 40 // static workload assumption: M=N=K=4096, uniform sparsity=0.01 -> nnzPerRow=40

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define IS_CLOSE(a, b) (abs(a - b) < 1e-5 && abs(a - b) / (abs(a) + abs(b) + 1e-5) < 1e-5)

#define CHECK_CUDA(call) do {                               \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess) {                               \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1);                                            \
    }                                                       \
} while(0)

#define CHECK_CUSPARSE(call) do {                           \
    cusparseStatus_t stat = call;                           \
    if (stat != CUSPARSE_STATUS_SUCCESS) {                  \
        std::cerr << "cuSPARSE error" << std::endl;         \
        exit(1);                                            \
    }                                                       \
} while(0)

__global__ void oneThreadPerRow(int *d_csrOffsets, int *d_columns, float *d_values, float *d_B, float *d_C, int m, int k, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    int start = d_csrOffsets[row];
    int end = d_csrOffsets[row+1];
    for (int j = 0; j < n; j++) {
        float sum = 0.0f;
        for (int i = start; i < end; i++) {
            sum += d_values[i] * d_B[d_columns[i]*n + j];
        }
        d_C[row*n + j] = sum;
    }
}

__global__ void oneThreadPerRowPrefetch(int *d_csrOffsets, int *d_columns, float *d_values, float *d_B, float *d_C, int m, int k, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    int start = d_csrOffsets[row];
    int end = d_csrOffsets[row+1];
    float r_values[NNZ_PER_ROW];
    int r_columns[NNZ_PER_ROW];
    for (int i = start; i < end; i++) {
        r_values[i-start] = d_values[i];
        r_columns[i-start] = d_columns[i];
    }
    
    for (int j = 0; j < n; j++) {
        float sum = 0.0f;
        for (int i = start; i < end; i++) {
            sum += r_values[i-start] * d_B[r_columns[i-start]*n + j];
        }
        d_C[row*n + j] = sum;
    }
}

__global__ void oneBlockPerRow(int *d_csrOffsets, int *d_columns, float *d_values, float *d_B, float *d_C, int m, int k, int n) {
    int row = blockIdx.x;
    int start = d_csrOffsets[row];
    int end = d_csrOffsets[row+1];
    __shared__ float s_values[NNZ_PER_ROW];
    __shared__ int s_columns[NNZ_PER_ROW];
    for (int i = 0; i < CEIL_DIV(end - start, 32); i++) {
        if (threadIdx.x + i*32 < NNZ_PER_ROW) {
            s_values[threadIdx.x + i*32] = d_values[start + threadIdx.x + i*32];
            s_columns[threadIdx.x + i*32] = d_columns[start + threadIdx.x + i*32];
        }
    }
    __syncthreads();
    for (int t = 0; t < n / blockDim.x; t++) {
        int j = t * blockDim.x + threadIdx.x;
        if (j < n) {
            float sum = 0.0f;
            for (int i = 0; i < NNZ_PER_ROW; i++) {
                sum += s_values[i] * d_B[s_columns[i]*n + j];
            }
            d_C[row*n + j] = sum;
        }
    }
}

template<int THREADS_PER_BLOCK, int THREADS_PER_ROW>
__global__ void oneBlockMultiRow(int *d_csrOffsets, int *d_columns, float *d_values, float *d_B, float *d_C, int m, int k, int n) {
    const int ROWS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_ROW;
    int wi = threadIdx.x / THREADS_PER_ROW;
    int row = blockIdx.x * THREADS_PER_BLOCK / THREADS_PER_ROW + wi;
    int start = d_csrOffsets[row];
    int end = d_csrOffsets[row+1];
    __shared__ float s_values[ROWS_PER_BLOCK][NNZ_PER_ROW];
    __shared__ int s_columns[ROWS_PER_BLOCK][NNZ_PER_ROW];
    for (int i = 0; i < CEIL_DIV(end - start, THREADS_PER_ROW); i++) {
        if (threadIdx.x%THREADS_PER_ROW + i*THREADS_PER_ROW < NNZ_PER_ROW) {
            s_values[wi][threadIdx.x%THREADS_PER_ROW + i*THREADS_PER_ROW] = d_values[start + threadIdx.x%THREADS_PER_ROW + i*THREADS_PER_ROW];
            s_columns[wi][threadIdx.x%THREADS_PER_ROW + i*THREADS_PER_ROW] = d_columns[start + threadIdx.x%THREADS_PER_ROW + i*THREADS_PER_ROW];
        }
    }
    __syncthreads();
    for (int t = 0; t < n / THREADS_PER_ROW; t++) {
        int j = t * THREADS_PER_ROW + threadIdx.x % THREADS_PER_ROW;
        if (j < n) {
            float sum = 0.0f;
            for (int i = 0; i < NNZ_PER_ROW; i++) {
                sum += s_values[wi][i] * d_B[s_columns[wi][i]*n + j];
            }
            d_C[row*n + j] = sum;
        }
    }
}

void run_kernel(int knum, int *d_csrOffsets, int *d_columns, float *d_values, float *d_B, float *d_C, int m, int k, int n) {
    int nThreadsPerBlock = 1;
    switch (knum) {
        case 0:
            oneThreadPerRow<<<CEIL_DIV(m, nThreadsPerBlock), nThreadsPerBlock>>>(d_csrOffsets, d_columns, d_values, d_B, d_C, m, k, n);
            break;
        case 1:
            oneThreadPerRowPrefetch<<<CEIL_DIV(m, nThreadsPerBlock), nThreadsPerBlock>>>(d_csrOffsets, d_columns, d_values, d_B, d_C, m, k, n);
            break;
        case 2:
            nThreadsPerBlock = 32;
            oneBlockPerRow<<<m, nThreadsPerBlock>>>(d_csrOffsets, d_columns, d_values, d_B, d_C, m, k, n);
            break;
        case 3:
            const int THREADS_PER_BLOCK = 512; const int THREADS_PER_ROW = 32;
            oneBlockMultiRow<THREADS_PER_BLOCK, THREADS_PER_ROW><<<CEIL_DIV(m, THREADS_PER_BLOCK/THREADS_PER_ROW), THREADS_PER_BLOCK>>>(d_csrOffsets, d_columns, d_values, d_B, d_C, m, k, n);
            break;
    }
}

int main(int argc, char* argv[]) {
    // Static workload assumptions: M=N=K=4096, uniformsparsity=0.01 -> nnzPerRow=40
    const int n = 4096, m = 4096, k = 4096; // matrix dimensions
    const double sparsity = 0.01; // assume uniform 1% nonzero
    const int nnz = static_cast<int>(n * m * sparsity); // given above, 167772
    int knum = 0;

    if (argc > 1)
        knum = atoi(argv[1]);

    std::cout << "Matrix: " << n << "x" << m << ", k=" << k
              << ", sparsity=" << sparsity
              << ", nnz=" << nnz << "\n";

    // Generate random CSR matrix
    std::vector<int> h_csrOffsets(n+1);
    std::vector<int> h_columns(nnz);
    std::vector<float> h_values(nnz);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> valDist(0.0f, 1.0f);
    std::uniform_int_distribution<int> colDist(0, m-1);

    int offset = 0;
    for (int row = 0; row < n; row++) {
        h_csrOffsets[row] = offset;
        for (int j = 0; j < NNZ_PER_ROW; j++) {
            h_columns[offset] = colDist(rng);
            h_values[offset] = valDist(rng);
            offset++;
        }
    }
    h_csrOffsets[n] = offset;

    // Dense matrices
    std::vector<float> h_B(m*k);
    std::vector<float> h_C(n*k, 0.0f);
    std::vector<float> h_C_ref(n*k, 0.0f);

    std::mt19937 rngB(0);
    std::uniform_real_distribution<float> valDistB(0.0f, 1.0f);

    for (int i = 0; i < m*k; i++)
        h_B[i] = valDistB(rngB);

    // Device memory
    int *d_csrOffsets, *d_columns;
    float *d_values, *d_B, *d_C_ref, *d_C;
    CHECK_CUDA(cudaMalloc(&d_csrOffsets, (n+1)*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_columns, nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nnz*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, m*k*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, n*k*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_ref, n*k*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_csrOffsets, h_csrOffsets.data(), (n+1)*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_columns, h_columns.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values.data(), nnz*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), m*k*sizeof(float), cudaMemcpyHostToDevice));

    // cuSPARSE setup
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC_ref;

    CHECK_CUSPARSE(cusparseCreateCsr(&matA, n, m, nnz,
        d_csrOffsets, d_columns, d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, m, k, k, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC_ref, n, k, k, d_C_ref, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC_ref,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // Warmup
    run_kernel(knum, d_csrOffsets, d_columns, d_values, d_B, d_C, m, k, n);

    // Cusparse Reference
    CHECK_CUSPARSE(cusparseSpMM(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC_ref,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

    // Verify warmup result against reference
    CHECK_CUDA(cudaMemcpy(h_C_ref.data(), d_C_ref, n*k*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, n*k*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n*k; i++) {
        if (!IS_CLOSE(h_C_ref[i], h_C[i])) {
            std::cerr << "Error: h_C_ref[" << i << "] = " << h_C_ref[i] << ", h_C[" << i << "] = " << h_C[i] << std::endl;
            std::cerr << "Failed correctness check!" << std::endl;
            return -1;
        }
    }

    // Benchmark timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    int repeat_times = 50;
    for (int i=0; i<repeat_times; i++) {
        run_kernel(knum, d_csrOffsets, d_columns, d_values, d_B, d_C, m, k, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= repeat_times; // average per call

    // Metrics
    double flops = 2.0 * nnz * k;
    double gflops = (flops / (ms * 1e-3)) / 1e9;

    double bytes = nnz * (sizeof(float) + sizeof(int)) +
                   (n+1) * sizeof(int) +
                   (m*k + n*k) * sizeof(float);
    double bandwidth = (bytes / (ms * 1e-3)) / 1e9;

    std::cout << "Latency: " << ms << " ms\n";
    std::cout << "GFLOP/s: " << gflops << "\n";
    std::cout << "GB/s: " << bandwidth << "\n";

    // Cleanup
    cudaFree(d_csrOffsets); cudaFree(d_columns); cudaFree(d_values);
    cudaFree(d_B); cudaFree(d_C); cudaFree(dBuffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC_ref);
    cusparseDestroy(handle);
}