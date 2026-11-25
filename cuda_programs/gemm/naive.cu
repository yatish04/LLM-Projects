#include <cuda_runtime.h>
/**
 * Pros of this approach:
 * - Simple and straightforward implementation. Every thread
 * is responsible to reduce the entire row of A and column of B
 * and compute the sum and store it in C.
 * 
 * Cons of this approach:
 * Every thread tx,ty, loads the entire tx row of A and
 * ty column of B which is redundant.
 * 
 */
__global__ void matrix_multiplication_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    int output_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (output_row < M && output_col < K) {
        float sum = 0.0f;

        for (int i = 0; i < N; i++) {
            sum += A[output_row * N + i] *
                   B[i * K + output_col];
        }

        C[output_row * K + output_col] = sum;
    }
}

extern "C" void solve(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid(
        (K + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A, B, C, M, N, K
    );

    cudaDeviceSynchronize();
}