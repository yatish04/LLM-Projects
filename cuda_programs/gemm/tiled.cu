#include<cuda_runtime.h>
#define TILE_DIM 16
/**
 * By using shared memory, we can reduce the number of global memory access
 * So A shared memory loads every tileth element A
 * and B shared memory loads every tileth element B
 * we wait until all threads have loaded
 * then we compute the sum of the products of the elements in the tile
 * 
 * So the reduction hre is that every thread_x and thread_y need not load
 * the entire row and column of A and B which are redundant. 
 */

__global__ void matrix_multiply_tiled(const float* A, const float*B, const float* C, int M, int N, int K){
    int output_col = blockDim.x*blockIdx.x+threadIdx.x;
    int output_row =  blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    float sum = 0.0f;
    int num_tiles = (N + TILE_DIM - 1) / TILE_DIM;
    if(output_col<K && output_row<M){
       // this one thread will load the row of A and column of B into shared memory
       for(int tile=0; tile<num_tiles; tile++){
        int a_col = tile*TILE_DIM + threadIdx.x;
        int b_row = tile*TILE_DIM + threadIdx.y;
        if(a_col<N && output_row<M){
            As[threadIdx.y][threadIdx.x] = A[output_row*N + a_col];
        }else{
            As[threadIdx.y][threadIdx.x] = 0.0f;    
       }
       if(b_row<N && output_col<K){
            Bs[threadIdx.y][threadIdx.x] = B[b_row*K + output_col];
       }else{
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
       }
         __syncthreads();
         for(int k=0; k<TILE_DIM; k++){
            sum += As[threadIdx.y][k]*Bs[k][threadIdx.x];
         }
         __syncthreads();
    }
    if(output_col<K && output_row<M){
        C[output_row*K + output_col] = sum;
    }
}