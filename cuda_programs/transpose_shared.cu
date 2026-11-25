#include<cuda_runtime.h>
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_shared(float *output, float *input, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    ## compute the thread id -- we need to take block idx and get the thread id within the block and then compute the global thread id

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}
// Host launch:
// transpose_shared<<<(cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM>>>(output, input, rows, cols);
// Note: The above launch configuration is just a placeholder. You need to adjust the grid and block dimensions based on your specific matrix size and the TILE_DIM and BLOCK_ROWS values.
