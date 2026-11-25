#include <cuda_runtime.h>

__global__ void softmax(float *input, float *output, int n)
{
    __shared__ float sdata[256];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    // find the maximum value in the input array for numerical stability
    float local_max = -FLT_MAX;
    for (int i = tid; i < n; i += blockDim.x)
    {
        local_max = fmaxf(local_max, input[bid * n + i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] = fmaxf(
                shared[tid],
                shared[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = shared[0];
    // Step 3: Compute sum of exponentials
    float local_sum = 0.0f;

    for (int c = tid; c < cols; c += blockDim.x)
    {
        float x = input[row * cols + c];
        local_sum += expf(x - row_max);
    }

    shared[tid] = local_sum;
    __syncthreads();

    // Step 4: Reduce to denominator
    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float denominator = shared[0];

    // Step 5: Normalize the output
    for (int c = tid; c < cols; c += blockDim.x)
    {
        float x = input[row * cols + c];
        output[row * cols + c] =
            expf(x - row_max) / denominator;
    }
}
// Host launch:
// softmax_rows<<<rows, 256>>>(input, output, cols);