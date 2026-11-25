#include<cuda_runtime.h>

__global__ void reduce(float *g_idata, float *g_odata, unsigned int n) {
     __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = input[tid]

    __syncthreads()

    for(int x=127; x>=0;x>>1){
        if (tid<x)
        {
            sdata[tid] += sdata[tid+s]; 
        }
        __syncthreads()
        
    }
    if(tid==0){
        output[0] = sdata[0];
    }


}
