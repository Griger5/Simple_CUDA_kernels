#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <curand_kernel.h>

const int numOfThreads = 1024;

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// change this code if you want to test a different function
__device__ void function(double x, double *y) {
    *y = pow(2*x+15, 3)/(3*x+1);
}

__global__ void MC_Integral(double *sum, double a, double b, int N) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride =  gridDim.x * blockDim.x;

    __shared__ double sumOfThreads[numOfThreads];
    double x, y;
    sumOfThreads[threadIdx.x] = 0;

    curandState_t rng;
	curand_init(clock64(), index, 0, &rng);

    for (int i=index; i<N; i+=stride) {
        x = curand_uniform_double(&rng) * (b-a) + a;
        function(x, &y);
        sumOfThreads[threadIdx.x] += y;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        sum[blockIdx.x] = 0;
        for (int i = 0; i<numOfThreads; i++) {
            sum[blockIdx.x] += sumOfThreads[i];
        }
    }
}

int main() {
    int deviceId;
    int numberOfSMs = 20;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    int N = 100000000;
    double *sum;

    // calculate the integral in the interval [a,b]
    double a = 0.5;
    double b = 2;

    size_t size = sizeof(double);

    cudaMallocManaged(&sum, size);
    cudaMemPrefetchAsync(sum, size, deviceId);

    int threadsPerBlock = numOfThreads;
    int blocksPerGrid = numberOfSMs;

    MC_Integral<<<blocksPerGrid, threadsPerBlock>>>(sum, a, b, N);

    checkCuda(cudaDeviceSynchronize());

    double result = 0;

    for (int i=0; i<blocksPerGrid; i++) {
        result += sum[i];
    }
    cudaFree(sum);

    printf("%f", (b-a)*result/N);

    return 0;
}