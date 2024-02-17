#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

void initMatrix(int *a, int N, int min, int max) {
    for (int i=0; i<N; i++) {
        a[i] = rand() % (max - min + 1) + min;
    }
}

__global__ void matrixMultiply(int *a, int *b, int *c, int N) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < N && col < N) {
        c[row * N + col] = 0;
        for(int i=0; i<N; ++i) {
            c[row * N + col] += a[row * N + i] * b[col + i * N];
        }
    }
}

void cpu_matrixMulti(int *a, int *b, int *c, int N) {
    for( int row = 0; row < N; ++row ) {
        for( int col = 0; col < N; ++col ) {
            c[row * N + col] = 0;
            for ( int k = 0; k < N; ++k ) {
                c[row * N + col] += a[row * N + k] * b[k * N + col];
            }
        }
    }
}

void matrixMultiTest(int *a, int *b, int N) {
    bool error = false;
    for( int row = 0; row < N && !error; ++row ) {
        for( int col = 0; col < N && !error; ++col ) {
            if (a[row * N + col] != b[row * N + col]) {
                printf("FOUND ERROR at b[%d][%d]\n", row, col);
                error = true;
                break;
            }
        }
    }
    if (!error)
    printf("Success!\n");
}

int main() {
    srand(time(NULL));
    
    int N = 1024;
    int *a, *b, *c;
    int size = N * N * sizeof(int);

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    initMatrix(a, N*N, 0, 10);
    initMatrix(b, N*N, 0, 10);
    initMatrix(c, N*N, 0, 0);

    dim3 threadsPerBlock (32, 32, 1);
    dim3 blocksPerGrid ((N / threadsPerBlock.x)+1, (N / threadsPerBlock.y)+1, 1);

    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, N);
    
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

// uncomment to test the gpu multiplication
/*
    int *d;
    cudaMallocManaged(&d, size);
    initMatrix(d, N*N, 0, 0);
    cpu_matrixMulti(a, b, d, N);
    matrixMultiTest(d, c, N);    
    cudaFree(d);
*/

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}