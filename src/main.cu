#include "Furion.h"
#include <chrono>
// #include <mpi.h>

#ifdef CUDA
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#endif

using namespace Furion_NS;

__global__ void device(float*phys, int NThreads)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < NThreads)
    {
        phys[i + 0] = 1*10+i;
    }
}

void host(float* phys, int NThreads)
{
    const static int kk = 100;
    int threadsPerBlock = 128;
    int blocksPerGrid = (kk + threadsPerBlock - 1) / threadsPerBlock;

    device << <blocksPerGrid, threadsPerBlock >> > (phys, kk);
}

int main(int argc, char* argv[])
{
    auto start = std::chrono::high_resolution_clock::now();
    srand((unsigned)time(NULL));

    int rank = 0;
    int size = 1;

    // MPI_Init(0, 0);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &size);

    #ifdef CUDA
    //cout << "cuda" << endl;
    //const static int kk = 100;
    //float* phy = new float[kk];

    //float* d_data;
    //cudaMalloc((void**)&d_data, (100) * sizeof(float));
    //host(d_data, 100);
    //cudaMemcpy(phy, d_data, (kk) * sizeof(float), cudaMemcpyDeviceToHost);
    //cout << phy[1] << endl;

    ////std::cin.get();
    //cudaFree(d_data);

    auto furion = new Furion(rank, size);

    #endif

    //auto furion = new Furion(rank, size);

    //delete furion;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Total time: " << duration.count() / 1e6 << " seconds" << std::endl;

    // MPI_Finalize();

    return 0;
}
