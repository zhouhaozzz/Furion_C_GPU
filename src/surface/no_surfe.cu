#include "no_surfe.h"

using namespace Furion_NS;

No_Surfe::No_Surfe() : meri_X(nullptr), sag_Y(nullptr), V(nullptr), adress(nullptr)
{
    cout << "surfe" << endl;
}

No_Surfe::~No_Surfe()
{

}

__global__ void Furion_NS::value_cuda(real_t* Vq, real_t* Z, real_t* X, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n)
    {
        Vq[i] = 0;
    }
}

void No_Surfe::value(real_t* Vq, real_t* Z, real_t* X, int n)
{
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

    //real_t* d_Vq, * d_X, * d_Z;

    //cudaMalloc((void**)&d_Vq, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_X, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_Z, Furion::n * sizeof(real_t));

    Furion_NS::value_cuda << <blocksPerGrid, threadsPerBlock >> > (Vq, Z, X, n);

    //cudaMemcpy(Vq, d_Vq, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);

    //cudaFree(d_Vq);
    //cudaFree(d_X);
    //cudaFree(d_Z);
}