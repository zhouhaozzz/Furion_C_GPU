#include "Furion_Vector_Angle.h"

using namespace Furion_NS;

Furion_Vector_Angle::Furion_Vector_Angle()
{

}

Furion_Vector_Angle::~Furion_Vector_Angle()
{

}

__global__ void Furion_NS::Furion_vector_angle_cuda(real_t* Phi, real_t* Psi, real_t* L, real_t* M, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n)
    {
        Psi[i] = asin(M[i]);
        Phi[i] = asin(L[i] / cos(Psi[i]));
    }

    __syncthreads();
}

void Furion_Vector_Angle::Furion_vector_angle(real_t* Phi, real_t* Psi, real_t* L, real_t* M)
{
    int n = Furion::n;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

    //real_t* d_L, * d_M;
    //cudaMalloc((void**)&d_L, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_M, Furion::n * sizeof(real_t));
    //cudaMemcpy(d_L, L, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_M, M, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);

    real_t* d_phi, * d_psi;
    cudaMalloc((void**)&d_phi, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_psi, Furion::n * sizeof(real_t));

    Furion_NS::Furion_vector_angle_cuda << <blocksPerGrid, threadsPerBlock >> > (d_phi, d_psi, L, M, n);

    cudaMemcpy(Phi, d_phi, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(Psi, d_psi, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);

    cudaFree(d_phi);
    cudaFree(d_psi);
    //cudaFree(d_L);
    //cudaFree(d_M);
}