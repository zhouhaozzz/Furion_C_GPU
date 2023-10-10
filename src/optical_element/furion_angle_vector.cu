#include "Furion_Angle_Vector.h"

using namespace Furion_NS;

Furion_Angle_Vector::Furion_Angle_Vector()
{

}

Furion_Angle_Vector::~Furion_Angle_Vector()
{

}

__global__ void Furion_NS::Cal_FAV(real_t* Phi, real_t* Psi, real_t* L, real_t* M, real_t* N, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        L[i] = sin(Phi[i]) * cos(Psi[i]);
        M[i] = sin(Psi[i]);
        N[i] = cos(Phi[i]) * cos(Psi[i]);
    }

    __syncthreads();
}

void Furion_Angle_Vector::Furion_angle_vector(real_t* Phi, real_t* Psi, real_t* L, real_t* M, real_t* N)
{
    int n = Furion::n;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

    real_t* d_phi, * d_psi;
    cudaMalloc((void**)&d_phi, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_psi, Furion::n * sizeof(real_t));
    cudaMemcpy(d_phi, Phi, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_psi, Psi, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);

    real_t* d_L, * d_M, * d_N;
    cudaMalloc((void**)&d_L, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_M, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_N, Furion::n * sizeof(real_t));

    Furion_NS::Cal_FAV << <blocksPerGrid, threadsPerBlock >> > (d_phi, d_psi, d_L, d_M, d_N, n);

    cudaMemcpy(L, d_L, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(M, d_M, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(N, d_N, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    //
    //cout << M[0] << endl;
    //std::cin.get();
    cudaFree(d_phi);
    cudaFree(d_psi);
    cudaFree(d_L);
    cudaFree(d_M);
    cudaFree(d_N);

}




