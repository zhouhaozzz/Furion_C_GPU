#include "Furion_Reflect_Vector.h"

using namespace Furion_NS;

Furion_Reflect_Vector::Furion_Reflect_Vector()
{

}

Furion_Reflect_Vector::~Furion_Reflect_Vector()
{

}

void Furion_Reflect_Vector::Furion_reflect_Vector(real_t* cos_Alpha, real_t* L2, real_t* M2, real_t* N2, real_t* L1, real_t* M1, real_t* N1, real_t* Nx, real_t* Ny, real_t* Nz, real_t lambda, real_t m, real_t n0, real_t b, real_t* Z2, real_t* h_slope, real_t Cff)
{
    int n = Furion::n;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

    matrixCross(cos_Alpha, L2, M2, N2, L1, M1, N1, Nx, Ny, Nz, lambda, m, n0, b, Z2, h_slope, Cff);

    //delete[] t_Base_x, t_Base_y, t_Base_z, t_Base_model;
    //delete[] sin_Alpha, sin_Beta, cos_Beta;
}

__global__ void Furion_NS::matrixCross_cuda(real_t* L2, real_t* M2, real_t* N2, real_t* Nx, real_t* Ny, real_t* Nz, real_t* L1, real_t* M1, real_t* N1, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n)
    {
        L2[i] = N1[i] * Ny[i] - M1[i] * Nz[i];
        M2[i] = -N1[i] * Nx[i] + L1[i] * Nz[i];
        N2[i] = M1[i] * Nx[i] - L1[i] * Ny[i];
    }

    __syncthreads();
}

__global__ void Furion_NS::t_Base_cuda(real_t* t_Base_x, real_t* t_Base_y, real_t* t_Base_z, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    real_t t_Base_model;

    if (i < n)
    {
        t_Base_model = sqrt(t_Base_x[i] * t_Base_x[i] + t_Base_y[i] * t_Base_y[i] + t_Base_z[i] * t_Base_z[i]);
        t_Base_x[i] = t_Base_x[i] / t_Base_model;
        t_Base_y[i] = t_Base_y[i] / t_Base_model;
        t_Base_z[i] = t_Base_z[i] / t_Base_model;
    }

    __syncthreads();
}

__global__ void Furion_NS::Furion_reflect_Vector_cuda(real_t* cos_Alpha, real_t* L2, real_t* M2, real_t* N2, real_t* t_Base_x, real_t* t_Base_y, real_t* t_Base_z, real_t* L1, real_t* M1, real_t* N1, real_t* Nx, real_t* Ny, real_t* Nz, real_t lambda, real_t m, real_t n0, real_t b, real_t* Z2, real_t* h_slope, real_t Cff, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    real_t sin_Alpha;
    real_t sin_Beta;
    real_t cos_Beta;

    if (i < n)
    {
        cos_Alpha[i] = fabs(L1[i] * Nx[i] + M1[i] * Ny[i] + N1[i] * Nz[i]);
        sin_Alpha = sqrt(1 - cos_Alpha[i] * cos_Alpha[i]);
        sin_Beta = sin_Alpha - m * (n0 * (1 + b * Z2[i])) * lambda - (1 + Cff) * h_slope[i] * cos_Alpha[i];
        cos_Beta = sqrt(1 - sin_Beta * sin_Beta);

        L2[i] = cos_Beta * Nx[i] + sin_Beta * t_Base_x[i];
        M2[i] = cos_Beta * Ny[i] + sin_Beta * t_Base_y[i];
        N2[i] = cos_Beta * Nz[i] + sin_Beta * t_Base_z[i];
    }
}

void Furion_Reflect_Vector::matrixCross(real_t* cos_Alpha, real_t* L2, real_t* M2, real_t* N2, real_t* L1, real_t* M1, real_t* N1, real_t* Nx, real_t* Ny, real_t* Nz, real_t lambda, real_t m, real_t n0, real_t b, real_t* Z2, real_t* h_slope, real_t Cff)  //There is no matrix cross function in the Eigen library, and it can only be implemented by custom
{
    int n = Furion::n;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

    real_t* d_cos_Alpha, * d_Z2, * d_h_slope;
    real_t* d_L1, * d_M1, * d_N1;
    real_t* d_L2, * d_M2, * d_N2;
    real_t* d_Nx, * d_Ny, * d_Nz;
    real_t* d_t_Base_x, * d_t_Base_y, * d_t_Base_z;

    cudaMalloc((void**)&d_cos_Alpha, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Z2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_h_slope, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_L1, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_M1, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_N1, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_L2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_M2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_N2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Nx, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Ny, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Nz, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_t_Base_x, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_t_Base_y, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_t_Base_z, Furion::n * sizeof(real_t));

    //cudaMemcpy(d_cos_Alpha, cos_Alpha, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z2, Z2, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h_slope, h_slope, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L1, L1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M1, M1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N1, N1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Nx, Nx, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ny, Ny, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Nz, Nz, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);

    Furion_NS::matrixCross_cuda << <blocksPerGrid, threadsPerBlock >> > (d_L2, d_M2, d_N2, d_Nx, d_Ny, d_Nz, d_L1, d_M1, d_N1, n);

    Furion_NS::matrixCross_cuda << <blocksPerGrid, threadsPerBlock >> > (d_t_Base_x, d_t_Base_y, d_t_Base_z, d_L2, d_M2, d_N2, d_Nx, d_Ny, d_Nz, n);

    Furion_NS::t_Base_cuda << <blocksPerGrid, threadsPerBlock >> > (d_t_Base_x, d_t_Base_y, d_t_Base_z, n);

    Furion_NS::Furion_reflect_Vector_cuda << <blocksPerGrid, threadsPerBlock >> > (d_cos_Alpha, d_L2, d_M2, d_N2, d_t_Base_x, d_t_Base_y, d_t_Base_z, d_L1, d_M1, d_N1, d_Nx, d_Ny, d_Nz, lambda, m, n0, b, d_Z2, d_h_slope, Cff, n);

    cudaMemcpy(cos_Alpha, d_cos_Alpha, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(L2, d_L2, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(M2, d_M2, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(N2, d_N2, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);

    cudaFree(d_cos_Alpha);
    cudaFree(d_Z2);
    cudaFree(d_h_slope);
    cudaFree(d_L1);
    cudaFree(d_M1);
    cudaFree(d_N1);
    cudaFree(d_L2);
    cudaFree(d_M2);
    cudaFree(d_N2);
    cudaFree(d_Nx);
    cudaFree(d_Ny);
    cudaFree(d_Nz);
    cudaFree(d_t_Base_x);
    cudaFree(d_t_Base_y);
    cudaFree(d_t_Base_z);
}