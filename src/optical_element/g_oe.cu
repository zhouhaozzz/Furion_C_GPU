#include "G_Oe.h"
#include "g_beam.h"

using namespace Furion_NS;

G_Oe::G_Oe(G_Beam* beam_in, real_t ds, real_t di, real_t chi, real_t theta, No_Surfe* surface, Grating* grating)
    : beam_in(beam_in), grating(grating), surface(surface), theta(theta), chi(chi)//, beam_out(beam_in), Cff(0), theta2(0)
{
    cout << "G_Oe 初始化" << endl;
}

G_Oe::~G_Oe()
{
    delete[] X_, Y_, PHI, PSI, Phase;
    delete[] L1, M1, N1, X1, Y1, Z1, X2, Y2, Z2, cos_Alpha;
    cout << "~G_Oe的析构" << endl;

}

__global__ void Furion_NS::g_oe_cuda(real_t* X_, real_t* Y_, real_t* Phase, real_t* X3, real_t* Y3, real_t* Z3, real_t* L3, real_t* M3, real_t* N3, real_t* T, real_t* Z2, real_t lambda, real_t m, real_t n0, real_t b, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    real_t T1;

    if (i < n)
    {
        T1 = -Z3[i] / N3[i];
        X_[i] = X3[i] + T1 * L3[i];
        Y_[i] = Y3[i] + T1 * M3[i];
        Phase[i] = (T[i] + T1) / lambda * 2 * Pi - n0 * m * 2 * Pi * Z2[i] - 0.5 * m * b * n0 * 2 * Pi * (Z2[i] * Z2[i]);
    }

    __syncthreads();
}

void G_Oe::g_oe_GPU(real_t* X3, real_t* Y3, real_t* Z3, real_t* L3, real_t* M3, real_t* N3, real_t* T, real_t* Z2, real_t lambda, real_t m, real_t n0, real_t b)
{
    int n = Furion::n;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

    real_t* d_X_, * d_Y_, * d_Phase;
    real_t* d_X3, * d_Y3, * d_Z3;
    real_t* d_L3, * d_M3, * d_N3, * d_T, * d_Z2;
    cudaMalloc((void**)&d_X3, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Y3, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Z3, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_L3, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_M3, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_N3, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_X3, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_X_, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Y_, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_T, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Z2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Phase, Furion::n * sizeof(real_t));

    cudaMemcpy(d_L3, L3, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M3, M3, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N3, N3, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X3, X3, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y3, Y3, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z3, Z3, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z2, Z2, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T, T, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);

    Furion_NS::g_oe_cuda << <blocksPerGrid, threadsPerBlock >> > (d_X_, d_Y_, d_Phase, d_X3, d_Y3, d_Z3, d_L3, d_M3, d_N3, d_T, d_Z2, lambda, m, n0, b, n);

    cudaMemcpy(this->X_, d_X_, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->Y_, d_Y_, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->Phase, d_Phase, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);

    cudaFree(d_X_);
    cudaFree(d_Y_);
    cudaFree(d_Phase);
    cudaFree(d_X3);
    cudaFree(d_Y3);
    cudaFree(d_Z3);
    cudaFree(d_N3);
    cudaFree(d_L3);
    cudaFree(d_M3);
    cudaFree(d_N3);
    cudaFree(d_Z2);
    cudaFree(d_T);
}

void G_Oe::reflect(G_Beam* beam_in, real_t ds, real_t di, real_t chi, real_t theta)
{
    real_t* L = new real_t[Furion::n];
    real_t* M = new real_t[Furion::n];
    real_t* N = new real_t[Furion::n];
    real_t* T = new real_t[Furion::n];
    real_t* T1 = new real_t[Furion::n];
    real_t* Nx = new real_t[Furion::n];
    real_t* Ny = new real_t[Furion::n];
    real_t* Nz = new real_t[Furion::n];
    real_t* hslope = new real_t[Furion::n];
    real_t* L2 = new real_t[Furion::n];
    real_t* M2 = new real_t[Furion::n];
    real_t* N2 = new real_t[Furion::n];
    real_t* X3 = new real_t[Furion::n];
    real_t* Y3 = new real_t[Furion::n];
    real_t* Z3 = new real_t[Furion::n];
    real_t* L3 = new real_t[Furion::n];
    real_t* M3 = new real_t[Furion::n];
    real_t* N3 = new real_t[Furion::n];
    
    f_a_v.Furion_angle_vector(beam_in->phi, beam_in->psi, L, M, N);               //[phi,psi]-&gt; [L,M,N] angles are converted to unit vectors
    
    source_to_oe(beam_in->XX, beam_in->YY, ds, L, M, N);           //From light source coordinate system to optical component coordinate system
    
    intersection(T);                                      //The intersection of light and optical components
    
    normal(Nx, Ny, Nz);                                   //mormal Find reflection vector

    h_slope(hslope, this->Y2);      //Calculate the surface slope error    Find the slope of the corresponding position

    this->theta2 = Pi / 2 - asin(sin(Pi / 2 - theta) - grating->n0 * grating->m * grating->lambda_G);
    this->Cff = cos(Pi / 2 - this->theta2) / cos(Pi / 2 - this->theta);

    f_r_v.Furion_reflect_Vector(this->cos_Alpha, L2, M2, N2, this->L1, this->M1, this->N1, Nx, Ny, Nz, grating->lambda_G, grating->m, grating->n0, grating->b, this->Z2, hslope, this->Cff);

    oe_to_image(X3, Y3, Z3, L3, M3, N3, this->X2, this->Y2, this->Z2, di, L2, M2, N2);

    g_oe_GPU(X3, Y3, Z3, L3, M3, N3, T, Z2, grating->lambda_G, grating->m, grating->n0, grating->b);

    f_v_a.Furion_vector_angle(this->PHI, this->PSI, L3, M3);
    
    beam_out = new G_Beam((this->X_), (this->Y_), (this->PHI), (this->PSI), beam_in->lambda);
    
    delete[] L, M, N;
    delete[] T, T1;
    delete[] Nx, Ny, Nz;
    delete[] hslope;
    delete[] L2, M2, N2;
    delete[] X3, Y3, Z3, L3, M3, N3;
}

void G_Oe::source_to_oe(real_t* X, real_t* Y, real_t ds, real_t* L, real_t* M, real_t* N)
{
    int n = Furion::n;
    real_t* OS = new real_t[9];
    real_t* OS_0 = new real_t[9];
    real_t* OS_1 = new real_t[9];
    f_rx.furion_rotx(theta, OS_0);
    f_rz.furion_rotz(chi, OS_1);

    real_t* Z = new real_t[1];
    Z[0] = -ds;

    matrixMulti_33(OS, OS_0, OS_1);
    G_Oe::matrixMulti_3n(this->X1, this->Y1, this->Z1, OS, X, Y, Z, 0, n);

    matrixMulti_3nn(this->L1, this->M1, this->N1, OS, L, M, N, 0, n);

    delete[] Z, OS, OS_0, OS_1;
    cout << " G_Oe的source_to_oe" << endl;
}

void G_Oe::matrixMulti_33(real_t* matrix, real_t* matrix1, real_t* matrix2)  //XYZ:1*3; LMN:1*n
{
    for (int i = 0; i < 3; i++)
    {
        matrix[i] = matrix1[0] * matrix2[i] + matrix1[1] * matrix2[i + 3] + matrix1[2] * matrix2[i + 6];
        matrix[i + 3] = matrix1[3] * matrix2[i] + matrix1[4] * matrix2[i + 3] + matrix1[5] * matrix2[i + 6];
        matrix[i + 6] = matrix1[6] * matrix2[i] + matrix1[7] * matrix2[i + 3] + matrix1[8] * matrix2[i + 6];
    }

    //__syncthreads();
}

__global__ void Furion_NS::matrixMulti_3n_cuda(real_t* L2, real_t* M2, real_t* N2, real_t* matrix, real_t* L, real_t* M, real_t* N, real_t dx, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        L2[i] = matrix[0] * L[i] + matrix[1] * M[i] + matrix[2] * N[0];
        M2[i] = matrix[3] * L[i] + matrix[4] * M[i] + matrix[5] * N[0];
        N2[i] = matrix[6] * L[i] + matrix[7] * M[i] + matrix[8] * N[0] + dx;
    }

    __syncthreads();
}

void G_Oe::matrixMulti_3n(real_t* L2, real_t* M2, real_t* N2, real_t* matrix, real_t* L, real_t* M, real_t* N, real_t dx, int n)  //XYZ:1*3; LMN:1*n
{
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

    real_t* d_L2, * d_M2, * d_N2, * d_L, * d_M, * d_N, * d_matrix;
    cudaMalloc((void**)&d_L2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_M2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_N2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_L, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_M, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_N, 1 * sizeof(real_t));
    cudaMalloc((void**)&d_matrix, 9 * sizeof(real_t));

    cudaMemcpy(d_L, L, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, 1 * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix, matrix, 9 * sizeof(real_t), cudaMemcpyHostToDevice);

    Furion_NS::matrixMulti_3n_cuda << <blocksPerGrid, threadsPerBlock >> > (d_L2, d_M2, d_N2, d_matrix, d_L, d_M, d_N, dx, n);

    cudaMemcpy(L2, d_L2, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(M2, d_M2, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(N2, d_N2, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);

    cudaFree(d_L2);
    cudaFree(d_M2);
    cudaFree(d_N2);
    cudaFree(d_L);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_matrix);
}

__global__ void Furion_NS::matrixMulti_3nn_cuda(real_t* L2, real_t* M2, real_t* N2, real_t* matrix, real_t* L, real_t* M, real_t* N, real_t dx, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        L2[i] = matrix[0] * L[i] + matrix[1] * M[i] + matrix[2] * N[i];
        M2[i] = matrix[3] * L[i] + matrix[4] * M[i] + matrix[5] * N[i];
        N2[i] = matrix[6] * L[i] + matrix[7] * M[i] + matrix[8] * N[i] + dx;
    }

    __syncthreads();
}

void G_Oe::matrixMulti_3nn(real_t* L2, real_t* M2, real_t* N2, real_t* matrix, real_t* L, real_t* M, real_t* N, real_t dx, int n)  //XYZ:1*3; LMN:1*n
{
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

    real_t* d_L2, * d_M2, * d_N2, * d_L, * d_M, * d_N, * d_matrix;
    cudaMalloc((void**)&d_L2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_M2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_N2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_L, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_M, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_N, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_matrix, 9 * sizeof(real_t));

    cudaMemcpy(d_L, L, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix, matrix, 9 * sizeof(real_t), cudaMemcpyHostToDevice);

    Furion_NS::matrixMulti_3nn_cuda << <blocksPerGrid, threadsPerBlock >> > (d_L2, d_M2, d_N2, d_matrix, d_L, d_M, d_N, dx, n);

    cudaMemcpy(L2, d_L2, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(M2, d_M2, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(N2, d_N2, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);

    cudaFree(d_L2);
    cudaFree(d_M2);
    cudaFree(d_N2);
    cudaFree(d_L);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_matrix);
}

void G_Oe::matrixMulti(real_t* L2, real_t* M2, real_t* N2, real_t* matrix, real_t* L, real_t* M, real_t* N, int n)  //XYZ:1*3; LMN:1*n
{
    for (int i = 0; i < n; i++)
    {
        L2[i] = matrix[0] * L[i] + matrix[1] * M[i] + matrix[2] * N[i];
        M2[i] = matrix[3] * L[i] + matrix[4] * M[i] + matrix[5] * N[i];
        N2[i] = matrix[6] * L[i] + matrix[7] * M[i] + matrix[8] * N[i];
    }

    //__syncthreads();
}

__global__ void Furion_NS::intersection_cuda(real_t* T, real_t* X2, real_t* Y2, real_t* Z2, real_t* X1, real_t* Y1, real_t* Z1, real_t* L1, real_t* M1, real_t* N1, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        T[i] = Y1[i] / M1[i];
        X2[i] = X1[i] + T[i] * L1[i];
        Y2[i] = 0;
        Z2[i] = Z1[i] + T[i] * N1[i];
    }

    __syncthreads();
}


void G_Oe::intersection(real_t* T)
{
    int n = Furion::n;

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

    real_t* d_X2, * d_Y2, * d_Z2;
    real_t* d_X1, * d_Y1, * d_Z1;
    real_t* d_L1, * d_M1, * d_N1, * d_T;
    cudaMalloc((void**)&d_X2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Y2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Z2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_L1, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_M1, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_N1, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_X1, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Y1, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Z1, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_T, Furion::n * sizeof(real_t));

    cudaMemcpy(d_L1, L1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M1, M1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N1, N1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X1, X1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y1, Y1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z1, Z1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X2, X2, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y2, Y2, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z2, Z2, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);

    Furion_NS::intersection_cuda << <blocksPerGrid, threadsPerBlock >> > (d_T, d_X2, d_Y2, d_Z2, d_X1, d_Y1, d_Z1, d_L1, d_M1, d_N1, n);

    cudaMemcpy(T, d_T, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->X2, d_X2, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->Y2, d_Y2, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->Z2, d_Z2, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);

    cudaFree(d_X1);
    cudaFree(d_Y1);
    cudaFree(d_Z1);
    cudaFree(d_X2);
    cudaFree(d_Y2);
    cudaFree(d_Z2);
    cudaFree(d_L1);
    cudaFree(d_M1);
    cudaFree(d_N1);
    cudaFree(d_T);

    cout << " G_Oe的intersection" << endl;

}

__global__ void Furion_NS::normal_cuda(real_t* Nx, real_t* Ny, real_t* Nz, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n)
    {
        Ny[i] = 0;
        Nz[i] = 0;
        Nx[i] = 0;
    }

    __syncthreads();
}


void G_Oe::normal(real_t* Nx, real_t* Ny, real_t* Nz)
{
    int n = Furion::n;

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

    real_t* d_Nx, * d_Ny, * d_Nz;

    cudaMalloc((void**)&d_Nx, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Ny, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Nz, Furion::n * sizeof(real_t));

    Furion_NS::normal_cuda << <blocksPerGrid, threadsPerBlock >> > (d_Nx, d_Ny, d_Nz, n);

    cudaMemcpy(Nx, d_Nx, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(Ny, d_Ny, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(Nz, d_Nz, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);

    cudaFree(d_Nx);
    cudaFree(d_Ny);
    cudaFree(d_Nz);

    cout << "G_Oe的normal" << endl;
}

__global__ void Furion_NS::h_slope1_cuda(real_t* delta_X, real_t* delta_Z, real_t* L1, real_t* N1, real_t* X2, real_t* Z2, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    real_t delta_Z1;

    if (i < n)
    {
        delta_Z1 = sqrt(L1[i] * L1[i] + N1[i] * N1[i]);
        delta_Z[i] = 1e-10 * N1[i] * delta_Z1 + Z2[i];
        delta_X[i] = 1e-10 * L1[i] * delta_Z1 + X2[i];
    }

    __syncthreads();
}

__global__ void Furion_NS::h_slope2_cuda(real_t* h_slope, real_t* Y2, real_t* h0, real_t* h1, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n)
    {
        h_slope[i] = (h1[i] - h0[i]) / 1e-10;
        Y2[i] = h0[i] + Y2[i];
    }

    __syncthreads();
}

void G_Oe::h_slope(real_t* h_slope, real_t* Y2)
{
    int n = Furion::n;

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

    real_t* h0 = new real_t[Furion::n];
    real_t* h1 = new real_t[Furion::n];
    real_t* delta_Z = new real_t[Furion::n];
    real_t* delta_X = new real_t[Furion::n];

    real_t* d_L1, * d_N1, * d_X2, * d_Y2, * d_Z2;
    real_t* d_h0, * d_h1, * d_h_slope;
    real_t* d_delta_X, * d_delta_Z;

    cudaMalloc((void**)&d_L1, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_N1, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_X2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Y2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_Z2, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_h0, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_h1, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_h_slope, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_delta_X, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_delta_Z, Furion::n * sizeof(real_t));

    cudaMemcpy(d_L1, this->L1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N1, this->N1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X2, this->X2, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y2, this->Y2, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z2, this->Z2, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);

    surface->value(h0, this->Z2, this->X2, n);
    cudaMemcpy(d_h0, h0, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    //cudaFree(d_h0);

    Furion_NS::h_slope1_cuda << <blocksPerGrid, threadsPerBlock >> > (d_delta_X, d_delta_Z, d_L1, d_N1, d_X2, d_Z2, n);
    cudaFree(d_L1);
    cudaFree(d_N1);
    cudaFree(d_X2);
    cudaFree(d_Z2);

    cudaMemcpy(delta_X, d_delta_X, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_Z, d_delta_Z, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaFree(d_delta_X);
    cudaFree(d_delta_Z);

    surface->value(h1, delta_Z, delta_X, n);

    cudaMemcpy(d_h1, h1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    Furion_NS::h_slope2_cuda << <blocksPerGrid, threadsPerBlock >> > (d_h_slope, d_Y2, d_h0, d_h1, n);
    cudaMemcpy(h_slope, d_h_slope, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->Y2, d_Y2, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);

    cudaFree(d_h0);
    cudaFree(d_h1);
    cudaFree(d_Y2);
    cudaFree(d_h_slope);

    delete[] h0, h1, delta_Z, delta_X;
    cout << " G_Oe的h_slope" << endl;
}

void G_Oe::oe_to_image(real_t* X3, real_t* Y3, real_t* Z3, real_t* L3, real_t* M3, real_t* N3, real_t* X2, real_t* Y2, real_t* Z2, real_t di, real_t* L2, real_t* M2, real_t* N2)
{
    int n = Furion::n;

    real_t* OS = new real_t[9];
    real_t* OS_0 = new real_t[9];
    real_t* OS_1 = new real_t[9];

    real_t* X0 = new real_t[Furion::n];
    real_t* Y0 = new real_t[Furion::n];
    real_t* Z0 = new real_t[Furion::n];

    f_rx.furion_rotx(this->theta2, OS_0);
    G_Oe::matrixMulti_3nn(X3, Y3, Z3, OS_0, X2, Y2, Z2, -di, n);

    f_rz.furion_rotz(-1 * this->chi, OS_0);
    f_rx.furion_rotx(this->theta2, OS_1);
    G_Oe::matrixMulti_33(OS, OS_0, OS_1);
    G_Oe::matrixMulti_3nn(L3, M3, N3, OS, L2, M2, N2, 0, n);

    delete[] X0, Y0, Z0, OS, OS_0, OS_1;
    cout << " G_Oe的oe_to_image" << endl;
}
