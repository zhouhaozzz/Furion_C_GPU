#include "G_Cylinder_Ellipse.h"

using namespace Furion_NS;

G_Cylinder_Ellipse::G_Cylinder_Ellipse(G_Beam* beam_in, real_t ds, real_t di, real_t chi, real_t theta, No_Surfe* surface, real_t r1, real_t r2, Grating* grating)
    : G_Oe(beam_in, ds, di, chi, theta, surface, grating)
{
    this->r1 = r1;
    this->e = sqrt(r1 * r1 + r2 * r2 - 2 * r1 * r2 * cos(Pi - 2 * theta)) / 2;
    this->alpha = acos((r1 * r1 + 4 * this->e * this->e - r2 * r2) / (4 * r1 * this->e));
    this->beta = acos((r2 * r2 + 4 * this->e * this->e - r1 * r1) / (4 * r2 * this->e));
    this->a = (r1 + r2) / 2;
    this->b = sqrt(this->a * this->a - this->e * this->e);
    //cout << this->r1 << " " << e << " " << alpha << " " << beta << " " << a << " " << b<<endl;
}

G_Cylinder_Ellipse::~G_Cylinder_Ellipse()
{
    cout << "~G_Cylinder_Ellipse的析构" << endl;
    delete[] T, Nx, Ny, Nz;
}

void G_Cylinder_Ellipse::run(G_Beam* beam_in, real_t ds, real_t di, real_t chi, real_t theta, No_Surfe* surface, real_t r1, real_t r2, Grating* grating)
{
    reflect(beam_in, ds, di, chi, theta);
    //cout << center->T[0] << endl;

    cout << "G_Cylinder_Ellipse的run" << endl;

}

void G_Cylinder_Ellipse::source_to_oe(real_t* X1, real_t* Y1, real_t* Z1, real_t* L1, real_t* M1, real_t* N1, real_t* X, real_t* Y, real_t ds, real_t* L, real_t* M, real_t* N)
{
    int n = Furion::n;

    real_t* OS = new real_t[9];
    real_t* OS_0 = new real_t[9];
    real_t* OS_1 = new real_t[9];
    f_rx.furion_rotx(this->alpha, OS_0);
    f_rz.furion_rotz(this->chi, OS_1);

    real_t* Z = new real_t[1];
    Z[0] = this->r1 - ds;

    G_Oe::matrixMulti_33(OS, OS_0, OS_1);
    G_Oe::matrixMulti_3n(X1, Y1, Z1, OS, X, Y, Z, -this->e, n);
    G_Oe::matrixMulti_3nn(L1, M1, N1, OS, L, M, N, 0, n);

    //Furion_rotx(obj.alpha)* (Furion_rotz(obj.chi) * [X; Y; repmat(obj.r1 - ds, 1, n)]) - repmat([0; 0; obj.e], 1, n);

    delete[] Z, OS, OS_0, OS_1;
    cout << "G_Cylinder_Ellipse的source_to_oe" << endl;
}

__global__ void Furion_NS::intersection_cuda_GCE(real_t* T, real_t* X2, real_t* Y2, real_t* Z2, real_t* X1, real_t* Y1, real_t* Z1, real_t* L1, real_t* M1, real_t* N1, real_t a, real_t b, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    real_t A;
    real_t B;
    real_t C;

    real_t a2 = a * a;
    real_t b2 = b * b;
    real_t ab2 = a2 * b2;

    if (i < n)
    {
        A = b2 * N1[i] * N1[i] + a2 * M1[i] * M1[i];
        B = 2 * (b2 * N1[i] * Z1[i] + a2 * M1[i] * Y1[i]);
        C = a2 * Y1[i] * Y1[i] + b2 * Z1[i] * Z1[i] - ab2;
        T[i] = (-B + sqrt(B * B - 4 * A * C)) / (2 * A);

        X2[i] = X1[i] + T[i] * L1[i];
        Y2[i] = Y1[i] + T[i] * M1[i];
        Z2[i] = Z1[i] + T[i] * N1[i];
    }

    __syncthreads();
}

void G_Cylinder_Ellipse::intersection(real_t* T, real_t* X2, real_t* Y2, real_t* Z2, real_t* X1, real_t* Y1, real_t* Z1, real_t* L1, real_t* M1, real_t* N1)
{
    int n = Furion::n;

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

    //real_t* d_X2, * d_Y2, * d_Z2;
    //real_t* d_X1, * d_Y1, * d_Z1;
    //real_t* d_L1, * d_M1, * d_N1, * d_T;
    //cudaMalloc((void**)&d_X2, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_Y2, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_Z2, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_L1, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_M1, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_N1, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_X1, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_Y1, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_Z1, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_T, Furion::n * sizeof(real_t));

    //cudaMemcpy(d_L1, this->L1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_M1, this->M1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_N1, this->N1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_X1, this->X1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_Y1, this->Y1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_Z1, this->Z1, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);

    Furion_NS::intersection_cuda_GCE << <blocksPerGrid, threadsPerBlock >> > (T, X2, Y2, Z2, X1, Y1, Z1, L1, M1, N1, this->a, this->b, n);

    cudaMemcpy(this->T, T, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(this->T, d_T, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(this->X2, d_X2, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(this->Y2, d_Y2, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(this->Z2, d_Z2, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);

    //cudaFree(d_X1);
    //cudaFree(d_Y1);
    //cudaFree(d_Z1);
    //cudaFree(d_X2);
    //cudaFree(d_Y2);
    //cudaFree(d_Z2);
    //cudaFree(d_L1);
    //cudaFree(d_M1);
    //cudaFree(d_N1);
    //cudaFree(d_T);

    cout << "G_Cylinder_Ellipse的intersection" << endl;
}

__global__ void Furion_NS::normal_cuda_GCE(real_t* Nx, real_t* Ny, real_t* Nz, real_t* X2, real_t* Y2, real_t* Z2, real_t a, real_t b, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    real_t a2 = a * a;
    real_t b2 = b * b;
    real_t a4 = a2 * a2;
    real_t b4 = b2 * b2;
    //real_t ab2 = a2 * b2;
    real_t data;

    if (i < n)
    {
        data = sqrt(Y2[i] * Y2[i] / b4 + Z2[i] * Z2[i] / a4);
        Ny[i] = -(Y2[i] / b2) / data;
        Nz[i] = -(Z2[i] / a2) / data;
        Nx[i] = 0;
    }

    __syncthreads();
}

void G_Cylinder_Ellipse::normal(real_t* X2, real_t* Y2, real_t* Z2, real_t* Nx, real_t* Ny, real_t* Nz)
{
    int n = Furion::n;

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

    //real_t* d_X2, * d_Y2, * d_Z2;
    //real_t* d_Nx, * d_Ny, * d_Nz;
    //cudaMalloc((void**)&d_X2, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_Y2, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_Z2, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_Nx, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_Ny, Furion::n * sizeof(real_t));
    //cudaMalloc((void**)&d_Nz, Furion::n * sizeof(real_t));

    //cudaMemcpy(d_X2, this->X2, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_Y2, this->Y2, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_Z2, this->Z2, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);

    Furion_NS::normal_cuda_GCE << <blocksPerGrid, threadsPerBlock >> > (Nx, Ny, Nz, X2, Y2, Z2, this->a, this->b, n);

    //cudaMemcpy(Nx, d_Nx, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(Ny, d_Ny, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(Nz, d_Nz, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->Nx, Nx, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->Ny, Ny, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->Nz, Nz, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);

    //cudaFree(d_X2);
    //cudaFree(d_Y2);
    //cudaFree(d_Z2);
    //cudaFree(d_Nx);
    //cudaFree(d_Ny);
    //cudaFree(d_Nz);

    cout << "G_Cylinder_Ellipse的normal" << endl;
}

__global__ void Furion_NS::h_slope_cuda_GCE(real_t* h_slope, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n)
    {
        h_slope[i] = 0;
    }
}

void G_Cylinder_Ellipse::h_slope(real_t* X2, real_t* Y2, real_t* Z2, real_t* L1, real_t* N1, real_t* h_slope)
{
    int n = Furion::n;

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

    //real_t* d_h_slope;
    //cudaMalloc((void**)&d_h_slope, Furion::n * sizeof(real_t));

    Furion_NS::h_slope_cuda_GCE << <blocksPerGrid, threadsPerBlock >> > (h_slope, n);

    //cudaMemcpy(h_slope, d_h_slope, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);

    //cudaFree(d_h_slope);

    cout << " G_Cylinder_Ellipse的h_slope" << endl;
}