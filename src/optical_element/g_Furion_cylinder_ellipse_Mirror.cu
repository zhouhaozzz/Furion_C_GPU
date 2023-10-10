#include "g_Furion_cylinder_ellipse_Mirror.h"

using namespace Furion_NS;

G_Furion_Cylinder_Ellipse_Mirror::G_Furion_Cylinder_Ellipse_Mirror(G_Beam* beam_in, real_t ds, real_t di, real_t chi, real_t theta, No_Surfe* surface, real_t r1, real_t r2, Grating* grating)
    : G_Oe(beam_in, ds, di, chi, theta, surface, grating)//, center(beam_in, ds, di, chi, theta, surface, r1, r2, grating)
{
    cout << "G_Furion_Cylinder_Ellipse_Mirror 初始化" << endl;
}

G_Furion_Cylinder_Ellipse_Mirror::~G_Furion_Cylinder_Ellipse_Mirror()
{
    cout << "~G_Furion_Cylinder_Ellipse_Mirror的析构" << endl;
}

void G_Furion_Cylinder_Ellipse_Mirror::run(G_Beam* beam_in, real_t ds, real_t di, real_t chi, real_t theta, No_Surfe* surface, real_t r1, real_t r2, Grating* grating)
{
    cout << "G_Furion_Cylinder_Ellipse_Mirror的run" << endl;

    set_center(beam_in, ds, di, chi, theta, surface, r1, r2, grating);
    
    reflect(beam_in, ds, di, chi, theta);
}

void G_Furion_Cylinder_Ellipse_Mirror::intersection(real_t* T)
{
    int n = Furion::n;

    //for (int i = 0; i < n; i++) { T[i] = center->T[i]; }
    center_to_oe_p(this->X2, this->Y2, this->Z2, center->X2, center->Y2, center->Z2);
    cout << "G_Furion_Cylinder_Ellipse_Mirror的intersection" << endl;


}

void G_Furion_Cylinder_Ellipse_Mirror::normal(real_t* Nx, real_t* Ny, real_t* Nz)
{
    center_to_oe_v(Nx, Ny, Nz, center->Nx, center->Ny, center->Nz);
    cout << "G_Furion_Cylinder_Ellipse_Mirror的normal" << endl;
}

__global__ void Furion_NS::matrixMulti_3n_GFCEM(real_t* L2, real_t* M2, real_t* N2, real_t* matrix, real_t* L, real_t* M, real_t* N, real_t dx, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        L2[i] = matrix[0] * L[i] + matrix[1] * M[i] + matrix[2] * (N[i] + dx);
        M2[i] = matrix[3] * L[i] + matrix[4] * M[i] + matrix[5] * (N[i] + dx);
        N2[i] = matrix[6] * L[i] + matrix[7] * M[i] + matrix[8] * (N[i] + dx);
    }
    __syncthreads();
}

void G_Furion_Cylinder_Ellipse_Mirror::matrixMulti_3n(real_t* L2, real_t* M2, real_t* N2, real_t* matrix, real_t* L, real_t* M, real_t* N, real_t dx, int n)  //XYZ:1*3; LMN:1*n
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
    //cout << "sdasdasdasd" << endl;
    //exit(0);

    cudaMemcpy(d_L, L, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix, matrix, 9 * sizeof(real_t), cudaMemcpyHostToDevice);

    Furion_NS::matrixMulti_3n_GFCEM << <blocksPerGrid, threadsPerBlock >> > (d_L2, d_M2, d_N2, d_matrix, d_L, d_M, d_N, dx, n);

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


void G_Furion_Cylinder_Ellipse_Mirror::center_to_oe_p(real_t* X2, real_t* Y2, real_t* Z2, real_t* X, real_t* Y, real_t* Z)
{
    real_t a1 = center->e;
    real_t alpha1 = center->alpha;
    int n = beam_in->n;

    real_t* OS_0 = new real_t[9];
    real_t* OS_1 = new real_t[9];

    f_rx.furion_rotx(this->theta, OS_0);
    f_rx.furion_rotx(-alpha1, OS_1);

    real_t* X0 = new real_t[Furion::n];
    real_t* Y0 = new real_t[Furion::n];
    real_t* Z0 = new real_t[Furion::n];

    matrixMulti_3n(X0, Y0, Z0, OS_1, X, Y, Z, a1, n);
    matrixMulti_3n(X2, Y2, Z2, OS_0, X0, Y0, Z0, -center->r1, n);

    //Furion_rotx(obj.theta)*((Furion_rotx(-alpha1)*[X;Y;Z+a1])-repmat([0; 0;obj.center.r1],1,n))

    delete[] OS_0, OS_1, X0, Y0, Z0;
}

void G_Furion_Cylinder_Ellipse_Mirror::center_to_oe_v(real_t* Nx, real_t* Ny, real_t* Nz, real_t* L, real_t* M, real_t* N)
{
    real_t alpha1 = center->alpha;
    int n = beam_in->n;

    real_t* OS = new real_t[9];
    real_t* OS_0 = new real_t[9];
    real_t* OS_1 = new real_t[9];
    f_rx.furion_rotx(this->theta, OS_0);
    f_rx.furion_rotx(-alpha1, OS_1);

    matrixMulti_33(OS, OS_0, OS_1);
    G_Oe::matrixMulti_3nn(Nx, Ny, Nz, OS, L, M, N, 0, n);

    delete[] OS, OS_0, OS_1;

}

void G_Furion_Cylinder_Ellipse_Mirror::set_center(G_Beam* beam_in, real_t ds, real_t di, real_t chi, real_t theta, No_Surfe* surface, real_t r1, real_t r2, Grating* grating)
{
    cout << "G_Furion_Cylinder_Ellipse_Mirror的set_center" << endl;
    center = new G_Cylinder_Ellipse(beam_in, ds, di, chi, theta, surface, r1, r2, grating);
    center->run(beam_in, ds, di, chi, theta, surface, r1, r2, grating);
}