#include "G_Beam.h"

using namespace Furion_NS;

G_Beam::G_Beam(real_t* XX, real_t* YY, real_t* phi, real_t* psi, real_t lambda) :
    XX(XX), YY(YY), phi(phi), psi(psi), n(Furion::n), lambda(lambda)
{
    cout << "g_beam ³õÊ¼»¯" << endl;
}

G_Beam::~G_Beam()
{
    //delete XX, YY, psi, phi;   
}

__global__ void Furion_NS::G_Beam_cuda(real_t* XX, real_t* YY, real_t* phi, real_t* psi, real_t distance, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n)
    {
        XX[i] = XX[i] + distance * tan(phi[i]) * cos(psi[i]);
        YY[i] = YY[i] + distance * tan(psi[i]);
    }

    __syncthreads();
}

G_Beam G_Beam::translate(real_t distance)
{
    int n = Furion::n;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

    real_t* d_XX, * d_YY, * d_phi, * d_psi;
    cudaMalloc((void**)&d_XX, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_YY, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_phi, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_psi, Furion::n * sizeof(real_t));
    cudaMemcpy(d_XX, XX, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_YY, YY, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi, phi, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_psi, psi, Furion::n * sizeof(real_t), cudaMemcpyHostToDevice);

    Furion_NS::G_Beam_cuda << <blocksPerGrid, threadsPerBlock >> > (d_XX, d_YY, d_phi, d_psi, distance, n);

    cudaMemcpy(XX, d_XX, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(YY, d_YY, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_phi);
    cudaFree(d_psi);
    cudaFree(d_XX);
    cudaFree(d_YY);

    //for (int i = 0; i < n; i++)
    //{
    //    XX[i] = XX[i] + distance * tan(phi[i]) * cos(psi[i]);
    //    YY[i] = YY[i] + distance * tan(psi[i]);
    //}
    cout << "G_BeamµÄtranslate" << endl;
    
    return G_Beam(XX, YY, phi, psi, lambda);
}

void G_Beam::plot_sigma(real_t distance, int rank1)
{
    G_Beam beam = translate(distance);
    f_p_s.Furion_plot_sigma(beam.XX, beam.YY, beam.phi, beam.psi, rank1);
}


