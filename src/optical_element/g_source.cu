#include "g_source.h"
#include "g_beam.h"
#include <chrono>
using namespace Furion_NS;

G_Source::G_Source(real_t sigma_beamsize, real_t sigma_divergence, int n, real_t lambda, int rank1) : beam_out(XX, YY, phi, psi, lambda)
{
    //int n = Furion::n;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

    real_t* d_XX, * d_YY, * d_phi, * d_psi;
    //unsigned long long seed = 1234;
    cudaMalloc((void**)&d_XX, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_YY, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_phi, Furion::n * sizeof(real_t));
    cudaMalloc((void**)&d_psi, Furion::n * sizeof(real_t));
    
    //Get_Time(1);
    Furion_NS::normrnd_cuda << <blocksPerGrid, threadsPerBlock >> > (d_XX, 0, sigma_beamsize, n, Get_Time(1), rank1);// Normal random number
    Furion_NS::normrnd_cuda << <blocksPerGrid, threadsPerBlock >> > (d_YY, 0, sigma_beamsize, n, Get_Time(200), rank1);
    Furion_NS::normrnd_cuda << <blocksPerGrid, threadsPerBlock >> > (d_phi, 0, sigma_divergence, n, Get_Time(30), rank1);
    Furion_NS::normrnd_cuda << <blocksPerGrid, threadsPerBlock >> > (d_psi, 0, sigma_divergence, n, Get_Time(4000), rank1);

    cudaMemcpy(this->XX, d_XX, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->YY, d_YY, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->phi, d_phi, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->psi, d_psi, Furion::n * sizeof(real_t), cudaMemcpyDeviceToHost);
    //cout << Get_Time(1) << " " << Get_Time(20) << endl;
    cudaFree(d_XX);
    cudaFree(d_YY);
    cudaFree(d_phi);
    cudaFree(d_psi);

    beam_out = G_Beam(this->XX, this->YY, this->phi, this->psi, lambda);
    cout << "g_source的初始化" << endl;
}

G_Source::~G_Source()
{
    delete XX, YY, psi, phi;
}

unsigned int G_Source::Get_Time(int n1)
{
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    long long int timestamp = duration.count();
    unsigned int seed = static_cast<unsigned int>(n1) + static_cast<unsigned int>(timestamp);

    return seed;
}

__global__ void Furion_NS::normrnd_cuda(real_t* resultArray, real_t mu, real_t sigma_beamsize, int n, unsigned int seed, int rank1)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    curandState state;
    curand_init(seed, i, 0, &state); // 初始化随机数发生器状态

    if (i < n) 
    {
        resultArray[i] = mu + sigma_beamsize * curand_normal(&state); // 生成正态分布随机数
    }

    __syncthreads();
}

void G_Source::normrnd(real_t* resultArray, real_t mu, real_t sigma_beamsize, int n, int n1, int rank1)
{
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    long long int timestamp = duration.count();
    unsigned int seed = static_cast<unsigned int>(rank1 + n1) + static_cast<unsigned int>(timestamp);
    std::default_random_engine generator(seed);
    std::normal_distribution<real_t> distribution(mu, sigma_beamsize);

    // Generate random numbers
    for (int i = 0; i < n; ++i) {
        resultArray[i] = distribution(generator);
    }
}
