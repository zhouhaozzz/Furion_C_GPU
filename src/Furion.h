#pragma once

#ifndef FUR_FURION_H_
#define FUR_FURION_H_

#include <iostream>
#include <math.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <cstdio>
#include <cmath>
#include <random>

#ifdef CUDA
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#endif

#ifdef HIGH_PRECISION
typedef double real_t;
#else
typedef float real_t;
#endif

//#include <mpi.h>

using namespace std;

#define Pi 3.1415926536
#define E  2.71828
#define BLOCK_SIZE 256


namespace Furion_NS
{
    class Furion
    {
    public:
        class Grating* grating;            // Grate setting
        class G_Source* g_source;
        class G_Beam* g_beam;
        class No_Surfe* no_surfe;
        class G_Furion_Cylinder_Ellipse_Mirror* g_Furion_cylinder_ellipse_Mirror;
        class G_Furion_ellipsoid_Mirror* g_Furion_ellipsoid_Mirror;

        Furion(int rank1, int size1);
        ~Furion();

        size_t i = 0;
        const static int n = 100000;
        real_t Lambda[5] = { 1, 1.55, 2, 2.5, 3 };
        real_t pre_Mirror_theta[5] = { 0.82448, 1.0266, 1.1662, 1.3039, 1.4285 };
        real_t grating_theta[5] = { 6.397, 7.9674, 9.0524, 10.124, 11.093 };
        real_t beamsize[5] = { 0.113155108251141, 0.134122084191793, 0.137907788181078, 0.139653456385901, 0.145485139180038 };
        real_t divergence[5] = { 3.9, 4.9731, 6.6307, 8.2884, 9.1 };
        real_t lambda, psigmax, psigmay, vsigmax, vsigmay;
    };
}

#endif