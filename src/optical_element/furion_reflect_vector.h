#pragma once

#ifndef FUR_FURION_REFLECT_VECTOR_H_
#define FUR_FURION_REFLECT_VECTOR_H_

#include "Furion.h"

namespace Furion_NS
{
    class Furion_Reflect_Vector
    {
    public:

        Furion_Reflect_Vector();
        ~Furion_Reflect_Vector();

        void Furion_reflect_Vector(real_t* cos_Alpha, real_t* L2, real_t* M2, real_t* N2, real_t* L1, real_t* M1, real_t* N1, real_t* Nx, real_t* Ny, real_t* Nz, real_t lambda, real_t m, real_t n0, real_t b, real_t* Z2, real_t* h_slope, real_t Cff);
        void matrixCross(real_t* cos_Alpha, real_t* L2, real_t* M2, real_t* N2, real_t* L1, real_t* M1, real_t* N1, real_t* Nx, real_t* Ny, real_t* Nz, real_t lambda, real_t m, real_t n0, real_t b, real_t* Z2, real_t* h_slope, real_t Cff);  //There is no matrix cross function in the Eigen library, and it can only be implemented by custom    
    };

    __global__ void Furion_reflect_Vector_cuda(real_t* cos_Alpha, real_t* L2, real_t* M2, real_t* N2, real_t* t_Base_x, real_t* t_Base_y, real_t* t_Base_z, real_t* L1, real_t* M1, real_t* N1, real_t* Nx, real_t* Ny, real_t* Nz, real_t lambda, real_t m, real_t n0, real_t b, real_t* Z2, real_t* h_slope, real_t Cff, int n);
    __global__ void matrixCross_cuda(real_t* L2, real_t* M2, real_t* N2, real_t* Nx, real_t* Ny, real_t* Nz, real_t* L1, real_t* M1, real_t* N1, int n);
    __global__ void t_Base_cuda(real_t* t_Base_x, real_t* t_Base_y, real_t* t_Base_z, int n);
}

#endif
