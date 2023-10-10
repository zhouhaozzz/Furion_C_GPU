#pragma once

#ifndef FUR_FURION_ANGLE_VECTOR_H_
#define FUR_FURION_ANGLE_VECTOR_H_

#include "Furion.h"

namespace Furion_NS
{
    class Furion_Angle_Vector
    {
    private:
        /* data */
    public:
        Furion_Angle_Vector();
        ~Furion_Angle_Vector();

        void Furion_angle_vector(real_t* Phi, real_t* Psi, real_t* L, real_t* M, real_t* N);
    };

    __global__ void Cal_FAV(real_t* Phi, real_t* Psi, real_t* L, real_t* M, real_t* N, int n);
}

#endif

