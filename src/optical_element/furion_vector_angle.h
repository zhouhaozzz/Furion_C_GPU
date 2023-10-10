#pragma once

#ifndef FUR_FURION_VECTOR_ANGLE_H_
#define FUR_FURION_VECTOR_ANGLE_H_

#include "Furion.h"

namespace Furion_NS
{
    class Furion_Vector_Angle
    {
    public:
        Furion_Vector_Angle();
        ~Furion_Vector_Angle();

        void Furion_vector_angle(real_t* Phi, real_t* Psi, real_t* L, real_t* M);
    };

    __global__ void Furion_vector_angle_cuda(real_t* Phi, real_t* Psi, real_t* L, real_t* M, int n);
}

#endif

//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
