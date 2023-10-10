#pragma once

#ifndef FUR_NO_SURFE_H_
#define FUR_NO_SURFE_H_

#include "Furion.h"

namespace Furion_NS
{
    class No_Surfe
    {
    public:
        real_t* meri_X = new real_t[Furion::n]; //Meridian coordinates
        real_t* sag_Y = new real_t[Furion::n]; //Sagittal direction coordinates
        real_t* V = new real_t[Furion::n]; //Height profile error value
        real_t* adress = new real_t[Furion::n];

        No_Surfe();
        ~No_Surfe();

        void value(real_t* Vq, real_t* Z, real_t* X, int n);
    };

    __global__ void value_cuda(real_t* Vq, real_t* Z, real_t* X, int n);
}

#endif




