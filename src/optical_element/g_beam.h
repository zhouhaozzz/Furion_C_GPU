#pragma once

#ifndef FUR_G_BEAM_H_
#define FUR_G_BEAM_H_

#include "Furion.h"
#include "Furion_Plot_Sigma.h"

namespace Furion_NS
{
    class G_Beam
    {
    public:

        real_t lambda;          //wave length
        int n;
        real_t* XX = new real_t[Furion::n];
        real_t* YY = new real_t[Furion::n];
        real_t* psi = new real_t[Furion::n];
        real_t* phi = new real_t[Furion::n];

        G_Beam(real_t* XX, real_t* YY, real_t* phi, real_t* psi, real_t lambda);
        ~G_Beam();

        class Furion_Plot_Sigma f_p_s;

        G_Beam translate(real_t distance);
        void plot_sigma(real_t distance, int rank1);
    };

    __global__ void G_Beam_cuda(real_t* XX, real_t* YY, real_t* phi, real_t* psi, real_t distance, int n);
}

#endif

