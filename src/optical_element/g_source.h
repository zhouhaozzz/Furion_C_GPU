#pragma once

#ifndef FUR_G_SOURCE_H_
#define FUR_G_SOURCE_H_

#include "Furion.h"
#include "g_beam.h"

namespace Furion_NS
{
    class G_Source
    {
    private:
        /* data */
    public:
        class G_Beam beam_out;
        real_t lambda;          //wave length
        int n;
        real_t* XX = new real_t[Furion::n];
        real_t* YY = new real_t[Furion::n];
        real_t* psi = new real_t[Furion::n];
        real_t* phi = new real_t[Furion::n];

        G_Source(real_t sigma_beamsize, real_t sigma_divergence, int n, real_t lambda, int rank1);
        ~G_Source();

        void normrnd(real_t* resultArray, real_t mu, real_t sigma_beamsize, int n, int n1, int rank1);
        unsigned int Get_Time(int n1);
    };

    __global__ void normrnd_cuda(real_t* resultArray, real_t mu, real_t sigma_beamsize, int n, unsigned int n1, int rank1);
}

#endif

