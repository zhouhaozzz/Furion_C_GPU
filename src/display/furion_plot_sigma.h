#pragma once

#ifndef FUR_FURION_PLOT_SIGMA_H_
#define FUR_FURION_PLOT_SIGMA_H_

#include "Furion.h"

namespace Furion_NS
{
    class Furion_Plot_Sigma
    {
    public:

        Furion_Plot_Sigma();
        ~Furion_Plot_Sigma();

        void Furion_plot_sigma(real_t* X, real_t* Y, real_t* Phi, real_t* Psi, int rank1);
        string inttoStr(int s);
    };
}

#endif



