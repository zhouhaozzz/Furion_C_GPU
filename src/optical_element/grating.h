#pragma once

#ifndef FUR_GRATING_H_
#define FUR_GRATING_H_

#include "Furion.h"

namespace Furion_NS
{
    class Grating
    {
    private:
        /* data */
    public:
        real_t n0;          //Grating linear density
        real_t b;           //Variable line distance parameter
        real_t m;           //Diffraction order
        real_t lambda_G;    //Grating wavelength

        Grating(real_t n0, real_t b, real_t m, real_t lambda_G);
        ~Grating();
    };
}

#endif


