#pragma once

#ifndef FUR_G_FURION__CYLINDER_ELLIPSE_MIRROR_H_
#define FUR_G_FURION__CYLINDER_ELLIPSE_MIRROR_H_

#include "Furion.h"
#include "g_oe.h"
#include "g_cylinder_ellipse.h"

namespace Furion_NS
{
    class G_Furion_Cylinder_Ellipse_Mirror : public G_Oe
    {
    public:

        G_Furion_Cylinder_Ellipse_Mirror(G_Beam* beam_in, real_t ds, real_t di, real_t chi, real_t theta, No_Surfe* surface, real_t r1, real_t r2, Grating* grating);
        ~G_Furion_Cylinder_Ellipse_Mirror();

        void run(G_Beam* beam_in, real_t ds, real_t di, real_t chi, real_t theta, No_Surfe* surface, real_t r1, real_t r2, Grating* grating);

        G_Cylinder_Ellipse* center;

        void intersection(real_t* T) override;
        void normal(real_t* Nx, real_t* Ny, real_t* Nz) override;
        void matrixMulti_3n(real_t* L2, real_t* M2, real_t* N2, real_t* matrix, real_t* L, real_t* M, real_t* N, real_t dx, int n) override;  //XYZ:1*3; LMN:1*n
        void center_to_oe_p(real_t* X2, real_t* Y2, real_t* Z2, real_t* X, real_t* Y, real_t* Z);
        void center_to_oe_v(real_t* Nx, real_t* Ny, real_t* Nz, real_t* L, real_t* M, real_t* N);
        virtual void set_center(G_Beam* beam_in, real_t ds, real_t di, real_t chi, real_t theta, No_Surfe* surface, real_t r1, real_t r2, Grating* grating);
    };

    __global__ void matrixMulti_3n_GFCEM(real_t* L2, real_t* M2, real_t* N2, real_t* matrix, real_t* L, real_t* M, real_t* N, real_t dx, int n);
}

#endif



