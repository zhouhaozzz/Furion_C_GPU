#pragma once

#ifndef FUR_G_CYLINDER_ELLIPSE_H_
#define FUR_G_CYLINDER_ELLIPSE_H_

#include "Furion.h"
#include "g_oe.h"

namespace Furion_NS
{

    class G_Cylinder_Ellipse : public G_Oe       //The tracking phase converts to Angle
    {
    public:

        real_t* T = new real_t[Furion::n];                     //Elliptic parameter
        real_t* Nx = new real_t[Furion::n];          //In center coordinates, the normal vector
        real_t* Ny = new real_t[Furion::n];          //In center coordinates, the normal vector
        real_t* Nz = new real_t[Furion::n];          //In center coordinates, the normal vector

        real_t alpha;
        real_t a;
        real_t b;
        real_t beta;
        real_t r1;
        real_t e;

        G_Cylinder_Ellipse(G_Beam* beam_in, real_t ds, real_t di, real_t chi, real_t theta, No_Surfe* surface, real_t r1, real_t r2, Grating* grating);
        ~G_Cylinder_Ellipse();

        void run(G_Beam* beam_in, real_t ds, real_t di, real_t chi, real_t theta, No_Surfe* surface, real_t r1, real_t r2, Grating* grating);

        void intersection(real_t* T) override;
        void source_to_oe(real_t* X, real_t* Y, real_t ds, real_t* L, real_t* M, real_t* N) override;
        void normal(real_t* Nx, real_t* Ny, real_t* Nz) override;
        void h_slope(real_t* h_slope, real_t* Y2) override;
    };

    __global__ void intersection_cuda_GCE(real_t* T, real_t* X2, real_t* Y2, real_t* Z2, real_t* X1, real_t* Y1, real_t* Z1, real_t* L1, real_t* M1, real_t* N1, real_t a, real_t b, int n);
    __global__ void normal_cuda_GCE(real_t* Nx, real_t* Ny, real_t* Nz, real_t* X2, real_t* Y2, real_t* Z2, real_t a, real_t b, int n);
    __global__ void h_slope_cuda_GCE(real_t* h_slope, int n);
}

#endif



