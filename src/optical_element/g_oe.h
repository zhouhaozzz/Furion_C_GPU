#pragma once

#ifndef FUR_G_OE_H_
#define FUR_G_OE_H_

#include "Furion.h"
#include "g_beam.h"
#include "grating.h"
#include "no_surfe.h"
#include "furion_angle_vector.h"
#include "furion_rotx.h"
#include "furion_rotz.h"
#include "furion_reflect_vector.h"
#include "furion_vector_angle.h"

namespace Furion_NS
{

    class G_Oe
    {
    public:
        real_t* X_ = new real_t[Furion::n];                     //Output beam X coordinates
        real_t* Y_ = new real_t[Furion::n];                     //Output beam Y coordinates
        real_t* PSI = new real_t[Furion::n];                   //Output beam PHI Angle
        real_t* PHI = new real_t[Furion::n];                   //Output beam PSI Angle
        real_t* Phase = new real_t[Furion::n];                 //Output parameter
        real_t* L1 = new real_t[Furion::n];          //Beam line direction in the optical component coordinate system [L1 M1 N1]
        real_t* M1 = new real_t[Furion::n];          //Beam line direction in the optical component coordinate system [L1 M1 N1]
        real_t* N1 = new real_t[Furion::n];          //Beam line direction in the optical component coordinate system [L1 M1 N1]
        real_t* X1 = new real_t[Furion::n];          //Beam position [X1,Y1,Z1] in the optical component coordinate system unit m
        real_t* Y1 = new real_t[Furion::n];          //Beam position [X1,Y1,Z1] in the optical component coordinate system unit m
        real_t* Z1 = new real_t[Furion::n];          //Beam position [X1,Y1,Z1] in the optical component coordinate system unit m
        real_t* X2 = new real_t[Furion::n];
        real_t* Y2 = new real_t[Furion::n];
        real_t* Z2 = new real_t[Furion::n];
        real_t* cos_Alpha = new real_t[Furion::n];

        int threadsPerBlock = BLOCK_SIZE;
        int blocksPerGrid = (Furion::n + threadsPerBlock - 1) / threadsPerBlock;

        real_t theta;       //Grazing Angle of incidence
        real_t chi;         //Direction of optical components
        real_t Cff;
        real_t theta2;

        G_Beam* beam_out;
        G_Beam* beam_in;
        Grating* grating;
        No_Surfe* surface;

        G_Oe(G_Beam* beam_in, real_t ds, real_t di, real_t chi, real_t theta, No_Surfe* surface, Grating* grating);
        virtual ~G_Oe();

        class Furion_Angle_Vector f_a_v;
        class Furion_Rotx f_rx;
        class Furion_Rotz f_rz;
        class Furion_Reflect_Vector f_r_v;
        class Furion_Vector_Angle f_v_a;

        void reflect(G_Beam* beam_in, real_t ds, real_t di, real_t chi, real_t theta);
        virtual void source_to_oe(real_t* X1, real_t* Y1, real_t* Z1, real_t* L1, real_t* M1, real_t* N1, real_t* X, real_t* Y, real_t ds, real_t* L, real_t* M, real_t* N);
        void matrixMulti(real_t* L2, real_t* M2, real_t* N2, real_t* matrix, real_t* L, real_t* M, real_t* N, int n);  //XYZ:1*3; LMN:1*n
        void matrixMulti_33(real_t* matrix, real_t* matrix1, real_t* matrix2);  //XYZ:1*3; LMN:1*n
        virtual void matrixMulti_3n(real_t* L2, real_t* M2, real_t* N2, real_t* matrix, real_t* L, real_t* M, real_t* N, real_t dx, int n);  //XYZ:1*3; LMN:1*n
        virtual void matrixMulti_3nn(real_t* L2, real_t* M2, real_t* N2, real_t* matrix, real_t* L, real_t* M, real_t* N, real_t dx, int n);  //XYZ:1*3; LMN:1*n
        virtual void intersection(real_t* T, real_t* X2, real_t* Y2, real_t* Z2, real_t* X1, real_t* Y1, real_t* Z1, real_t* L1, real_t* M1, real_t* N1);
        virtual void normal(real_t* X2, real_t* Y2, real_t* Z2, real_t* Nx, real_t* Ny, real_t* Nz);
        virtual void h_slope(real_t* X2, real_t* Y2, real_t* Z2, real_t* L1, real_t* N1, real_t* h_slope);
        void oe_to_image(real_t* X3, real_t* Y3, real_t* Z3, real_t* L3, real_t* M3, real_t* N3, real_t* X2, real_t* Y2, real_t* Z2, real_t di, real_t* L2, real_t* M2, real_t* N2);
        void g_oe_GPU(real_t* X_, real_t* Y_, real_t* Phase, real_t* X3, real_t* Y3, real_t* Z3, real_t* L3, real_t* M3, real_t* N3, real_t* T, real_t* Z2, real_t lambda, real_t m, real_t n0, real_t b);

    };

    __global__ void g_oe_cuda(real_t* X_, real_t* Y_, real_t* Phase, real_t* X3, real_t* Y3, real_t* Z3, real_t* L3, real_t* M3, real_t* N3, real_t* T, real_t* Z2, real_t lambda, real_t m, real_t n0, real_t b, int n);
    __global__ void matrixMulti_3n_cuda(real_t* L2, real_t* M2, real_t* N2, real_t* matrix, real_t* L, real_t* M, real_t* N, real_t dx, int n);
    __global__ void matrixMulti_3nn_cuda(real_t* L2, real_t* M2, real_t* N2, real_t* matrix, real_t* L, real_t* M, real_t* N, real_t dx, int n);
    __global__ void intersection_cuda(real_t* T, real_t* X2, real_t* Y2, real_t* Z2, real_t* X1, real_t* Y1, real_t* Z1, real_t* L1, real_t* M1, real_t* N1, int n);
    __global__ void normal_cuda(real_t* Nx, real_t* Ny, real_t* Nz, int n);
    __global__ void h_slope1_cuda(real_t* delta_X, real_t* delta_Z, real_t* L1, real_t* N1, real_t* X2, real_t* Z2, int n);
    __global__ void h_slope2_cuda(real_t* h_slope, real_t* Y2, real_t* h0, real_t* h1, int n);
}
#endif

//ofstream fileout("data.dat");
//fileout << std::fixed;
//fileout << std::setprecision(15);
//for (int i = 0; i < Furion::n; i++)
//{
//    fileout << this->X_[i] << " ";
//}
//fileout << "\n";
//
//for (int i = 0; i < Furion::n; i++)
//{
//    fileout << this->Y_[i] << " ";
//}
//fileout << "\n";
//
//for (int i = 0; i < Furion::n; i++)
//{
//    fileout << this->PHI[i] << " ";
//}
//fileout << "\n";
//for (int i = 0; i < Furion::n; i++)
//{
//    fileout << this->PSI[i] << " ";
//}
//fileout << "\n";
//
//fileout.close();
//
//exit(0);

//real_t sum = 0;
//real_t sum1 = 0;
//real_t sum2 = 0;
//for (int i = 0; i < Furion::n; i++)
//{
//    sum = sum + fabs(T[i] + T1[i]);
//    sum1 = sum1 + fabs(this->Phase[i]);
//    sum2 = sum2 + fabs(this->Phase[i]);
//}
//cout << sum << " " << sum1 << " " << sum2 << endl;
