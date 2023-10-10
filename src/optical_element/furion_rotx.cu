#include "Furion_Rotx.h"

using namespace Furion_NS;

Furion_Rotx::Furion_Rotx()
{

}

Furion_Rotx::~Furion_Rotx()
{

}

void Furion_Rotx::furion_rotx(real_t t, real_t* matrix)
{
    real_t ct = cos(t);
    real_t st = sin(t);
    real_t data[9] = { 1, 0, 0, 0, ct, -st, 0, st, ct };

    for (int i = 0; i < 9; i++)
    {
        matrix[i] = data[i];
    }
}

