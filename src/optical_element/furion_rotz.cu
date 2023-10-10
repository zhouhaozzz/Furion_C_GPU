#include "Furion_Rotz.h"

using namespace Furion_NS;

Furion_Rotz::Furion_Rotz()
{

}

Furion_Rotz::~Furion_Rotz()
{

}

void Furion_Rotz::furion_rotz(real_t t, real_t* matrix)
{
    real_t ct = cos(t * Pi / 180.0);
    real_t st = sin(t * Pi / 180.0);
    real_t data[9] = { ct, -st, 0, st, ct, 0, 0, 0, 1 };

    for (int i = 0; i < 9; i++)
    {
        matrix[i] = data[i];
    }
}
