#pragma once

#ifndef FUR_FURION_ROTZ_H_
#define FUR_FURION_ROTZ_H_

#include "Furion.h"

namespace Furion_NS
{
    class Furion_Rotz
    {
    public:

        Furion_Rotz();
        ~Furion_Rotz();
        void furion_rotz(real_t t, real_t* matrix);
    };
}

#endif
