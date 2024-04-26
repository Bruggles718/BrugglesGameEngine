#pragma once

#include "physics/HullCollider.cuh"

namespace bruggles {
    namespace physics {
        struct BoxCollider : public HullCollider {
            BoxCollider(const float i_w, const float i_h);
            void SetDimensions(const float i_w, const float i_h);
        };
    }
}