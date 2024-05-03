#pragma once
#include <cstdint>

typedef uint64_t Uint64;

namespace bruggles {
    namespace physics {
        struct CollisionObject;
        
        struct EndPoint {

            EndPoint();

            EndPoint(CollisionObject* i_object, Uint64 i_id, float i_value, bool i_isMin);

            CollisionObject* object = nullptr;
            Uint64 id = 0;
            float value = 0.0f;
            bool isMin = false;
        };
    }
}
