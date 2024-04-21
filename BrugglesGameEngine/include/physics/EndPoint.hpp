#pragma once
#include "physics/CollisionObject.hpp"

namespace bruggles {
	namespace physics {
        struct EndPoint {
            CollisionObject* object;
            Uint64 id = 0;
            float value = 0.0f;
            bool isMin = false;
        };

        void BinaryInsert(std::vector<EndPoint>& points, EndPoint& pointToInsert);
	}
}