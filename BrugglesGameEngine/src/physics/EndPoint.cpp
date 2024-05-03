#include "physics/EndPoint.hpp"

namespace bruggles {
	namespace physics {
		EndPoint::EndPoint() {
			object = nullptr;
			id = 0;
			value = 0.0f;
			isMin = false;
		}

		EndPoint::EndPoint(CollisionObject* i_object, Uint64 i_id, float i_value, bool i_isMin) {
			object = i_object;
			id = i_id;
			value = i_value;
			isMin = i_isMin;
		}
	}
}