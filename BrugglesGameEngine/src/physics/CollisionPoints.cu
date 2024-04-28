#include "physics/CollisionPoints.cuh"

namespace bruggles {
    namespace physics {
        __host__ __device__ CollisionPoints CollisionPoints::Flip() {
            Vector2 temp = this->B;
            this->B = this->A;
            this->A = temp;
            this->Normal = -this->Normal;
            return *this;
        }
    }
}