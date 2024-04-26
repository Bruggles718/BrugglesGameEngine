#include "physics/CollisionPoints.cuh"

namespace bruggles {
    namespace physics {
        __host__ __device__ CollisionPoints::CollisionPoints() {
            this->HasCollision = false;
        }

        __host__ __device__ CollisionPoints::CollisionPoints(Vector2& i_A, Vector2& i_B) {
            this->HasCollision = true;
            this->A = i_A;
            this->B = i_B;
            this->Normal = (this->A - this->B).Normalized();
            this->Depth = (this->A - this->B).Magnitude();
        }

        __host__ __device__ CollisionPoints CollisionPoints::Flip() {
            Vector2 temp = this->B;
            this->B = this->A;
            this->A = temp;
            this->Normal = -this->Normal;
            return *this;
        }
    }
}