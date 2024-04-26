#include "physics/BoxCollider.hpp"

namespace bruggles {
    namespace physics {
        BoxCollider::BoxCollider(const float i_w, const float i_h) {
            float halfWidth = i_w/2;
            float halfHeight = i_h/2;
            Vector2 topLeft{-halfWidth, -halfHeight};
            Vector2 topRight{halfWidth, -halfHeight};
            Vector2 bottomRight{halfWidth, halfHeight};
            Vector2 bottomLeft{-halfWidth, halfHeight};
            this->Vertices.PushBack(topLeft);
            this->Vertices.PushBack(topRight);
            this->Vertices.PushBack(bottomRight);
            this->Vertices.PushBack(bottomLeft);
        }

        void BoxCollider::SetDimensions(const float i_w, const float i_h) {
            this->Vertices.Clear();
            float halfWidth = i_w/2;
            float halfHeight = i_h/2;
            Vector2 topLeft{-halfWidth, -halfHeight};
            Vector2 topRight{halfWidth, -halfHeight};
            Vector2 bottomRight{halfWidth, halfHeight};
            Vector2 bottomLeft{-halfWidth, halfHeight};
            this->Vertices.PushBack(topLeft);
            this->Vertices.PushBack(topRight);
            this->Vertices.PushBack(bottomRight);
            this->Vertices.PushBack(bottomLeft);
        }
    }
}