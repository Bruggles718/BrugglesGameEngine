#include "physics/CollisionObject.cuh"
#include <iostream>

namespace bruggles {
    namespace physics {
        __host__ __device__ CollisionObject::CollisionObject() {};

        __host__ __device__ Transform& CollisionObject::GetTransform() {
            return m_transform;
        }

        Transform& CollisionObject::GetLastTransform() {
            return m_lastTransform;
        }

        void CollisionObject::SetTransform(Transform* tf) {
            m_transform.Position = tf->Position;
            m_transform.Rotation = tf->Rotation;
            m_transform.Scale = tf->Scale;
        }

        void CollisionObject::UpdateLastTransform() {
            m_lastTransform.Position = m_transform.Position;
            m_lastTransform.Rotation = m_transform.Rotation;
            m_lastTransform.Scale = m_transform.Scale;
        }

        std::pair<Vector2, Vector2> CollisionObject::TopLeftBottomRightAABB() {
            if (!collider) {
                std::cout << "NO COLLIDER FOUND" << std::endl;
                return std::pair<Vector2, Vector2>();
            }

            Vector2 top = collider->FindFurthestPoint(&GetTransform(), Vector2(0, -1));
            Vector2 left = collider->FindFurthestPoint(&GetTransform(), Vector2(-1, 0));
            Vector2 bottom = collider->FindFurthestPoint(&GetTransform(), Vector2(0, 1));
            Vector2 right = collider->FindFurthestPoint(&GetTransform(), Vector2(1, 0));
            Vector2 topLeft{ left.x, top.y };
            Vector2 bottomRight{ right.x, bottom.y };
            return std::pair<Vector2, Vector2>(topLeft, bottomRight);
        }
    }
}