#include "physics/CollisionObject.hpp"
#include <iostream>

namespace bruggles {
    namespace physics {


        EndPoint* CollisionObject::GetTop() {
            return m_top.get();
        }
        EndPoint* CollisionObject::GetLeft() {
            return m_left.get();
        }
        EndPoint* CollisionObject::GetBottom() {
            return m_bottom.get();
        }
        EndPoint* CollisionObject::GetRight() {
            return m_right.get();
        }

        Transform& CollisionObject::GetTransform() {
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

        void CollisionObject::SetEndPoint(EndPoint* i_e, float i_value, bool isMin) {
            i_e->isMin = isMin;
            i_e->value = i_value;
        }

        void CollisionObject::UpdateTopLeftBottomRightAABB() {
            if (!collider) {
                std::cout << "NO COLLIDER FOUND" << std::endl;
                return;
            }
            if (!m_top) {
                m_top = std::make_shared<EndPoint>(this, this->m_uniqueID, 0.0f, true);
            }
            if (!m_left) {
                m_left = std::make_shared<EndPoint>(this, this->m_uniqueID, 0.0f, true);
            }
            if (!m_bottom) {
                m_bottom = std::make_shared<EndPoint>(this, this->m_uniqueID, 0.0f, false);
            }
            if (!m_right) {
                m_right = std::make_shared<EndPoint>(this, this->m_uniqueID, 0.0f, false);
            }

            Vector2 top = collider->FindFurthestPoint(&GetTransform(), Vector2(0, -1));
            Vector2 left = collider->FindFurthestPoint(&GetTransform(), Vector2(-1, 0));
            Vector2 bottom = collider->FindFurthestPoint(&GetTransform(), Vector2(0, 1));
            Vector2 right = collider->FindFurthestPoint(&GetTransform(), Vector2(1, 0));
            SetEndPoint(m_top.get(), top.y, true);
            SetEndPoint(m_left.get(), left.x, true);
            SetEndPoint(m_bottom.get(), bottom.y, false);
            SetEndPoint(m_right.get(), right.x, false);
        }
    }
}