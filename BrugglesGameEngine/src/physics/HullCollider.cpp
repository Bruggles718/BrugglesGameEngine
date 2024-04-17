#include "physics/HullCollider.hpp"
#include "physics/PhysicsHelpers.hpp"
#include <cfloat>

namespace bruggles {
    namespace physics {
        HullCollider::HullCollider() {}

        HullCollider::HullCollider(std::vector<Vector2> i_vertices) {
            this->Vertices = i_vertices;
        }

        CollisionPoints HullCollider::CheckCollision(
            const Transform* i_transform,
            const Collider* i_other,
            const Transform* i_otherTransform
        ) const {
            return i_other->CheckCollisionWithHullCollider(i_otherTransform, this, i_transform);
        }

        CollisionPoints HullCollider::CheckCollisionWithCircleCollider(
            const Transform* i_transform,
            const CircleCollider* i_circleCollider,
            const Transform* i_circleColliderTransform
        ) const {
            return CalcHullCircleCollisionPoints(
                this, i_transform,
                i_circleCollider, i_circleColliderTransform
            );
        }

        CollisionPoints HullCollider::CheckCollisionWithHullCollider(
            const Transform* i_transform,
            const HullCollider* i_hullCollider,
            const Transform* i_hullColliderTransform
        ) const {
            return CalcHullHullCollisionPoints(
                this, i_transform,
                i_hullCollider, i_hullColliderTransform
            );
        }

        Vector2 HullCollider::FindFurthestPoint(const Transform* i_tf, const Vector2& direction) const {
            Vector2 maxPoint;
            float maxDistance = -FLT_MAX;

            for (Vector2 vertex : this->Vertices) {
                Transform scaleRot{
                    Vector2(0, 0),
                    i_tf->Rotation,
                    i_tf->Scale
                };
                float distance = Vector2::Dot(vertex.Transformed(&scaleRot), direction);
                if (distance > maxDistance) {
                    maxDistance = distance;
                    maxPoint = vertex.Transformed(&scaleRot);
                }
            }

            return maxPoint + i_tf->Position;
        }

        void HullCollider::Render(const Transform* i_tf, const Camera* i_camera) {
            SDL_SetRenderDrawColor(i_camera->m_renderer, 0xFF, 0xFF, 0xFF, 0xFF);
            for (int i = 0; i < this->Vertices.size(); i++) {
                Vector2 p1 = this->Vertices[i];
                Vector2 p2 = this->Vertices[(i + 1) % this->Vertices.size()];
                p1.ApplyTransform(i_tf);
                p2.ApplyTransform(i_tf);
                Transform invTf = i_camera->GetInverseTransform();
                p1.ApplyTransform(&invTf);
                p2.ApplyTransform(&invTf);

                SDL_RenderDrawLine(
                    i_camera->m_renderer,
                    (int)p1.x, (int)p1.y,
                    (int)p2.x, (int)p2.y
                );
            }
        }
    }
}