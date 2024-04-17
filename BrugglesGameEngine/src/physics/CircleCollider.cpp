#include "physics/CircleCollider.hpp"
#include "physics/PhysicsHelpers.hpp"

namespace bruggles {
    namespace physics {
        CircleCollider::CircleCollider(Vector2 i_center, float i_radius) {
            this->Center = i_center;
            this->Radius = i_radius;
        }

        CollisionPoints CircleCollider::CheckCollision(
            const Transform* i_transform,
            const Collider* i_other,
            const Transform* i_otherTransform
        ) const {
            CollisionPoints result = i_other->CheckCollisionWithCircleCollider(i_otherTransform, this, i_transform);
            return result;
        }

        CollisionPoints CircleCollider::CheckCollisionWithCircleCollider(
            const Transform* i_transform,
            const CircleCollider* i_circleCollider,
            const Transform* i_circleColliderTransform
        ) const {
            return CalcCircleCircleCollisionPoints(
                this, i_transform,
                i_circleCollider, i_circleColliderTransform
            );
        }

        CollisionPoints CircleCollider::CheckCollisionWithHullCollider(
            const Transform* i_transform,
            const HullCollider* i_hullCollider,
            const Transform* i_hullColliderTransform
        ) const {
            return CalcCircleHullCollisionPoints(
                this, i_transform,
                i_hullCollider, i_hullColliderTransform
            );
        }

        Vector2 CircleCollider::FindFurthestPoint(const Transform* i_tf, const Vector2& direction) const {

            Vector2 normalizedDirection = direction.Normalized();
            // x becomes y
            // y becomes -x
            Vector2 adjustedDir{normalizedDirection.y, -normalizedDirection.x};
            Vector2 result{0, 0};
            Vector2 center{
                Center.x,
                Center.y
            };
            center.ApplyTransform(i_tf);
            float px = center.x;
            float py = center.y;
            float sx = i_tf->Scale.x * Radius;
            float sy = i_tf->Scale.y * Radius;
            float w = i_tf->Rotation * M_PI/180;
            float dx = adjustedDir.x;
            float dy = adjustedDir.y;
            float drnx = dx*std::cos(-w) - dy*std::sin(-w);
            float drny = dx*std::sin(-w) + dy*std::cos(-w);

            float ax = std::cos(std::atan2(drny/sy, drnx/sx));
            float ay = std::sin(std::atan2(drny/sy, drnx/sx));

            float perpAngle = M_PI/2;
            float aperpx = ax*std::cos(perpAngle) - ay*std::sin(perpAngle);
            float aperpy = ax*std::sin(perpAngle) + ay*std::cos(perpAngle);

            float reTransformax = aperpx * sx;
            float reTransformay = aperpy * sy;

            result.x = reTransformax*std::cos(w) - reTransformay*std::sin(w) + px;
            result.y = reTransformax*std::sin(w) + reTransformay*std::cos(w) + py;

            return result;
        }

        void CircleCollider::Render(const Transform* i_tf, const Camera* i_camera) {
            SDL_SetRenderDrawColor(i_camera->m_renderer, 0xFF, 0xFF, 0xFF, 0xFF);
            float w = i_tf->Rotation * M_PI/180;

            Transform invTf = i_camera->GetInverseTransform();

            for (int i = 0; i < 360; i++) {
                Vector2 result{};
                Vector2 center{
                    Center.x,
                    Center.y
                };
                center.ApplyTransform(i_tf);
                float px = center.x;
                float py = center.y;
                float sx = i_tf->Scale.x * Radius;
                float sy = i_tf->Scale.y * Radius;
                float dx = std::cos(i * M_PI/180);
                float dy = std::sin(i * M_PI/180);
                float drnx = dx*std::cos(-w) - dy*std::sin(-w);
                float drny = dx*std::sin(-w) + dy*std::cos(-w);
                float nA2 = std::atan2(drny/sy, drnx/sx);
                result.x = sx*std::cos(nA2)*std::cos(w) - sy*std::sin(nA2)*std::sin(w) + px;
                result.y = sx*std::cos(nA2)*std::sin(w) + sy*std::sin(nA2)*std::cos(w) + py;
                result.ApplyTransform(&invTf);
                SDL_RenderDrawPoint(i_camera->m_renderer, (int)result.x, (int)result.y);
            }
        }
    }
}