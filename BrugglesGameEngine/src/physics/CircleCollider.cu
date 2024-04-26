#include "physics/CircleCollider.cuh"
#include "physics/PhysicsHelpers.cuh"
#include "math.h"

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

        __host__ Collider* CircleCollider::GetDeviceCopy() {
            CircleCollider* result = 0;
            cudaMalloc(&result, sizeof(CircleCollider));
            cudaMemcpy(result, this, sizeof(CircleCollider), cudaMemcpyHostToDevice);
            return result;
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
            float drnx = dx*cosf(-w) - dy*sinf(-w);
            float drny = dx*sinf(-w) + dy*cosf(-w);

            float ax = cosf(atan2f(drny/sy, drnx/sx));
            float ay = sinf(atan2f(drny/sy, drnx/sx));

            float perpAngle = M_PI/2;
            float aperpx = ax*cosf(perpAngle) - ay*sinf(perpAngle);
            float aperpy = ax*sinf(perpAngle) + ay*cosf(perpAngle);

            float reTransformax = aperpx * sx;
            float reTransformay = aperpy * sy;

            result.x = reTransformax*cosf(w) - reTransformay*sinf(w) + px;
            result.y = reTransformax*sinf(w) + reTransformay*cosf(w) + py;

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
                float dx = cosf(i * M_PI/180);
                float dy = sinf(i * M_PI/180);
                float drnx = dx*cosf(-w) - dy*sinf(-w);
                float drny = dx*sinf(-w) + dy*cosf(-w);
                float nA2 = atan2f(drny/sy, drnx/sx);
                result.x = sx*cosf(nA2)*cosf(w) - sy*sinf(nA2)*sinf(w) + px;
                result.y = sx*cosf(nA2)*sinf(w) + sy*sinf(nA2)*cosf(w) + py;
                result.ApplyTransform(&invTf);
                SDL_RenderDrawPoint(i_camera->m_renderer, (int)result.x, (int)result.y);
            }
        }
    }
}