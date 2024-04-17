#pragma once

#include "physics/Collider.hpp"
#include "Vector2.hpp"

namespace bruggles {
    namespace physics {
        struct CircleCollider : Collider {
            Vector2 Center;
            float Radius;

            CircleCollider(Vector2 i_center, float i_radius);

            CollisionPoints CheckCollision(
                const Transform* i_transform,
                const Collider* i_other,
                const Transform* i_otherTransform
            ) const override;

            CollisionPoints CheckCollisionWithCircleCollider(
                const Transform* i_transform,
                const CircleCollider* i_circleCollider,
                const Transform* i_circleColliderTransform
            ) const override;

            CollisionPoints CheckCollisionWithHullCollider(
                const Transform* i_transform,
                const HullCollider* i_hullCollider,
                const Transform* i_hullColliderTransform
            ) const override;

            Vector2 FindFurthestPoint(const Transform* i_tf, const Vector2& direction) const override;

            void Render(const Transform* i_tf, const Camera* i_camera) override;
        };
    }
}