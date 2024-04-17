#pragma once

#include <vector>
#include "physics/Collider.hpp"

namespace bruggles {
    namespace physics {
        struct HullCollider : Collider {
            HullCollider();

            HullCollider(std::vector<Vector2> i_vertices);

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

            std::vector<Vector2> Vertices;
        };
    }
}