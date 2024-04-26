#pragma once

#include "physics/CollisionPoints.cuh"
#include "Transform.cuh"
#include "Camera.hpp"
#include <SDL.h>

namespace bruggles {
    namespace components {
        class ColliderComponent;
    }
    namespace physics {

        struct CircleCollider;
        struct HullCollider;

        /**
         * Represents the physical volume of an object in 2D space.
        */
        struct Collider {
            __host__ __device__ virtual CollisionPoints CheckCollision(
                const Transform* i_transform,
                const Collider* i_other,
                const Transform* i_otherTransform
            ) const = 0;

            __host__ __device__ virtual CollisionPoints CheckCollisionWithCircleCollider(
                const Transform* i_transform,
                const CircleCollider* i_circleCollider,
                const Transform* i_circleColliderTransform
            ) const = 0;

            __host__ __device__ virtual CollisionPoints CheckCollisionWithHullCollider(
                const Transform* i_transform,
                const HullCollider* i_hullCollider,
                const Transform* i_hullColliderTransform
            ) const = 0;

            __host__ __device__ virtual Vector2 FindFurthestPoint(const Transform* i_tf, const Vector2& direction) const = 0;

            __host__ virtual Collider* GetDeviceCopy() = 0;

            virtual void Render(const Transform* i_tf, const Camera* i_camera) = 0;

            components::ColliderComponent* m_component;
        };
    }
}