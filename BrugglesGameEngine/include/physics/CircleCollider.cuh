#pragma once

#include "physics/Collider.cuh"
#include "Vector2.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace bruggles {
    namespace physics {
        struct CircleCollider : Collider {
            Vector2 Center;
            float Radius;

            __host__ __device__ CircleCollider(Vector2 i_center, float i_radius);

            __host__ __device__ CollisionPoints CheckCollision(
                const Transform* i_transform,
                const Collider* i_other,
                const Transform* i_otherTransform
            ) const override;

            __host__ __device__ CollisionPoints CheckCollisionWithCircleCollider(
                const Transform* i_transform,
                const CircleCollider* i_circleCollider,
                const Transform* i_circleColliderTransform
            ) const override;

            __host__ __device__ CollisionPoints CheckCollisionWithHullCollider(
                const Transform* i_transform,
                const HullCollider* i_hullCollider,
                const Transform* i_hullColliderTransform
            ) const override;

            __host__ __device__ Vector2 FindFurthestPoint(const Transform* i_tf, const Vector2& direction) const override;

            __host__ Collider* GetDeviceCopy() override;

            void Render(const Transform* i_tf, const Camera* i_camera) override;
        };
    }
}