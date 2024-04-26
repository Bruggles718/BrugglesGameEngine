#pragma once

#include "TDynamicArray.cuh"
#include "physics/Collider.cuh"


namespace bruggles {
    namespace physics {
        struct HullCollider : Collider {
            __host__ __device__ HullCollider();

            __host__ __device__ HullCollider(TDynamicArray<Vector2>& i_vertices);

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

            TDynamicArray<Vector2> Vertices = TDynamicArray<Vector2>();
        };
    }
}