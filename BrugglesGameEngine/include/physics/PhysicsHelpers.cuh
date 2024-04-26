#pragma once

#include "physics/CollisionPoints.cuh"
#include "physics/CircleCollider.cuh"
#include "physics/HullCollider.cuh"
#include "physics/Simplex.cuh"

namespace bruggles {
    namespace physics {
        /**
         * Computes a circle-on-circle collision
        */
        __host__ __device__ CollisionPoints CalcCircleCircleCollisionPoints(
            const CircleCollider* i_a, const Transform* i_ta,
            const CircleCollider* i_b, const Transform* i_tb
        );
        
        /**
         * Computes a circle-on-hull collision
        */
        __host__ __device__ CollisionPoints CalcCircleHullCollisionPoints(
            const CircleCollider* i_circleCollider, const Transform* i_tCircleCollider,
            const HullCollider* i_hullCollider, const Transform* i_tHullCollider
        );

        /**
         * Computes a hull-on-circle collision
        */
        __host__ __device__ CollisionPoints CalcHullCircleCollisionPoints(
            const HullCollider* i_hullCollider, const Transform* i_tHullCollider,
            const CircleCollider* i_circleCollider, const Transform* i_tCircleCollider
        );

        /**
         * Computes a hull-on-hull collision
        */
        __host__ __device__ CollisionPoints CalcHullHullCollisionPoints(
            const HullCollider* i_a, const Transform* i_ta,
            const HullCollider* i_b, const Transform* i_tb
        );

        /**
         * Finds the point on the outer minkowski difference of two colliders in a given direction
        */
        __host__ __device__ Vector2 MinkowskiSupport(
            const Collider* i_a, const Transform* i_ta,
            const Collider* i_b, const Transform* i_tb,
            Vector2 direction
        );

        /**
         * Updates the given Simplex and search direction
         * @return whether or not we have found a collision
        */
        __host__ __device__ bool NextSimplex(
            Simplex& vertices,
            Vector2& direction
        );

        /**
         * Returns the cross product of i_a cross i_b cross i_c
        */
        __host__ __device__ Vector2 TripleProduct(
            const Vector2& i_a, 
            const Vector2& i_b, 
            const Vector2& i_c
        );

        __host__ __device__ bool Line(Simplex& vertices, Vector2& direction);

        __host__ __device__ bool Triangle(Simplex& vertices, Vector2& direction);

        __host__ __device__ bool SameDirection(
            const Vector2& direction,
            const Vector2& ao
        );

        /**
         * An implementation of the Gilbert-Johnson-Keerthi distance algorithm.
         * 
         * Computes the Simplex needed to determine whether or not there is a collision between the two given colliders.
         * @return a pair whose first is a boolean of whether or not there is a collision, and whose second is the computed Simplex
        */
        __host__ __device__ std::pair<bool, Simplex> GJK(
            const Collider* i_a, const Transform* i_ta,
            const Collider* i_b, const Transform* i_tb
        );

        /**
         * An implementation of the Expanding Polytope Algorithm.
         * 
         * Computes the CollisionPoints for the given colliders using the given Simplex
         * @param i_simplex The Simplex returned from GJK
         * @return a CollisionPoints containing the collision normal, and the depth of the collision
        */
        __host__ __device__ CollisionPoints EPA(
            const Simplex& i_simplex,
            const Collider* i_colliderA, const Transform* i_tfA,
            const Collider* i_colliderB, const Transform* i_tfB
        );
    }
}