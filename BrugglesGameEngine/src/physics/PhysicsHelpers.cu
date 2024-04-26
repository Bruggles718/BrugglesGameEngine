#include "physics/PhysicsHelpers.cuh"
#include <cfloat>
#include <math.h>
#include <SDL.h>
#include "TDynamicArray.cuh"
#include <iostream>

namespace bruggles {
    namespace physics {
        CollisionPoints CalcCircleCircleCollisionPoints(
            const CircleCollider* i_a, const Transform* i_ta,
            const CircleCollider* i_b, const Transform* i_tb
        ) {
            std::pair<bool, Simplex> simplexData = GJK(
                i_a, i_ta,
                i_b, i_tb
            );

            #ifdef  __CUDA_ARCH__
            #else
            std::cout << "has collision: " << simplexData.first << std::endl;
            std::cout << "a radius: " << i_a->Radius << std::endl;
            std::cout << "b radius: " << i_a->Radius << std::endl;
            std::cout << "a tf pos: (" << i_ta->Position.x << ", " << i_ta->Position.y << ")" << std::endl;
            std::cout << "b tf pos: (" << i_tb->Position.x << ", " << i_tb->Position.y << ")" << std::endl;
            #endif
            

            if (!simplexData.first) return CollisionPoints();

            CollisionPoints p = EPA(simplexData.second, i_a, i_ta, i_b, i_tb);

            #ifdef  __CUDA_ARCH__
            #else
            std::cout << "has collision: " << p.HasCollision << std::endl;
            std::cout << "normal: (" << p.Normal.x << ", " << p.Normal.x << ")" << std::endl;
            std::cout << "depth: " << p.Depth << std::endl;
            #endif

            return p;
        }
        
        CollisionPoints CalcCircleHullCollisionPoints(
            const CircleCollider* i_circleCollider, const Transform* i_tCircleCollider,
            const HullCollider* i_hullCollider, const Transform* i_tHullCollider
        ) {
            std::pair<bool, Simplex> simplexData = GJK(
                i_circleCollider, i_tCircleCollider,
                i_hullCollider, i_tHullCollider
            );

            if (!simplexData.first) return CollisionPoints();

            return EPA(simplexData.second, i_circleCollider, i_tCircleCollider, i_hullCollider, i_tHullCollider);
        }

        CollisionPoints CalcHullCircleCollisionPoints(
            const HullCollider* i_hullCollider, const Transform* i_tHullCollider,
            const CircleCollider* i_circleCollider, const Transform* i_tCircleCollider
        ) {
            CollisionPoints result = CalcCircleHullCollisionPoints(
                i_circleCollider, i_tCircleCollider,
                i_hullCollider, i_tHullCollider
            );
            return result.Flip();
        }

        CollisionPoints CalcHullHullCollisionPoints(
            const HullCollider* i_a, const Transform* i_ta,
            const HullCollider* i_b, const Transform* i_tb
        ) {
            std::pair<bool, Simplex> simplexData = GJK(
                i_a, i_ta,
                i_b, i_tb
            );

            if (!simplexData.first) return CollisionPoints();

            return EPA(simplexData.second, i_a, i_ta, i_b, i_tb);
        }

        Vector2 MinkowskiSupport(
            const Collider* i_a, const Transform* i_ta,
            const Collider* i_b, const Transform* i_tb,
            Vector2 direction
        ) {
            return i_a->FindFurthestPoint(i_ta, direction) - i_b->FindFurthestPoint(i_tb, -direction);
        }

        std::pair<bool, Simplex> GJK(
            const Collider* i_a, const Transform* i_ta,
            const Collider* i_b, const Transform* i_tb
        ) {
            Vector2 support = MinkowskiSupport(i_a, i_ta, i_b, i_tb, Vector2::UnitX());
            
            Simplex vertices{};
            vertices.Push_Front(support);

            Vector2 direction = -support;

            for (int i = 0; i < 32; i++) {
                support = MinkowskiSupport(i_a, i_ta, i_b, i_tb, direction);

                // we've found the closest feature already, and there is no collision
                if (Vector2::Dot(support, direction) <= 0) {
                    break;
                }

                vertices.Push_Front(support);

                if (NextSimplex(vertices, direction)) {
                    return std::pair<bool, Simplex>(true, vertices);
                }
            }

            return std::pair<bool, Simplex>(false, vertices);
        }

        bool NextSimplex(
            Simplex& vertices,
            Vector2& direction
        ) {
            switch (vertices.Size()) {
                case 2: return Line(vertices, direction);
                case 3: return Triangle(vertices, direction);
            }
            return false;
        }

        bool SameDirection(
            const Vector2& direction,
            const Vector2& ao
        ) {
            return Vector2::Dot(direction, ao) > 0;
        }

        bool Line(
            Simplex& vertices,
            Vector2& direction
        ) {
            Vector2 a = vertices[0];
            Vector2 b = vertices[1];

            Vector2 ab = b - a;
            Vector2 ao = -a;

            direction = TripleProduct(ab, ao, ab);

            return false;
        }

        bool Triangle(
            Simplex& vertices,
            Vector2& direction
        ) {
            Vector2 a = vertices[0];
            Vector2 b = vertices[1];
            Vector2 c = vertices[2];

            Vector2 ab = b-a;
            Vector2 ac = c-a;
            Vector2 ao = -a;

            Vector2 abf = TripleProduct(ac, ab, ab);
            Vector2 acf = TripleProduct(ab, ac, ac);

            if (SameDirection(abf, ao)) {
                Vector2 arr[] = { a, b };
                vertices = TArray<Vector2>(arr, 2);
                return Line(vertices, direction);
            }

            if (SameDirection(acf, ao)) {
                Vector2 arr[] = { a, c };
                vertices = TArray<Vector2>(arr, 2);
                return Line(vertices, direction);
            }

            return true;
        }
        
        Vector2 TripleProduct(
            const Vector2& i_a, 
            const Vector2& i_b, 
            const Vector2& i_c
        ) {
            float AcrossBZ = i_a.x*i_b.y - i_a.y*i_b.x;

            Vector2 result = Vector2(
                -(AcrossBZ * i_c.y), 
                AcrossBZ * i_c.x
            );

            return result;
        }

        __host__ __device__ CollisionPoints EPA(
            const Simplex& i_simplex,
            const Collider* i_colliderA, const Transform* i_tfA,
            const Collider* i_colliderB, const Transform* i_tfB
        ) {
            TDynamicArray<Vector2> polytope{};
            for (int i = 0; i < i_simplex.Size(); i++) {
                #ifdef  __CUDA_ARCH__
                #else
                std::cout << i_simplex[i].x << ", " << i_simplex[i].y << std::endl;
                #endif
                polytope.PushBack(i_simplex[i]);
                #ifdef  __CUDA_ARCH__
                #else
                std::cout << polytope[i].x << ", " << polytope[i].y << std::endl;
                #endif
            }

            int minIdx = 0;
            float minDistance = FLT_MAX;
            Vector2 minNormal;

            #ifdef  __CUDA_ARCH__
            #else
            std::cout << polytope.Size() << std::endl;
            for (int i = 0; i < polytope.Size(); i++) {
                std::cout << polytope[i].x << ", " << polytope[i].y << std::endl;
            }
            #endif

            for (int i = 0; i < 32 && minDistance == FLT_MAX; i++) {
                for (int i = 0; i < polytope.Size(); i++) {
                    Vector2 a = polytope[i];
                    Vector2 b = polytope[(i + 1) % polytope.Size()];

                    Vector2 ab = b - a;

                    Vector2 normal = (Vector2(ab.y, -ab.x)).Normalized();
                    float distance = Vector2::Dot(normal, a);

                    if (distance < 0) {
                        distance *= - 1;
                        normal = normal * -1;
                    }

                    if (distance < minDistance) {
                        minDistance = distance;
                        minNormal = normal;
                        minIdx = i;
                    }
                }

                Vector2 support = MinkowskiSupport(
                    i_colliderA, i_tfA,
                    i_colliderB, i_tfB,
                    minNormal
                );

                float sDistance = Vector2::Dot(minNormal, support);

                if (std::abs(sDistance - minDistance) > 0.001f) {
                    minDistance = FLT_MAX;
                    polytope.Insert(minIdx + 1, support);
                }
            }

            if (minDistance == FLT_MAX) {
                return CollisionPoints();
            }

            CollisionPoints result;
            result.Normal = minNormal;
            result.Depth = minDistance;
            result.HasCollision = true;
            return result;
        }
    }
}