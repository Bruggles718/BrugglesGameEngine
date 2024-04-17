#pragma once

namespace bruggles {
    namespace components {
        class ColliderComponent;
    }
    namespace physics {
        /**
         * Represents the relevant information for a collision to be passed to a script when OnCollision is called
        */
        struct CollisionEvent {
            components::ColliderComponent* OtherCollider;
            CollisionPoints Points;
        };
    }
}