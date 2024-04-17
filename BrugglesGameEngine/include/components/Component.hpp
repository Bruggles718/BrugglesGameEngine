#pragma once

#include "GameObject.hpp"
#include "pybind11/pybind11.h"

namespace bruggles {
    namespace components {
        /**
         * Represents a property of a Game Object
        */
        class Component {
        public:
            /**
             * Called when this component is added to a Game Object
            */
            void SetGameObject(GameObject* i_object);

            /**
             * Calls GetComponent on this Component's Game Object
            */
            pybind11::object GetComponent(pybind11::object i_type);

            /**
             * Gets a pointer to the Game Object this Component is attached to
            */
            GameObject* GetGameObject();

            /**
             * Called when this Component is removed from a Game Object
            */
            virtual void OnComponentRemoved();
        
        protected:
            /**
             * Override this to add extra functionality to SetGameObject without removing 
             * the base functionality of setting this Component's Game Object
            */
            virtual void _SetGameObject(GameObject* i_object);
            
            GameObject* m_gameObject;
        };
    }
}