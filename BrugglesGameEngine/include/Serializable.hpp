#pragma once

#include <string>
#define SERIALIZE_TO_STRING(arg) "\"" #arg "\""
#define SERIALIZE_ROOT(Type, Value) "{\"Type\": " SERIALIZE_TO_STRING(Type) ", \"Value\": " Value "}"
#define SERIALIZE(Name, Type, Value) SERIALIZE_TO_STRING(Name) ": " SERIALIZE_ROOT(Type, Value)

namespace bruggles {
    /**
     * Describes an object that can be turned into a JSON format to be deserialized later
    */
    class Serializable {
    public:
        /**
         * returns a string in a JSON format representing this object
        */
        virtual std::string Serialize();
    };
}