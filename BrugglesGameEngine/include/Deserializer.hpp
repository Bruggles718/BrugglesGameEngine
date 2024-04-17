#pragma once

#include <string>
#include <unordered_map>
#include <functional>
#include <pybind11/pybind11.h>

namespace bruggles {
    /**
     * This class is for deserializing objects that have been serialized from extending the Serializable class
    */
    class Deserializer {
    public:
        /**
         * This method finds the correct deserialization function and returns back a string that can be interpreted by python into a python object
         * @param i_type the type of the serialized value being passed in
         * @param i_value the value of the serialized object
        */
        static std::string Deserialize(std::string i_type, pybind11::dict i_value);

        static bool CanDeserialize(std::string i_type);

        static void Init();
    private:
        static std::string DeserializeString(pybind11::dict i_dict);
        static std::string DeserializePrimitive(pybind11::dict i_dict);
        static std::string DeserializeVector2(pybind11::dict i_dict);
        static std::string DeserializeList(pybind11::dict i_dict);
        static std::string DeserializeSerializeField(pybind11::dict i_dict);
        static std::unordered_map<std::string, std::function<std::string(pybind11::dict)>> s_deserializers;
    };
}
