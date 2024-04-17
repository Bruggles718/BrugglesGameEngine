#include "Deserializer.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <vector>

namespace py = pybind11;

namespace bruggles {
    std::unordered_map<std::string, std::function<std::string(pybind11::dict)>> Deserializer::s_deserializers{};
    std::string Deserializer::DeserializePrimitive(py::dict i_dict) {
        std::string result = i_dict["Value"].cast<std::string>();
        return result;
        
    }

    std::string Deserializer::DeserializeString(py::dict i_dict) {
        std::string result = "\"" + i_dict["Value"].cast<std::string>() + "\"";
        return result;
    }

    std::string Deserializer::DeserializeVector2(py::dict i_dict) {
        std::string x = DeserializePrimitive(i_dict["Value"]["X"]);
        std::string y = DeserializePrimitive(i_dict["Value"]["Y"]);

        return "bruggles.Vector2(" + x + ", " + y + ")";
    }

    std::string Deserializer::DeserializeList(py::dict i_dict) {
        py::list value = i_dict["Value"];
        std::vector<std::string> result;
        for (py::handle element : value) {
            result.push_back(
                Deserialize(
                    element["Type"].cast<std::string>(),
                    py::cast<py::dict>(element)
                )
            );
        }

        std::string resultStr = "";
        for (int i = 0; i < result.size(); i++) {
            resultStr += result[i];
            if (i + 1 < result.size()) {
                resultStr += ", ";
            }
        }

        return "[" + resultStr + "]";
    }

    std::string Deserializer::DeserializeSerializeField(py::dict i_dict) {
        std::string type = i_dict["Value"]["Value"]["Type"].cast<std::string>();
        std::string field = Deserialize(type, i_dict["Value"]["Value"]);

        return "SerializeField.SerializeField(\"" + type + "\", " + field + ")";
    }

    std::string Deserializer::Deserialize(std::string i_type, py::dict i_value) {
        std::function<std::string(py::dict)> func = Deserializer::s_deserializers[i_type];
        return func(i_value);
    }

    bool Deserializer::CanDeserialize(std::string i_type) {
        return Deserializer::s_deserializers.find(i_type) != Deserializer::s_deserializers.end();
    }

    void Deserializer::Init() {
        Deserializer::s_deserializers["Vector2"] = DeserializeVector2;
        Deserializer::s_deserializers["list"] = DeserializeList;
        Deserializer::s_deserializers["float"] = DeserializePrimitive;
        Deserializer::s_deserializers["int"] = DeserializePrimitive;
        Deserializer::s_deserializers["str"] = DeserializeString;
        Deserializer::s_deserializers["bool"] = DeserializePrimitive;
        Deserializer::s_deserializers["SerializeField"] = DeserializeSerializeField;
    }
}