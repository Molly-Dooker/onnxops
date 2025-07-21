#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "state_manager.h"

namespace py = pybind11;
using namespace MyQuantLib;

PYBIND11_MODULE(my_quant_lib, m) {
    m.doc() = "Python bindings for my custom quantization library";

    py::class_<ObserverState>(m, "ObserverState")
     .def(py::init<>())
     .def_readonly("min", &ObserverState::min)
     .def_readonly("max", &ObserverState::max)
     .def("__repr__",(const ObserverState &s) {
            return "<ObserverState min=" + std::to_string(s.min) + ", max=" + std::to_string(s.max) + ">";
        });

    m.def("register_observer",(const std::string& id) {
        StateManager::get_instance().register_observer(id);
    }, py::arg("id"), "Registers a new observer with the given ID.");

    m.def("get_observer_state",(const std::string& id) {
        return StateManager::get_instance().get_state(id);
    }, py::arg("id"), "Gets the state of a registered observer.");
}