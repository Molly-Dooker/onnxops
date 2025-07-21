#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "moving_avg_min_max.h"

namespace py = pybind11;

PYBIND11_MODULE(custom_observer_ops, m) {
    m.doc() = "Python bindings for custom observer ops";

    py::class_<ObserverState>(m, "ObserverState")
        .def_readonly("min", &ObserverState::min)
        .def_readonly("max", &ObserverState::max)
        .def("__repr__", [](const ObserverState &s) {
            return "<ObserverState min=" + std::to_string(s.min) +
                   ", max=" + std::to_string(s.max) + ">";
        });

    m.def("get_stats", &ObserverManager::get_stats, "Get observer statistics by name",
          py::arg("name"));
}
