#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "state_manager.h"

namespace py = pybind11;
using namespace MyQuantLib;

PYBIND11_MODULE(my_quant_lib, m) {
    m.doc() = "Python bindings for my custom quantization library";

    // ObserverState 클래스 바인딩
    py::class_<ObserverState>(m, "ObserverState")
        .def(py::init<>())
        .def_readonly("min", &ObserverState::min)
        .def_readonly("max", &ObserverState::max)
        .def_readonly("bins", &ObserverState::bins)
        .def_readwrite("hist", &ObserverState::hist)
        .def("__repr__", [](const ObserverState &s) {
            return "<ObserverState min=" + std::to_string(s.min) +
                   ", max=" + std::to_string(s.max) +
                   ", bins=" + std::to_string(s.bins) + ">";
        });

    // MovingAverage 옵저버 등록
    m.def("register_moving_average_observer",
          [](const std::string& id) {
            StateManager::get_instance().register_moving_average(id);
          },
          py::arg("id"),
          "Registers a new moving‑average observer with the given ID.");

    // Histogram 옵저버 등록
    m.def("register_histogram_observer",
          [](const std::string& id, int64_t bins) {
            StateManager::get_instance().register_histogram(id, bins);
          },
          py::arg("id"), py::arg("bins"),
          "Registers a new histogram observer with the given ID and number of bins.");

    // MovingAverage 상태 조회
    m.def("get_observer_state",
          [](const std::string& id) {
            return StateManager::get_instance().get_state(id);
          },
          py::arg("id"),
          "Returns the ObserverState (min/max) for the given moving‑average observer ID.");

    // Histogram 데이터 조회
    m.def("get_histogram",
          [](const std::string& id) {
            return StateManager::get_instance().get_state(id).hist;
          },
          py::arg("id"),
          "Returns the latest histogram (as a list of counts) for the given histogram observer ID.");
}
