#include <pybind11/embed.h>
#include <pybind11/stl.h>
namespace py = pybind11;
std::map<std::string, py::object> get_default_parameters() {
    py::gil_scoped_acquire guard{};
    py::object stub = py::module_::import("plugins.piano_companion_stub");
    py::object func = stub.attr("get_default_parameters");
    auto res = func();
    return res.cast<std::map<std::string, py::object>>();
}
PYBIND11_MODULE(piano_companion_plugin, m) {
    m.def("get_default_parameters", &get_default_parameters);
}
