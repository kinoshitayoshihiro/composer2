#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<py::dict> generateBar(py::dict preset) {
    py::gil_scoped_acquire guard{};
    py::object stub = py::module_::import("plugins.modcompose_stub");
    py::object func = stub.attr("generateBar");
    py::object res = func(preset);
    return res.cast<std::vector<py::dict>>();
}

PYBIND11_MODULE(modcompose_plugin, m) {
    m.def("generateBar", &generateBar);
}
