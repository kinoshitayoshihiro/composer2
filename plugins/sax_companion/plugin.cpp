#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<py::dict> generateSax(py::dict options) {
    py::gil_scoped_acquire guard{};
    py::object stub = py::module_::import("plugins.sax_companion_stub");
    py::object func = stub.attr("generate_notes");
    py::object res = func(options);
    return res.cast<std::vector<py::dict>>();
}

PYBIND11_MODULE(sax_companion_plugin, m) {
    m.def("generateSax", &generateSax);
}
