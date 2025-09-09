#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<py::dict> generateSax(py::dict options) {
    py::gil_scoped_acquire guard{};
    try {
        py::object mod = py::module_::import("plugins.sax_companion_plugin");
        if (!py::hasattr(mod, "generateSax")) {
            throw std::runtime_error("missing generateSax");
        }
        py::object res = mod.attr("generateSax")(options);
        return res.cast<std::vector<py::dict>>();
    } catch (const std::exception& e) {
        try {
            py::module_ logging = py::module_::import("logging");
            logging.attr("error")(py::str("sax_companion plugin load failed: {}").format(e.what()));
        } catch (const std::exception&) {
            // Ignore logging errors
        }
        py::dict err;
        err["error"] = "PluginLoadError";
        err["message"] = e.what();
        return {err};
    }
}

PYBIND11_MODULE(sax_companion_plugin, m) {
    m.def("generateSax", &generateSax);
}
