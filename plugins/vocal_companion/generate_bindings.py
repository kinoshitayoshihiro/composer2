import pathlib
import textwrap

CPP = textwrap.dedent(
    """
    #include <pybind11/embed.h>
    #include <pybind11/stl.h>
    namespace py = pybind11;

    std::map<std::string, py::object> get_default_parameters() {
        py::gil_scoped_acquire guard{};
        py::object stub = py::module_::import("plugins.vocal_companion_stub");
        py::object func = stub.attr("get_default_parameters");
        auto res = func();
        return res.cast<std::map<std::string, py::object>>();
    }

    PYBIND11_MODULE(vocal_companion_plugin, m) {
        m.def("get_default_parameters", &get_default_parameters);
    }
    """
)


def main() -> None:
    out_dir = pathlib.Path(__file__).resolve().parent / "bindings"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plugin.cpp").write_text(CPP)


if __name__ == "__main__":
    main()
