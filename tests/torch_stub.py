import importlib
import math
import sys
from types import ModuleType


def _stub_torch() -> None:
    if importlib.util.find_spec("torch") is not None:
        return

    class Tensor(list):
        def unsqueeze(self, dim: int = 0) -> "Tensor":  # pragma: no cover - simple
            if dim != 0:
                raise NotImplementedError
            return Tensor([list(self)])

        def tolist(self) -> list:
            return list(self)

    def tensor(data, dtype=None):  # type: ignore[override]
        return Tensor(data if isinstance(data, list) else [data])

    def arange(n: int, dtype=None) -> Tensor:
        return Tensor(list(range(n)))

    def ones(*shape: int, dtype=None) -> Tensor:
        def build(dims):
            if len(dims) == 1:
                return [1] * dims[0]
            return [build(dims[1:]) for _ in range(dims[0])]

        return Tensor(build(list(shape)))

    def sigmoid(t: Tensor) -> Tensor:
        def apply(x):
            if isinstance(x, list):
                return [apply(i) for i in x]
            return 1 / (1 + math.exp(-x))

        return Tensor(apply(t))

    class NoGrad:
        def __enter__(self) -> None:  # pragma: no cover - stub
            pass

        def __exit__(self, *exc: object) -> None:  # pragma: no cover - stub
            pass

    torch = ModuleType("torch")
    torch.Tensor = Tensor
    torch.tensor = tensor
    torch.arange = arange
    torch.ones = ones
    torch.sigmoid = sigmoid
    torch.no_grad = NoGrad
    torch.float32 = float
    torch.long = int
    torch.bool = bool
    torch.manual_seed = lambda _seed: None  # pragma: no cover - stub
    torch.get_float32_matmul_precision = lambda: "medium"

    class _Device:
        def __init__(self, kind: str) -> None:
            self.type = kind

    torch.device = lambda kind: _Device(kind)

    cuda = ModuleType("torch.cuda")
    cuda.is_available = lambda: False
    cuda.amp = ModuleType("torch.cuda.amp")
    cuda.amp.GradScaler = lambda *a, **k: None
    torch.cuda = cuda

    backends = ModuleType("torch.backends")
    mps = ModuleType("torch.backends.mps")
    mps.is_available = lambda: False
    backends.mps = mps
    torch.backends = backends

    nn = ModuleType("torch.nn")
    nn.Module = object
    torch.nn = nn

    utils = ModuleType("torch.utils")
    data = ModuleType("torch.utils.data")
    data.DataLoader = object
    data.Dataset = object
    utils.data = data
    torch.utils = utils

    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.utils"] = utils
    sys.modules["torch.utils.data"] = data

