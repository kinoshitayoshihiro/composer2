import numpy as np


def resample_poly(
    x: np.ndarray, up: int, down: int, axis: int = 0, window: None | np.ndarray = None
) -> np.ndarray:
    arr = np.asarray(x)
    n_in = arr.shape[axis]
    n_out = int(round(n_in * up / down))
    coords = np.linspace(0, n_in - 1, n_out)
    return np.interp(coords, np.arange(n_in), np.take(arr, np.arange(n_in), axis=axis))


class windows:
    @staticmethod
    def hamming(M: int, sym: bool = True) -> np.ndarray:
        return np.hamming(M)
