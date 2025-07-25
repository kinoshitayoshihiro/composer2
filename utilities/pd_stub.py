class _Series(list):
    def clip(self, lower, upper):
        return _Series([max(lower, min(upper, x)) for x in self])

    def fillna(self, val):
        return _Series([val if v in {None, ""} else v for v in self])

    def astype(self, typ):
        return _Series([typ(v) for v in self])

    def to_numpy(self):
        import numpy as np

        return np.array(self)


class _Loc:
    def __init__(self, df: "_DF") -> None:
        self._df = df

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return {k: self._df._data[k][idx] for k in self._df._data}
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self._df)))]
        return [self[i] for i in idx]


class _DF:
    def __init__(self, data: dict[str, list] | list):
        if isinstance(data, list):
            data = {"roll": data}
        self._data = {
            k: (v.tolist() if hasattr(v, "tolist") else list(v))
            for k, v in data.items()
        }

    def __len__(self) -> int:
        return len(next(iter(self._data.values()), []))

    def __iter__(self):
        first = next(iter(self._data.values()), [])
        return iter(first)

    def __getitem__(self, key: str) -> _Series:
        return _Series(self._data[key])

    def __setitem__(self, key: str, val: list) -> None:
        self._data[key] = list(val)

    @staticmethod
    def read_csv(path: str | "Path") -> "_DF":
        import csv
        from pathlib import Path

        p = Path(path)
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            data: dict[str, list] = {k: [] for k in reader.fieldnames or []}
            for row in reader:
                for k in data:
                    val = row.get(k, "")
                    try:
                        num = float(val)
                        val = int(num) if num.is_integer() else num
                    except Exception:
                        pass
                    data[k].append(val)
        return _DF(data)

    @property
    def columns(self):
        return list(self._data.keys())

    @property
    def values(self):
        import numpy as np

        return np.array([self._data[k] for k in self._data]).T

    @property
    def loc(self) -> _Loc:  # type: ignore[override]
        return _Loc(self)

    def to_csv(self, path: str | None = None, index: bool = False) -> None:
        import csv

        keys = list(self._data.keys())
        rows = zip(*[self._data[k] for k in keys])
        if path is None:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerows(rows)

    def iterrows(self):
        for i in range(len(self)):
            yield i, {k: self._data[k][i] for k in self._data}


def install_pandas_stub() -> None:
    import sys
    import types

    pd_mod = types.ModuleType("pandas")
    pd_mod.DataFrame = lambda data: _DF(data)
    pd_mod.read_csv = lambda path: _DF.read_csv(path)  # type: ignore
    pd_mod.Series = _Series
    sys.modules["pandas"] = pd_mod
