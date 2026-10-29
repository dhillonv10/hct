"""Inspect and load pandas pickles from untrusted sources without running arbitrary code.

    python3 -I safepickle.py audit FILE.bz2          # list every (module, name) the pickle requests; executes nothing
    from safepickle import load_strict               # load with an exact allowlist (raises on anything else)
"""
import bz2
import pickle
import sys

# exact names a plain DataFrame / Series / ndarray pickle needs; anything else is refused
ALLOW = {
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.series", "Series"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas.core.internals.blocks", "new_block"),
    ("pandas.core.internals.blocks", "maybe_coerce_values"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas._libs.internals", "BlockPlacement"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.indexes.multi", "MultiIndex"),
    ("pandas.core.indexes.category", "CategoricalIndex"),
    ("pandas.core.arrays.categorical", "Categorical"),
    ("pandas.core.dtypes.dtypes", "CategoricalDtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("builtins", "slice"),
    ("builtins", "object"),
}


class _Dummy:
    """Absorbs any call / attribute / state update so the unpickler can run to the end without touching real code."""
    def __new__(cls, *a, **k):
        return object.__new__(cls)

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Dummy()

    def __setstate__(self, s):
        pass

    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        return _Dummy()

    def __setitem__(self, k, v):
        pass

    def __getitem__(self, k):
        return _Dummy()


class _Record(pickle.Unpickler):
    seen = None

    def find_class(self, module, name):
        self.seen.add((module, name))
        return _Dummy


def _frombuffer(buf, dtype, shape, order):
    import numpy as np
    return np.frombuffer(buf, dtype=dtype).reshape(shape, order=order).copy()


def _remap(module, name):
    """Legacy (pandas 1.x / numpy 1.x) names -> their modern, equally inert equivalents."""
    import numpy as np
    import pandas as pd
    if (module, name) == ("pandas.core.indexes.numeric", "Int64Index"):
        return pd.Index
    if (module, name) == ("numpy.core.numeric", "_frombuffer"):
        return _frombuffer
    if (module, name) == ("numpy.core.multiarray", "_reconstruct"):
        return np._core.multiarray._reconstruct
    return None


class _Strict(pickle.Unpickler):
    def find_class(self, module, name):
        r = _remap(module, name)
        if r is not None:
            return r
        if (module, name) not in ALLOW:
            raise pickle.UnpicklingError(f"refused {module}.{name}")
        return super().find_class(module, name)


def audit(path):
    u = _Record(bz2.open(path, "rb"))
    u.seen = set()
    try:
        u.load()
    except Exception as e:  # placeholders make downstream reconstruction fail; the globals seen so far are what matters
        print("(record pass stopped:", type(e).__name__, ")")
    for m, n in sorted(u.seen):
        print(("OK      " if (m, n) in ALLOW else "NOT-ALLOWED  ") + f"{m}.{n}")
    return u.seen


def load_strict(path):
    with bz2.open(path, "rb") as f:
        return _Strict(f).load()


if __name__ == "__main__":
    if sys.argv[1] == "audit":
        audit(sys.argv[2])
