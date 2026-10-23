import pickle, io
_ALLOWED = {('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'), ('numpy', 'dtype'), ('_codecs', 'encode')}
class _RU(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:               # numpy-scalar reconstruction only; anything else is refused
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked global {module}.{name}')
def load(path):
    with open(path,'rb') as f: return _RU(io.BytesIO(f.read())).load()
