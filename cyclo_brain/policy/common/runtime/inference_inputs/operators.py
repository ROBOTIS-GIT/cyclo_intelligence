"""CPU reference operators. Framework integrations may register tensor equivalents."""

import numpy as np

from .graph import Registry, fields


def _one(values):
    if len(values) != 1:
        raise ValueError("operator requires exactly one input")
    return values[0]


def default_registry():
    registry = Registry()

    def identity(options, context):
        fields(options, set(), "identity options")
        return _one

    registry.register("identity", identity)

    def select(options, context):
        fields(options, {"key"}, "select options")
        key = options.get("key")
        if not isinstance(key, (str, int)) or isinstance(key, bool):
            raise ValueError("select requires a string key or integer index")
        return lambda values: _one(values).derived(_one(values).value[key])

    registry.register("select", select)

    def array_operation(kind):
        def compile_op(options, context):
            allowed = {"stack": {"axis", "sequence"}, "concat": {"axis", "sequence"}, "slice": {"axis", "start", "stop", "step"},
                       "transpose": {"axes"}, "cast": {"dtype"}, "unsqueeze": {"axis"}}[kind]
            fields(options, allowed, f"{kind} options")
            axis = options.get("axis", 0)
            if type(axis) is not int:
                raise ValueError("axis must be an integer")
            if type(options.get("sequence", False)) is not bool:
                raise ValueError("sequence must be boolean")
            if kind == "cast":
                if not isinstance(options.get("dtype"), str):
                    raise ValueError("cast requires an explicit dtype")
                dtype = np.dtype(options.get("dtype"))
                if dtype.kind not in "biuf":
                    raise ValueError("only numeric dtypes are supported")
            if kind == "transpose":
                axes = options.get("axes")
                if not isinstance(axes, list) or any(type(v) is not int for v in axes) or sorted(axes) != list(range(len(axes))):
                    raise ValueError("transpose axes must be a permutation")
            if kind == "slice":
                if any(v is not None and type(v) is not int for k, v in options.items() if k != "axis") or options.get("step") == 0:
                    raise ValueError("slice bounds must be integers and step nonzero")

            def run(values):
                arrays = [v.value for v in values]
                if kind in {"stack", "concat"}:
                    if options.get("sequence", False):
                        if len(arrays) != 1 or not isinstance(arrays[0], (tuple, list)):
                            raise ValueError("sequence mode requires one explicit collection")
                        arrays = arrays[0]
                    return (np.stack if kind == "stack" else np.concatenate)(arrays, axis=axis)
                value = _one(values)
                array = np.asarray(value.value)
                if kind == "cast":
                    result = array.astype(dtype, copy=True)
                elif kind == "transpose":
                    result = np.transpose(array, axes).copy()
                elif kind == "unsqueeze":
                    result = np.expand_dims(array, axis).copy()
                else:
                    selection = [slice(None)] * array.ndim
                    selection[axis] = slice(options.get("start"), options.get("stop"), options.get("step"))
                    result = array[tuple(selection)].copy()
                # Axis-changing operations must explicitly redeclare semantic axes.
                return value.derived(result, semantics={})
            return run
        return compile_op

    for name in ("stack", "concat", "slice", "transpose", "cast", "unsqueeze"):
        registry.register(name, array_operation(name))
    return registry
