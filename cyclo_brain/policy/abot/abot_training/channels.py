"""Named training-channel selection, separate from the robot's raw profile."""

from copy import deepcopy
from pathlib import Path


VECTORS = {"state_names": "observation.state", "action_names": "action"}


def _names(value, label):
    if (not isinstance(value, list) or not value
            or any(not isinstance(n, str) or not n or n.strip() != n or "\x00" in n for n in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{label} requires nonempty, unique channel names")
    return list(value)


def resolve_channels(info, selection=None):
    if selection is None:
        selection = {key: "all" for key in VECTORS}
    if not isinstance(selection, dict) or set(selection) != set(VECTORS):
        raise ValueError("channel_selection requires exactly state_names and action_names")
    source, selected = {"version": 1}, {"version": 1}
    indices = {}
    for key, feature in VECTORS.items():
        metadata = info["features"][feature]
        source[key] = _names(metadata.get("names"), f"source {key}")
        if metadata.get("shape") != [len(source[key])]:
            raise ValueError(f"Source shape differs from channel names: {feature}")
        requested = selection[key]
        selected[key] = list(source[key]) if requested == "all" else _names(requested, key)
        missing = set(selected[key]) - set(source[key])
        if missing:
            raise ValueError(f"Unknown {key}: {sorted(missing)}")
        # Match Cyclo's LeRobot YAML adapter: explicit lists retain their order.
        indices[feature] = [source[key].index(n) for n in selected[key]]
    return {"source": source, "selected": selected, "indices": indices}


def resolve_dimensions(padding, selected):
    dimensions = {}
    if not isinstance(padding, dict) or set(padding) != {"state_dim", "action_dim"}:
        raise ValueError("padding requires state_dim and action_dim")
    for kind in ("state", "action"):
        count = len(selected[f"{kind}_names"])
        dimension = padding[f"{kind}_dim"]
        if dimension == "selected":
            dimension = count
        if type(dimension) is not int or dimension < count:
            raise ValueError(f"{kind} padding must not truncate selected channels")
        dimensions[f"{kind}_dim"] = dimension
    return dimensions


def read_selection(path):
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in result:
                raise ValueError("Selection YAML contains a duplicate or non-string key")
            result[key] = loader.construct_object(value_node, deep=True)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
    value = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    if not isinstance(value, dict) or set(value) != set(VECTORS):
        raise ValueError("Selection YAML requires exactly state_names and action_names")
    for key in VECTORS:
        if value[key] != "all":
            _names(value[key], key)
    return deepcopy(value)
