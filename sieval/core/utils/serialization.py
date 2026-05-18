import importlib
import warnings
from collections.abc import Iterable
from contextlib import suppress
from typing import Any

_result_type_registry: dict[str, type] = {}
_scanned_modules: set[str] = set()
global_type_registry: dict[str, type] = dict(_result_type_registry)
_warned_legacy_classname_lookup = False


def _warn_legacy_classname_lookup() -> None:
    global _warned_legacy_classname_lookup
    if _warned_legacy_classname_lookup:
        return
    warnings.warn(
        "Class-name-only deserialization lookup is deprecated and will be removed "
        "in a future release. Please rely on module-qualified lookup "
        "(__sieval_mod__ + __sieval_cls__).",
        DeprecationWarning,
        stacklevel=3,
    )
    _warned_legacy_classname_lookup = True


def sieval_record(cls):
    """
    Decorator marking a result type as serializable/deserializable with type metadata.
    Stored in _result_type_registry for reconstruction.
    """
    cls.__sieval_record__ = True
    class_name = cls.__name__
    module_name = cls.__module__
    # Keep class-name key for backward compatibility
    _result_type_registry.setdefault(class_name, cls)
    # Add module-qualified key to avoid cross-module class-name collisions
    _result_type_registry[f"{module_name}:{class_name}"] = cls
    return cls


def register_types(
    registry: dict[str, type], types: Iterable[type], module_name: str | None
) -> None:
    """Scan *types* for ``@sieval_record``-marked classes and add them to *registry*."""
    if module_name and module_name in _scanned_modules:
        return
    for t in types:
        if isinstance(t, type) and getattr(t, "__sieval_record__", False):
            class_name = t.__name__
            qualified_name = f"{t.__module__}:{class_name}"
            registry.setdefault(class_name, t)
            registry[qualified_name] = t
            global_type_registry.setdefault(class_name, t)
            global_type_registry[qualified_name] = t
    if module_name:
        _scanned_modules.add(module_name)


def obj_to_dict(obj: Any, add_type: bool) -> Any:
    """Convert *obj* to a JSON-compatible dict, optionally embedding type markers."""
    if obj is None or isinstance(obj, str | int | float | bool):
        return obj
    if isinstance(obj, dict):
        return {k: obj_to_dict(v, add_type) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [obj_to_dict(v, add_type) for v in obj]
    if isinstance(obj, tuple):
        return {
            "__sieval_cls__": "tuple",
            "items": [obj_to_dict(v, add_type) for v in obj],
        }
    if isinstance(obj, set):
        return {
            "__sieval_cls__": "set",
            "items": [
                obj_to_dict(v, add_type) for v in sorted(obj, key=lambda x: repr(x))
            ],
        }
    data = {}
    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if v is not None:
                data[k] = obj_to_dict(v, add_type)
    elif hasattr(obj, "__slots__"):
        for s in obj.__slots__:
            with suppress(AttributeError):
                v = getattr(obj, s)
                if v is not None:
                    data[s] = obj_to_dict(v, add_type)
    else:
        return repr(obj)
    if add_type and getattr(obj.__class__, "__sieval_record__", False):
        data["__sieval_mod__"] = obj.__class__.__module__
        data["__sieval_cls__"] = obj.__class__.__name__
    return data


def dict_to_obj(val: Any, registry: dict[str, type]) -> Any:
    """Reconstruct typed objects from dicts using ``__sieval_cls__`` markers."""
    if isinstance(val, list):
        return [dict_to_obj(v, registry) for v in val]
    if isinstance(val, dict) and "__sieval_cls__" not in val:
        return {k: dict_to_obj(v, registry) for k, v in val.items()}
    if isinstance(val, dict):
        cls_name = val.get("__sieval_cls__")
        if cls_name in ("tuple", "set"):
            items = [dict_to_obj(v, registry) for v in val.get("items", [])]
            return tuple(items) if cls_name == "tuple" else set(items)
        mod_name = val.get("__sieval_mod__")
        payload = {
            k: dict_to_obj(v, registry)
            for k, v in val.items()
            if k not in ("__sieval_cls__", "__sieval_mod__")
        }
        target: type | None = None
        if isinstance(mod_name, str):
            target = registry.get(f"{mod_name}:{cls_name}")
            if target is None:
                fallback = registry.get(cls_name)
                if (
                    isinstance(fallback, type)
                    and getattr(fallback, "__module__", None) == mod_name
                ):
                    # Deprecated compatibility path; remove after migration window.
                    _warn_legacy_classname_lookup()
                    target = fallback

        if target is None and mod_name:
            with suppress(Exception):
                mod = importlib.import_module(mod_name)
                cand = getattr(mod, cls_name, None)
                if isinstance(cand, type) and getattr(cand, "__sieval_record__", False):
                    target = cand
                    registry.setdefault(cls_name, target)
                    registry[f"{mod_name}:{cls_name}"] = target
        if target is None and not mod_name:
            # Deprecated compatibility path for payloads missing __sieval_mod__.
            _warn_legacy_classname_lookup()
            target = registry.get(cls_name)
        if target is None:
            return payload
        with suppress(Exception):
            return target(**payload)
        return payload
    return val
