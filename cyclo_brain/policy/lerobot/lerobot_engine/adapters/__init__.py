"""Model adapters reuse common input assembly and robot execution services."""

from .definition import AdapterDefinition
from .registry import AdapterRegistry, resolve_adapter

__all__ = ["AdapterDefinition", "AdapterRegistry", "resolve_adapter"]
