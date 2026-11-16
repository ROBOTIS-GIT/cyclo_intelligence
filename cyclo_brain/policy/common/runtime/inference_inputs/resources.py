"""Shared retained-memory accounting across callback history and feature state."""

import threading


class Budget:
    def __init__(self, limit=256 * 1024 * 1024):
        if type(limit) is not int or limit <= 0:
            raise ValueError("session memory limit must be positive")
        self.limit = limit
        self._sizes = {}
        self._used = 0
        self._lock = threading.Lock()

    @property
    def used(self):
        with self._lock:
            return self._used

    def resize(self, key, size):
        if type(size) is not int or size < 0:
            raise ValueError("retained allocation size must be non-negative")
        with self._lock:
            total = self._used - self._sizes.get(key, 0) + size
            if total > self.limit:
                raise MemoryError("session retained-memory budget exceeded")
            self._used = total
            if size:
                self._sizes[key] = size
            else:
                self._sizes.pop(key, None)

    def transfer(self, source, destination):
        with self._lock:
            size = self._sizes.pop(source)
            self._used -= self._sizes.get(destination, 0)
            self._sizes[destination] = size


def retained_size(value):
    """Storage estimate without importing a tensor framework or allocating a copy."""
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    if isinstance(value, (str, bool, int, float)) or value is None:
        return len(str(value).encode()) + 32
    if isinstance(value, (tuple, list, dict)):
        items = value.items() if isinstance(value, dict) else enumerate(value)
        return 64 + sum(retained_size(item) + len(str(key)) for key, item in items)
    raise TypeError("register a measurable numeric tensor or structured input value")
