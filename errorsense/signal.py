from __future__ import annotations

import traceback
from types import MappingProxyType
from typing import Any


def _deep_freeze(obj: Any) -> Any:
    """Recursively freeze dicts into MappingProxyType and lists into tuples.

    Note: lists become tuples. Code receiving signal data should check
    for Sequence, not list specifically.
    """
    if isinstance(obj, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_deep_freeze(item) for item in obj)
    return obj


class Signal:
    """Immutable container for error/event data.

    All values are deep-frozen at construction time — skills get a
    truly read-only view. Dict-like access for convenience.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        raw = {**(data or {}), **kwargs}
        object.__setattr__(self, "_data", _deep_freeze(raw))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> Any:
        return self._data.keys()

    def values(self) -> Any:
        return self._data.values()

    def items(self) -> Any:
        return self._data.items()

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable deep copy of the signal data."""
        return _thaw(self._data)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Signal is immutable")

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("Signal is immutable")

    def __delitem__(self, key: str) -> None:
        raise TypeError("Signal is immutable")

    def __repr__(self) -> str:
        return f"Signal({dict(self._data)!r})"

    @classmethod
    def from_http(
        cls,
        status_code: int,
        body: str = "",
        headers: dict[str, str] | None = None,
    ) -> Signal:
        return cls(
            {
                "status_code": status_code,
                "body": body,
                "headers": headers or {},
            }
        )

    @classmethod
    def from_grpc(cls, code: int, details: str = "") -> Signal:
        return cls({"grpc_code": code, "details": details})

    @classmethod
    def from_exception(cls, exc: BaseException) -> Signal:
        return cls(
            {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
            }
        )


def _thaw(obj: Any) -> Any:
    """Recursively convert MappingProxyType back to dict and tuples to lists."""
    if isinstance(obj, (MappingProxyType, dict)):
        return {k: _thaw(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_thaw(item) for item in obj]
    return obj
