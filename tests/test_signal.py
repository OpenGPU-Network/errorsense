import pytest

from errorsense import Signal


class TestSignal:
    def test_dict_access(self):
        s = Signal({"a": 1, "b": "hello"})
        assert s["a"] == 1
        assert s["b"] == "hello"

    def test_get_with_default(self):
        s = Signal({"a": 1})
        assert s.get("a") == 1
        assert s.get("missing") is None
        assert s.get("missing", 42) == 42

    def test_contains(self):
        s = Signal({"x": 10})
        assert "x" in s
        assert "y" not in s

    def test_immutable_setattr(self):
        s = Signal({"a": 1})
        with pytest.raises(AttributeError, match="immutable"):
            s.foo = "bar"

    def test_immutable_setitem(self):
        s = Signal({"a": 1})
        with pytest.raises(TypeError, match="immutable"):
            s["a"] = 2

    def test_immutable_delitem(self):
        s = Signal({"a": 1})
        with pytest.raises(TypeError, match="immutable"):
            del s["a"]

    def test_to_dict(self):
        data = {"a": 1, "b": 2}
        s = Signal(data)
        assert s.to_dict() == data
        # Modifying returned dict doesn't affect signal
        d = s.to_dict()
        d["c"] = 3
        assert "c" not in s

    def test_from_http(self):
        s = Signal.from_http(status_code=503, body="error", headers={"x": "y"})
        assert s["status_code"] == 503
        assert s["body"] == "error"
        assert s["headers"] == {"x": "y"}

    def test_from_http_defaults(self):
        s = Signal.from_http(status_code=200)
        assert s["body"] == ""
        assert s["headers"] == {}

    def test_from_grpc(self):
        s = Signal.from_grpc(code=14, details="unavailable")
        assert s["grpc_code"] == 14
        assert s["details"] == "unavailable"

    def test_from_exception(self):
        try:
            raise ValueError("test error")
        except ValueError as e:
            s = Signal.from_exception(e)
        assert s["exception_type"] == "ValueError"
        assert s["message"] == "test error"
        assert isinstance(s["traceback"], (list, tuple))

    def test_keys_values_items(self):
        s = Signal({"a": 1, "b": 2})
        assert set(s.keys()) == {"a", "b"}
        assert set(s.values()) == {1, 2}
        assert set(s.items()) == {("a", 1), ("b", 2)}

    def test_repr(self):
        s = Signal({"x": 1})
        assert "Signal" in repr(s)
        assert "x" in repr(s)

    def test_kwargs_constructor(self):
        s = Signal(status_code=404, body="not found")
        assert s["status_code"] == 404
        assert s["body"] == "not found"

    def test_deep_immutability(self):
        s = Signal.from_http(status_code=500, body="err", headers={"content-type": "text/html"})
        with pytest.raises(TypeError):
            s["headers"]["content-type"] = "mutated"

    def test_to_dict_returns_mutable_copy(self):
        s = Signal.from_http(status_code=500, headers={"x": "y"})
        d = s.to_dict()
        d["headers"]["x"] = "mutated"
        assert s["headers"]["x"] == "y"  # original unchanged
