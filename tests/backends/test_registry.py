import pytest

from angel_ai.backends.cpu import CPUBackend
from angel_ai.backends.registry import available_backends, get_backend
from angel_ai.errors import UnsupportedConfigError


def test_cpu_backend_always_available():
    assert "cpu" in available_backends()


def test_get_backend_returns_cpu_instance():
    backend = get_backend("cpu")
    assert isinstance(backend, CPUBackend)
    assert backend.name == "cpu"


def test_get_backend_auto_never_fails():
    backend = get_backend("auto")
    assert backend.is_available()


def test_get_backend_unknown_name_raises():
    with pytest.raises(UnsupportedConfigError):
        get_backend("quantum-computer")


def test_cpu_backend_does_not_support_quantization():
    backend = CPUBackend()
    assert backend.supports_quantization("qlora") is False
    assert backend.supports_quantization("int8") is False
