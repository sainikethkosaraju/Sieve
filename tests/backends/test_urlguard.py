from __future__ import annotations

import pytest

from sieve.backends import BackendUrlError, validate_backend_url


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data",
        "http://[fd00:ec2::254]/",
        "http://metadata.google.internal/",
        "file:///etc/passwd",
        "gopher://127.0.0.1/",
        "ftp://127.0.0.1/",
        "http://user:pw@127.0.0.1:8080/",
        "http://127.0.0.1@evil.example/",
        "http://evil.example/",
        "http://127.0.0.1.evil.example/",
        "http://2130706433/",  # decimal-encoded loopback is not an allow-listed string
        "http://0x7f.0.0.1/",
        "http://0.0.0.0:8080/",
        "http://169.254.1.1/",
        "http://127.0.0.1:8080/?x=1",
        "http://127.0.0.1:8080/#frag",
        "http://127.0.0.1:99999/",
        "http:///nohost",
        "",
    ],
)
def test_rejected(url: str) -> None:
    with pytest.raises(BackendUrlError):
        validate_backend_url(url)


def test_metadata_blocked_even_if_allow_listed() -> None:
    with pytest.raises(BackendUrlError):
        validate_backend_url("http://169.254.169.254/", ["169.254.169.254"])
    with pytest.raises(BackendUrlError):
        validate_backend_url("http://metadata.google.internal/", ["metadata.google.internal"])


def test_accepted_and_normalised() -> None:
    assert validate_backend_url("http://127.0.0.1:8081/") == "http://127.0.0.1:8081"
    assert validate_backend_url("http://LOCALHOST:8081") == "http://localhost:8081"
    assert validate_backend_url("http://[::1]:8081/v1/") == "http://[::1]:8081/v1"
    assert (
        validate_backend_url("http://llama-small:8080", ["llama-small"])
        == "http://llama-small:8080"
    )
    assert validate_backend_url("https://llama.test", ["llama.test"]) == "https://llama.test"
