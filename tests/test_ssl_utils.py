from genai_bench.utils import get_requests_verify


def test_get_requests_verify_defaults_to_true(monkeypatch):
    for env_var in (
        "GENAI_BENCH_SSL_CA_BUNDLE",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
    ):
        monkeypatch.delenv(env_var, raising=False)

    assert get_requests_verify() is True


def test_get_requests_verify_prefers_genai_bench_env(monkeypatch):
    monkeypatch.setenv("GENAI_BENCH_SSL_CA_BUNDLE", "/tmp/genai-ca.pem")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/tmp/requests-ca.pem")
    monkeypatch.setenv("SSL_CERT_FILE", "/tmp/ssl-cert.pem")

    assert get_requests_verify() == "/tmp/genai-ca.pem"


def test_get_requests_verify_falls_back_to_requests_ca_bundle(monkeypatch):
    monkeypatch.delenv("GENAI_BENCH_SSL_CA_BUNDLE", raising=False)
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/tmp/requests-ca.pem")
    monkeypatch.setenv("SSL_CERT_FILE", "/tmp/ssl-cert.pem")

    assert get_requests_verify() == "/tmp/requests-ca.pem"
