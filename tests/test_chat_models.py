"""
Tests for ChatAuxen.

These tests exercise the env-var / kwarg resolution logic only; they
don't hit the network. End-to-end calls against a live Auxen instance
happen in manual smoke testing.
"""

import pytest

from langchain_auxen import ChatAuxen


SAMPLE_BASE = "https://api.auxen.ai/v1/inst_test"
SAMPLE_KEY = "auxk_test_fake"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("AUXEN_BASE_URL", raising=False)
    monkeypatch.delenv("AUXEN_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)


def test_constructs_with_explicit_options():
    model = ChatAuxen(base_url=SAMPLE_BASE, api_key=SAMPLE_KEY)
    # ChatOpenAI normalizes to openai_api_base; we should see /v1 suffix
    assert "/v1" in str(model.openai_api_base)
    assert SAMPLE_BASE in str(model.openai_api_base)


def test_appends_v1_when_missing():
    model = ChatAuxen(base_url=SAMPLE_BASE, api_key=SAMPLE_KEY)
    assert str(model.openai_api_base).rstrip("/").endswith("/v1")


def test_preserves_v1_when_present():
    """URLs that already end in /v1 should not get a second /v1 appended."""
    model = ChatAuxen(base_url=SAMPLE_BASE + "/v1", api_key=SAMPLE_KEY)
    assert not str(model.openai_api_base).rstrip("/").endswith("/v1/v1")


def test_reads_env_vars(monkeypatch):
    monkeypatch.setenv("AUXEN_BASE_URL", SAMPLE_BASE)
    monkeypatch.setenv("AUXEN_API_KEY", SAMPLE_KEY)
    model = ChatAuxen()
    assert SAMPLE_BASE in str(model.openai_api_base)


def test_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("AUXEN_BASE_URL", "https://example.com")
    monkeypatch.setenv("AUXEN_API_KEY", "from_env")
    model = ChatAuxen(base_url=SAMPLE_BASE, api_key=SAMPLE_KEY)
    assert SAMPLE_BASE in str(model.openai_api_base)


def test_llm_type_identifier():
    model = ChatAuxen(base_url=SAMPLE_BASE, api_key=SAMPLE_KEY)
    assert model._llm_type == "auxen-chat"


def test_default_model_set():
    """When no model is passed, a sensible default should be applied."""
    model = ChatAuxen(base_url=SAMPLE_BASE, api_key=SAMPLE_KEY)
    assert model.model_name  # any non-empty string is fine
