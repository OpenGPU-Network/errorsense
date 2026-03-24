import os

import pytest

from errorsense.llm import LLMConfig, DEFAULT_BASE_URL, DEFAULT_MODEL


class TestLLMConfigDefaults:
    def test_bare_config_uses_builtin_defaults(self):
        cfg = LLMConfig()
        assert cfg.model == DEFAULT_MODEL
        assert cfg.base_url == DEFAULT_BASE_URL
        assert cfg.api_key == ""

    def test_explicit_args_override_defaults(self):
        cfg = LLMConfig(api_key="sk-123", model="custom-model", base_url="https://example.com")
        assert cfg.api_key == "sk-123"
        assert cfg.model == "custom-model"
        assert cfg.base_url == "https://example.com"


class TestLLMConfigEnvVars:
    def test_env_vars_override_defaults(self, monkeypatch):
        monkeypatch.setenv("ERRORSENSE_LLM_API_KEY", "sk-env")
        monkeypatch.setenv("ERRORSENSE_MODEL", "env-model")
        monkeypatch.setenv("ERRORSENSE_LLM_URL", "https://env.example.com")

        cfg = LLMConfig()
        assert cfg.api_key == "sk-env"
        assert cfg.model == "env-model"
        assert cfg.base_url == "https://env.example.com"

    def test_explicit_args_override_env_vars(self, monkeypatch):
        monkeypatch.setenv("ERRORSENSE_LLM_API_KEY", "sk-env")
        monkeypatch.setenv("ERRORSENSE_MODEL", "env-model")
        monkeypatch.setenv("ERRORSENSE_LLM_URL", "https://env.example.com")

        cfg = LLMConfig(api_key="sk-arg", model="arg-model", base_url="https://arg.example.com")
        assert cfg.api_key == "sk-arg"
        assert cfg.model == "arg-model"
        assert cfg.base_url == "https://arg.example.com"

    def test_partial_env_vars(self, monkeypatch):
        monkeypatch.setenv("ERRORSENSE_MODEL", "env-model")

        cfg = LLMConfig()
        assert cfg.api_key == ""
        assert cfg.model == "env-model"
        assert cfg.base_url == DEFAULT_BASE_URL
