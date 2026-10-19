"""Provider-neutral, bounded structured calls. Credentials are read only from env."""
from __future__ import annotations

import json
import os
import re
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class RoleSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-6.1-sol"
    reasoning_effort: Literal["low", "medium", "high", "xhigh"] = "high"
    max_output_tokens: int = Field(default=1800, ge=256, le=8192)


class RunLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_seconds: float = Field(default=180, gt=0, le=3600)
    max_candidates: int = Field(default=20, ge=2, le=256)
    max_expert_calls: int = Field(default=1, ge=0, le=10)
    max_api_calls: int = Field(default=24, ge=1, le=512)
    api_timeout_s: float = Field(default=30, gt=0, le=45)
    structured_retries: int = Field(default=2, ge=0, le=2)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    designer: RoleSettings = Field(default_factory=RoleSettings)
    scorer: RoleSettings = Field(default_factory=RoleSettings)
    expert: RoleSettings = Field(default_factory=lambda: RoleSettings(model="gpt-6-astra"))
    limits: RunLimits = Field(default_factory=RunLimits)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Settings:
        selected = path or os.environ.get("SPICEHARNESS_CONFIG")
        raw: dict[str, Any] = {}
        if selected:
            with Path(selected).expanduser().open("rb") as handle:
                raw = tomllib.load(handle)
        # Roles have their own table so mixed-provider configuration is explicit.
        roles = raw.pop("roles", {})
        for role in ("designer", "scorer", "expert"):
            role_data = dict(roles.get(role, raw.pop(role, {})))
            for key in ("provider", "model", "reasoning_effort"):
                value = os.environ.get(f"SPICEHARNESS_{role.upper()}_{key.upper()}")
                if value:
                    role_data[key] = value
            if role_data:
                # Selecting Anthropic requires a deliberate model choice.
                if role_data.get("provider") == "anthropic" and not role_data.get("model"):
                    raise ValueError(f"Set an explicit Anthropic model for {role}")
                raw[role] = role_data
        if roles.keys() - {"designer", "scorer", "expert"}:
            raise ValueError("Unknown role in configuration")
        limits = dict(raw.get("limits", {}))
        for key in RunLimits.model_fields:
            value = os.environ.get(f"SPICEHARNESS_{key.upper()}")
            if value:
                limits[key] = value
        if limits:
            raw["limits"] = limits
        return cls.model_validate(raw)


def safe_error(error: BaseException | str) -> str:
    message = str(error)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[REDACTED]", message)[:1600]


class ProviderError(RuntimeError):
    pass


T = TypeVar("T", bound=BaseModel)


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI requires all object fields to be required, including nullable ones."""
    schema = json.loads(json.dumps(schema))
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("default", None)
            if value.get("type") == "object" and "properties" in value:
                value["additionalProperties"] = False
                value["required"] = list(value["properties"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(schema)
    return schema


class ProviderRouter:
    """Separate designer/scorer/expert roles with one shared deadline and call cap.

    ``clients`` is dependency injection for tests; no test needs live credentials.
    Every attempt, including access validation, is recorded without prompt secrets.
    """
    def __init__(self, settings: Settings, *, deadline: float | None = None,
                 on_usage: Callable[[dict[str, Any]], None] | None = None,
                 clients: dict[str, Any] | None = None, calls_used: int = 0,
                 expert_calls_used: int = 0):
        self.settings = settings
        self.deadline = deadline or time.monotonic() + settings.limits.max_seconds
        self.on_usage = on_usage or (lambda entry: None)
        self.clients = dict(clients or {})
        self.calls = calls_used
        self.expert_calls = expert_calls_used

    def _client(self, provider: str) -> Any:
        if provider in self.clients:
            return self.clients[provider]
        env = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
        if not os.environ.get(env):
            raise ProviderError(f"{env} is required for configured {provider} roles")
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.environ[env], max_retries=0,
                            timeout=self.settings.limits.api_timeout_s)
        else:
            from anthropic import Anthropic
            client = Anthropic(api_key=os.environ[env], max_retries=0,
                               timeout=self.settings.limits.api_timeout_s)
        self.clients[provider] = client
        return client

    def _begin(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderError("Run deadline reached before API call")
        if self.calls >= self.settings.limits.max_api_calls:
            raise ProviderError("Run API call limit reached")
        self.calls += 1
        return min(remaining, self.settings.limits.api_timeout_s)

    def validate_access(self) -> None:
        """Fail before autonomous work if a configured model cannot be listed."""
        for provider in sorted({getattr(self.settings, role).provider for role in
                                ("designer", "scorer", "expert")}):
            client = self._client(provider)
            requested = {getattr(self.settings, role).model for role in
                         ("designer", "scorer", "expert")
                         if getattr(self.settings, role).provider == provider}
            available: set[str] = set()
            after: str | None = None
            while requested - available:
                timeout = self._begin()
                try:
                    kwargs: dict[str, Any] = {"timeout": timeout}
                    if provider == "anthropic":
                        kwargs["limit"] = 100
                        if after:
                            kwargs["after_id"] = after
                    page = client.models.list(**kwargs)
                    entries = list(page.data)
                    available.update(item.id for item in entries)
                    self.on_usage({"role": "validation", "provider": provider,
                                   "operation": "models.list", "success": True})
                    if not getattr(page, "has_more", False) or not entries:
                        break
                    after = entries[-1].id
                except Exception as exc:  # noqa: BLE001 -- both provider SDKs have independent exception families
                    self.on_usage({"role": "validation", "provider": provider,
                                   "operation": "models.list", "success": False})
                    raise ProviderError(f"Could not validate {provider} model access: {safe_error(exc)}") from None
            missing = sorted(requested - available)
            if missing:
                raise ProviderError(f"Configured {provider} models not available: {', '.join(missing)}. "
                                    "Choose accessible model IDs explicitly; no fallback is used.")

    def structured(self, role: Literal["designer", "scorer", "expert"],
                   prompt: str, schema: type[T]) -> T:
        config = getattr(self.settings, role)
        client = self._client(config.provider)
        json_schema = _strict_schema(schema.model_json_schema())
        last_error = ""
        for attempt in range(self.settings.limits.structured_retries + 1):
            if role == "expert" and self.expert_calls >= self.settings.limits.max_expert_calls:
                raise ProviderError("Expert API call limit reached")
            timeout = self._begin()
            if role == "expert":
                self.expert_calls += 1
            entry: dict[str, Any] = {"role": role, "provider": config.provider,
                                     "model": config.model, "attempt": attempt + 1,
                                     "operation": "structured", "success": False}
            try:
                request_prompt = prompt
                if last_error:
                    request_prompt += "\nPrevious output failed schema validation. Return a conforming object."
                if config.provider == "openai":
                    response = client.responses.create(
                        model=config.model, input=request_prompt,
                        reasoning={"effort": config.reasoning_effort},
                        max_output_tokens=config.max_output_tokens,
                        text={"format": {"type": "json_schema", "name": schema.__name__,
                                         "schema": json_schema, "strict": True}}, timeout=timeout)
                    if getattr(response, "status", "completed") != "completed":
                        raise ValueError("Provider returned an incomplete response")
                    body = response.output_text
                else:
                    response = client.messages.create(
                        model=config.model, max_tokens=config.max_output_tokens,
                        messages=[{"role": "user", "content": request_prompt}],
                        output_config={"format": {"type": "json_schema", "schema": json_schema},
                                       "effort": config.reasoning_effort}, timeout=timeout)
                    if getattr(response, "stop_reason", "end_turn") in {"max_tokens", "refusal"}:
                        raise ValueError("Provider stopped before a complete structured response")
                    body = "".join(block.text for block in response.content
                                   if getattr(block, "type", "") == "text")
                usage = getattr(response, "usage", None)
                if usage:
                    entry["tokens"] = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
                parsed = schema.model_validate_json(body)
                entry["success"] = True
                self.on_usage(entry)
                return parsed
            except Exception as exc:  # noqa: BLE001 -- sanitize SDK failures before recording them
                last_error = safe_error(exc)
                entry["error"] = last_error
                self.on_usage(entry)
                # Retry malformed structured output, not authentication/network failures.
                if not isinstance(exc, (ValueError, json.JSONDecodeError)):
                    break
        raise ProviderError(f"{role} structured output failed: {last_error}")
