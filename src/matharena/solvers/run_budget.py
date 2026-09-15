"""Per-attempt harness budgets, including usage reported before CLI exit."""

import math
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from harness_wrapper import Agent, TokenUsage


class RunBudget:
    def __init__(self, limits, cost_for_usage, *, priced):
        self.max_time_seconds = limits.get("max_time_seconds")
        self.max_cost_usd = limits.get("max_cost_usd") if priced else None
        self.cost_limit_grace_seconds = limits.get("cost_limit_grace_seconds")
        self.cost_limit_grace_prompt = limits.get("cost_limit_grace_prompt")
        for key in ("max_time_seconds", "max_cost_usd", "cost_limit_grace_seconds"):
            value = limits.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} must be a positive finite number")
        if self.cost_limit_grace_seconds is not None and (
            not isinstance(self.cost_limit_grace_prompt, str)
            or not self.cost_limit_grace_prompt.strip()
        ):
            raise ValueError(
                "cost_limit_grace_seconds requires cost_limit_grace_prompt"
            )
        self.cost_grace = None
        self._grace_deadline = None
        self.started_at = datetime.now(timezone.utc)
        self.started = time.monotonic()
        self.cost_for_usage = cost_for_usage
        self.reason = None
        self._lock = threading.RLock()
        self._requests = {}
        self._request_context = threading.local()
        self._native_usage = TokenUsage()

    def prompt_fields(self):
        fields = {"run_started_at": self.started_at.isoformat(timespec="seconds")}
        if self.max_time_seconds is not None:
            fields.update(
                run_time_limit_hours=f"{self.max_time_seconds / 3600:g}",
                run_deadline_at=(
                    self.started_at + timedelta(seconds=self.max_time_seconds)
                ).isoformat(timespec="seconds"),
            )
        return fields

    def capture_response(self, path, payload):
        """Coalesce per-response snapshots; never add native and API totals together."""
        kind = payload.get("type")
        key = None
        if kind == "message_start":
            response = payload.get("message", {})
            key = response.get("id")
            self._request_context.message_id = key
            raw_usage = response.get("usage")
        elif kind == "message_delta":
            key = getattr(self._request_context, "message_id", None)
            raw_usage = payload.get("usage")
            if key is None:
                return
        else:
            response = payload.get("response", payload)
            if not isinstance(response, Mapping):
                return
            key = response.get("id")
            raw_usage = response.get("usage")
            google = payload.get("usageMetadata")
            if isinstance(google, Mapping):
                raw_usage = {
                    "input_tokens": google.get("promptTokenCount", 0),
                    "output_tokens": google.get("candidatesTokenCount", 0)
                    + google.get("thoughtsTokenCount", 0),
                    "cache_read_tokens": google.get("cachedContentTokenCount", 0),
                }
        usage = Agent._normalize_token_usage(raw_usage)
        if usage is None:
            return
        with self._lock:
            # Chat Completions/Gemini proxies emit their final usage once,
            # sometimes without an ID. Responses/Anthropic have stable IDs.
            key = (path, key) if key is not None else object()
            previous = self._requests.get(key, TokenUsage())
            self._requests[key] = self._max_usage(previous, usage)
        self.exceeded()

    @staticmethod
    def _max_usage(*usages):
        return TokenUsage(
            **{
                field: max(getattr(usage, field) for usage in usages)
                for field in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                )
            }
        )

    def usage(self, native=None):
        with self._lock:
            if native is not None:
                self._native_usage = self._max_usage(self._native_usage, native)
            provider = TokenUsage(
                **{
                    field: sum(
                        getattr(usage, field) for usage in self._requests.values()
                    )
                    for field in (
                        "input_tokens",
                        "output_tokens",
                        "cache_read_tokens",
                        "cache_write_tokens",
                    )
                }
            )
            return self._max_usage(self._native_usage, provider)

    def exceeded(self):
        with self._lock:
            if self.reason is None:
                if (
                    self.max_time_seconds is not None
                    and time.monotonic() - self.started >= self.max_time_seconds
                ):
                    self.reason = "time_limit"
                elif self.cost_grace is not None:
                    if (
                        not self.cost_grace["completed"]
                        and time.monotonic() >= self._grace_deadline
                    ):
                        self.reason = "cost_grace_timeout"
                elif (
                    self.max_cost_usd is not None
                    and self.cost_for_usage(self.usage()) >= self.max_cost_usd
                ):
                    self.reason = "cost_limit"
            return self.reason is not None

    def begin_cost_grace(self):
        """Allow one timed final answer after stopping at the dollar limit."""
        with self._lock:
            if (
                self.reason != "cost_limit"
                or self.cost_limit_grace_seconds is None
                or self.cost_grace is not None
            ):
                return None
            now = time.monotonic()
            seconds = self.cost_limit_grace_seconds
            if self.max_time_seconds is not None:
                seconds = min(seconds, self.started + self.max_time_seconds - now)
            if seconds <= 0:
                self.reason = "time_limit"
                return None
            started_at = datetime.now(timezone.utc)
            deadline_at = (started_at + timedelta(seconds=seconds)).isoformat(
                timespec="seconds"
            )
            prompt = self.cost_limit_grace_prompt.format(
                max_cost_usd=self.max_cost_usd,
                cost_limit_grace_minutes=f"{seconds / 60:g}",
                cost_limit_deadline_at=deadline_at,
            )
            self.cost_grace = {
                "seconds": seconds,
                "started_at": started_at.isoformat(),
                "deadline_at": deadline_at,
                "completed": False,
            }
            self._grace_deadline = now + seconds
            self.reason = None
            return prompt

    def finish_cost_grace(self, completed):
        with self._lock:
            self.exceeded()
            if completed and self.reason is None:
                self.cost_grace["completed"] = True
            elif self.reason is None:
                self.reason = "cost_limit"

    def metadata(self):
        self.exceeded()
        return {
            "max_time_seconds": self.max_time_seconds,
            "max_cost_usd": self.max_cost_usd,
            "started_at": self.started_at.isoformat(),
            "deadline_at": self.prompt_fields().get("run_deadline_at"),
            "exceeded": self.reason,
            "usage_may_be_incomplete": self.reason is not None
            or self.cost_grace is not None,
            **(
                {"cost_limit_grace": dict(self.cost_grace)}
                if self.cost_grace is not None
                else {}
            ),
        }
