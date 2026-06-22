from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from hashlib import sha256
from threading import RLock
from typing import Iterator

from AI_agent.utils.config_handler import agent_config


@dataclass(frozen=True)
class UserContext:
    session_id: str
    user_id: str
    city: str
    current_month: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class SessionContextStore:
    """Keeps demo user information stable for the lifetime of a session."""

    def __init__(self) -> None:
        self._contexts: dict[str, UserContext] = {}
        self._lock = RLock()
        self._user_ids = tuple(agent_config.get("demo_user_ids", [str(i) for i in range(1001, 1011)]))
        self._cities = tuple(agent_config.get("demo_cities", ["深圳", "合肥", "杭州"]))
        self._default_month = str(agent_config.get("default_report_month", "2025-12"))

    def get_or_create(self, session_id: str) -> UserContext:
        normalized_id = session_id.strip() or "anonymous"
        with self._lock:
            if normalized_id not in self._contexts:
                digest = sha256(normalized_id.encode("utf-8")).digest()
                self._contexts[normalized_id] = UserContext(
                    session_id=normalized_id,
                    user_id=self._user_ids[digest[0] % len(self._user_ids)],
                    city=self._cities[digest[1] % len(self._cities)],
                    current_month=self._default_month,
                )
            return self._contexts[normalized_id]


session_context_store = SessionContextStore()
_active_user_context: ContextVar[UserContext | None] = ContextVar("active_user_context", default=None)


def get_active_user_context() -> UserContext | None:
    return _active_user_context.get()


@contextmanager
def use_user_context(context: UserContext) -> Iterator[None]:
    token = _active_user_context.set(context)
    try:
        yield
    finally:
        _active_user_context.reset(token)
