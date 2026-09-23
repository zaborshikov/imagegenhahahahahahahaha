from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .config import settings


@dataclass
class SessionState:
    id: str
    root: Path
    current_image: Path | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    revision: int = 0

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        # Keep prompt context bounded; images remain on disk for current process.
        self.messages = self.messages[-40:]
        self.persist_history()

    def persist_history(self) -> None:
        payload = {
            "id": self.id,
            "current_image": self.current_image.name if self.current_image else None,
            "revision": self.revision,
            "messages": self.messages,
        }
        (self.root / "chat.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def create(self) -> SessionState:
        sid = uuid.uuid4().hex
        root = settings.data_dir / "sessions" / sid
        root.mkdir(parents=True, exist_ok=True)
        state = SessionState(id=sid, root=root)
        state.persist_history()
        with self._lock:
            self._sessions[sid] = state
        return state

    def get(self, sid: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(sid)
        if state is None:
            raise KeyError(sid)
        return state


sessions = SessionStore()
