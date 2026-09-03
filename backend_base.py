from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional

from config import THINK_CLOSE, THINK_OPEN
from session import Session, messages_to_prompt_text


@contextmanager
def closing_stream(stream: Any) -> Iterator[Any]:
    # A Stop breaks out of the library's own generator; close it here so its
    # finally (which frees Metal buffers) runs on this thread instead of
    # whenever the GC happens to collect it.
    try:
        yield stream
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def with_max_tokens(params: Any, max_tokens: int) -> Any:
    p = copy.copy(params)  # shallow: dataclass fields only, no __init__ replay
    p.max_tokens = max(16, int(max_tokens))
    return p


def budget_thinking(
    session: Session,
    stream: Generator[str, None, None],
    budget: int,
    resume: Callable[[str, int], Generator[str, None, None]],
) -> Generator[str, None, None]:
    # Qwen's thinking budget is a harness job, not a model one: count the tokens
    # spent inside an open <think>, and once they pass the cap force the block
    # shut and resume from the truncated reasoning. Streams yield one token at a
    # time cumulatively, so a yield count is the token count.
    if budget <= 0:
        yield from stream
        return
    sent, spent, capped = "", 0, False
    with closing_stream(stream):
        for text in stream:
            if THINK_OPEN in text and THINK_CLOSE not in text:
                spent += 1
                if spent > budget:
                    capped = True
                    break
            yield text
            sent = text
    # A Stop landing on the cap must not buy a whole extra prefill.
    if not capped or session.stop_requested:
        return
    head = f"{sent}\n{THINK_CLOSE}\n\n"
    yield head
    yield from resume(head, spent)


@dataclass
class Prepared:
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    images: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)


class Backend(ABC):
    id: str = ""
    is_vision: bool = False

    # ---- lifecycle -------------------------------------------------------
    @abstractmethod
    def load(self, params: Any) -> tuple[str, Optional[int]]:
        # returns (status_markdown, detected_ctx|None); None = leave ctx slider alone
        ...

    @abstractmethod
    def unload(self) -> str: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    # ---- budgeting -------------------------------------------------------
    @abstractmethod
    def context_window(self, params: Any) -> int:
        ...

    @abstractmethod
    def count_prompt_tokens(self, messages: List[Dict[str, Any]]) -> int:
        ...

    # ---- generation ------------------------------------------------------
    @abstractmethod
    def prepare(self, session: Session, params: Any, web_ctx: str) -> Prepared:
        ...

    @abstractmethod
    def generate_stream(self, session: Session, params: Any, prepared: Prepared) -> Generator[str, None, None]:
        ...

    # ---- optional extensions (base no-ops; VlmBackend overrides) ---------
    def media_reserve(self, session: Session, params: Any) -> int:
        return 0

    def media_reserve_with_counts(
        self, session: Session, params: Any, pending_media: Optional[Dict[str, List[str]]] = None,
    ) -> tuple[int, Dict[str, int]]:
        return 0, {}

    def media_overflow_message(
        self, session: Session, params: Any, ctx: int, media_reserve: int, web_reserve: int,
    ) -> Optional[str]:
        return None

    def error_message(self, e: Exception) -> str:
        return f"❌ Generation error: {e}"

    # ---- shared helper ---------------------------------------------------
    @staticmethod
    def fallback_count(messages: List[Dict[str, Any]]) -> int:
        prompt_text = messages_to_prompt_text(messages)
        return max(1, len(prompt_text) // 4)
