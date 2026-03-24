"""Structured logging formatter.

All pipeline log calls pass data via extra={}. The formatter renders it as
key=value pairs appended to the message. To switch to JSON, replace
StructuredFormatter with a JSON-based one — no log call changes needed.
"""

import logging

# Attributes that LogRecord creates internally plus those injected by handlers/frameworks
# (uvicorn, asyncio, colorama, etc.) — everything else is extra context.
_BUILTIN_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None, None).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
    "stack_info",
    "exc_text",
    "exc_info",
    "relativeCreated",
    "msecs",
    "color",
    "color_message",
}


class StructuredFormatter(logging.Formatter):
    """Formatter that appends extra fields as key=value pairs.

    Output: "2026-03-24 10:15:08 [INFO] pipeline.vad: VAD completed | file=meeting.wav segments=42"
    """

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {k: v for k, v in record.__dict__.items() if k not in _BUILTIN_ATTRS}
        if not extras:
            return base
        ctx = " ".join(
            f'{k}="{v}"' if " " in str(v) else f"{k}={v}" for k, v in extras.items()
        )
        return f"{base} | {ctx}"
