"""Typed exceptions for MeetScribe."""

import logging

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, SpeachesAPIError) and exc.is_transient


speaches_retry = retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=lambda rs: logger.warning(
        "Retrying after %s (attempt %d)",
        rs.outcome.exception() if rs.outcome else None,
        rs.attempt_number,
    ),
    reraise=True,
)


class MeetScribeError(Exception):
    """Base exception for all MeetScribe errors."""


class ConfigurationError(MeetScribeError):
    """Invalid or missing configuration."""


class PipelineError(MeetScribeError):
    """Error during audio processing pipeline."""


class SpeachesAPIError(PipelineError):
    """Error communicating with a remote Speaches server.

    Attributes:
        status_code: HTTP status code (None for connection/timeout errors).
        endpoint: The API endpoint that failed.
        detail: Response body or error message.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        endpoint: str = "",
        detail: str = "",
    ):
        super().__init__(message)
        self.status_code = status_code
        self.endpoint = endpoint
        self.detail = detail

    @property
    def is_transient(self) -> bool:
        """Whether this error is likely transient and worth retrying."""
        if self.status_code is None:
            return True  # connection/timeout errors
        return self.status_code >= 500 or self.status_code == 429
