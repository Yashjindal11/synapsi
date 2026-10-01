class SynapSIError(Exception):
    """Base class for all SynapSI errors."""


class ConfigError(SynapSIError):
    """Invalid or incomplete configuration."""


class ProviderError(SynapSIError):
    """A model provider failed.

    ``retryable`` marks transient failures (rate limits, timeouts, 5xx).
    """

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class StructuredOutputError(SynapSIError):
    """A model did not return output matching the requested schema."""

    def __init__(self, message: str, *, raw: str = ""):
        super().__init__(message)
        self.raw = raw


class StopWorkflow(SynapSIError):
    """Raised by a step to end the workflow early (not an error)."""

    def __init__(self, reason: str = "stopped"):
        super().__init__(reason)
        self.reason = reason


class BudgetExceeded(SynapSIError):
    """A run exceeded its configured cost, token, or call budget."""
