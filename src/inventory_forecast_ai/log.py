"""Structured logging setup (structlog)."""

import logging

import structlog


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structlog for the whole application.

    Args:
        level: Minimum log level name (e.g. "INFO", "DEBUG").
        json_logs: Emit JSON lines instead of pretty console output.
    """
    level_number = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    logging.basicConfig(level=level_number, format="%(levelname)s %(name)s: %(message)s")

    renderer: structlog.typing.Processor
    renderer = structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level_number),
        cache_logger_on_first_use=True,
    )
