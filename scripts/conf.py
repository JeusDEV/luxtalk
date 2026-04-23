"""Load and setup configurations."""

import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


def logging_setup(loggingLevel: int = logging.INFO) -> None:
    """Setup the configuration."""
    logging.basicConfig(
        level=loggingLevel,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(
            console=Console(
                theme=Theme({
                    "logging.level.info": "#00B000",
                    "logging.level.warning": "#FFA600",
                    "logging.level.error": "bold " + "#FF0000",
                    "log.time": "#6C6C6C",
                    "log.path": "#5F5F5F",
                })
            ),
            rich_tracebacks=True,
            markup=True,
        )],
    )


if __name__ == "__main__":
    logging.warning("(!) This is a config loader bruh, use as module")
