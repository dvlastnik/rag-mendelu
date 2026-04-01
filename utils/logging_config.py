import logging
import os
from colorlog import ColoredFormatter

def setup_logging(log_level=None, silent_console=False, log_file=None):
    if log_level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, env_level, logging.INFO)

    handlers = []

    if not silent_console:
        colored_formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            }
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(colored_formatter)
        handlers.append(console_handler)

    if log_file:
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("docling").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name.upper())

def highlight_log(logger: logging.Logger, text: str, character: str = '/', length: int = 5, only_char: bool = False) -> None:
    side_string = length*character

    if only_char:
        logger.info(2*side_string + (len(text)+2)*character)
    else:
        logger.info(side_string + ' ' + text + ' ' + side_string)