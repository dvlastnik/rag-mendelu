import logging
import os
from colorlog import ColoredFormatter

def setup_logging(log_level=None):
    if log_level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, env_level, logging.INFO)

    formatter = ColoredFormatter(
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

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logging.basicConfig(
        level=log_level,
        handlers=[handler],
        force=True
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name.upper())

def highlight_log(logger: logging.Logger, text: str, character: str = '/', length: int = 5, only_char: bool = False) -> None:
    side_string = length*character

    if only_char:
        logger.info(2*side_string + (len(text)+2)*character)
    else:
        logger.info(side_string + ' ' + text + ' ' + side_string)