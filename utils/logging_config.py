import logging
from colorlog import ColoredFormatter

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def get_logger(name: str) -> logging.Logger:
    name = name.upper()
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

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logger

def highlight_log(logger: logging.Logger, text: str, character: str = '/', length: int = 5, only_char: bool = False) -> None:
    side_string = length*character

    if only_char:
        logger.info(2*side_string + (len(text)+2)*character)
    else:
        logger.info(side_string + ' ' + text + ' ' + side_string)