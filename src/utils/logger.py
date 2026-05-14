import sys
from pathlib import Path

from loguru import logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(name: str = "crypto_bot") -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        LOG_DIR / f"{name}_{{time:YYYY-MM-DD}}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        format="{time:HH:mm:ss} | {level: <8} | {name} | {message}",
    )
