"""Logger setup for the project."""

import logging
import os
import sys

LOG_LEVEL = os.getenv(key="LOG_LEVEL", default=logging.DEBUG)
assert LOG_LEVEL in [
    logging.DEBUG,
    logging.INFO,
    logging.WARNING,
    logging.ERROR,
    logging.CRITICAL,
], "Invalid log level configured in .env file"

if not os.path.exists("logs"):
    os.makedirs("logs")

# Singleton project logger
logger = logging.getLogger(name="Main Logger")

# Avoid adding multiple handlers to the singleton logger:
if not logger.hasHandlers():

    # Create formatter for logs
    formatter = logging.Formatter(
        "[%(levelname)s|%(filename)s|%(funcName)s|L%(lineno)d] %(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create stdout handler (writes to console)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(LOG_LEVEL)
    logger.addHandler(stdout_handler)

    # Create file handler (writes to log file)
    file_handler = logging.FileHandler(
        filename="logs/main.log", mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(LOG_LEVEL)
    logger.addHandler(file_handler)

logger.setLevel(LOG_LEVEL)
