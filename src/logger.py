import sys
from logging import INFO, Logger, StreamHandler, getLogger


def get_logger(name: str, log_level: str = INFO) -> Logger:
    logger = getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(log_level)
        logger.addHandler(StreamHandler(sys.stdout))
    return logger
