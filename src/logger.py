from logging import Logger, INFO, getLogger, StreamHandler
import sys


def get_logger(name: str, log_level: str = INFO) -> Logger:
    logger = getLogger(name)
    logger.setLevel(log_level)
    logger.addHandler(StreamHandler(sys.stdout))
    return logger
