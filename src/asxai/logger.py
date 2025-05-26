import logging
import config


def get_logger(name: str, level: str = config.LOG_LEVEL) -> logging.Logger:
    logger = logging.getLogger(name)
    logging_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(logging_level)

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "\n%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
