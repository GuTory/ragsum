'''Generic logger setup function that can be used across the database.'''

import logging
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str, log_file: str = None, level=logging.INFO, max_bytes=5 * 1024 * 1024, backup_count=3
) -> logging.Logger:
    """
    Set up and return a logger. Logs to a rotating file if log_file is provided,
    otherwise logs to the console.

    :param name: Name of the logger
    :param log_file: File path for the log file
    :param level: Logging level
    :param max_bytes: Max size in bytes before rotation
    :param backup_count: Number of backup files to keep
    :return: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        if log_file:
            handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        else:
            handler = logging.StreamHandler()

        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


if __name__ == '__main__':
    test_logger = setup_logger(__name__)

    test_logger.info("App started")
    test_logger.warning("Something might be off")
    test_logger.error("Oops, an error!")
