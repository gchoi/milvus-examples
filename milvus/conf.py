"""
This is an example config file.
 - Regarding additional loguru settings, see https://loguru.readthedocs.io/en/stable/
"""

import sys

from loguru import logger


class Config:
    DEBUG = False
    TESTING = False
    LOGURU_SETTINGS = {}


class DevConfig(Config):
    DEBUG = True
    LOGURU_SETTINGS = {
        "handlers": [
            dict(
                sink=sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <yellow>{message}</yellow>",
                level="DEBUG"
            )
        ],
        "levels": []
    }


class StagingConfig(Config):
    LOGURU_SETTINGS = {
        "handlers": [
            dict(
                sink=sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <yellow>{message}</yellow>",
                level="DEBUG",
            ),
            dict(
                sink="./logs/log_{time}.log",
                enqueue=True,
                serialize=True,
                rotation="12:00",  # new log file is generated every day at 12:00
                retention="3 days",  # log file will be deleted after 3 days
            ),
        ],
        "levels": [],
    }


class ProductionConfig(Config):
    LOGURU_SETTINGS = {
        "handlers": [
            dict(
                sink=sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <yellow>{message}</yellow>",
                level="DEBUG",
            ),
            dict(
                sink="./logs/log_{time}.log",
                enqueue=True,
                serialize=True,
                rotation="12:00",  # new log file is generated every day at 12:00
                compression="zip",  # log files will be compressed
                retention="10 days",  # log file will be deleted after 10 days
            ),
        ],
        "levels": [],
    }


class Logger:
    def __init__(self, env: str = "dev"):
        match env:
            case "dev":
                config = DevConfig()
            case "prod":
                config = ProductionConfig()
            case "staging":
                config = StagingConfig()
            case _:
                logger.error(f"Invalid environment: {env}")
                raise ValueError(f"Invalid environment: {env}")

        logger.remove()
        logger.configure(**config.LOGURU_SETTINGS)
        self.logger = logger
        return

    def info(self, msg):
        self.logger.info(msg)

    def success(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def error(self, msg):
        self.logger.error(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def critical(self, msg):
        self.logger.critical(msg)

    def __call__(self, *args, **kwargs):
        return self.logger
