from pythonjsonlogger import jsonlogger
import logging

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S.%f%z",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )
    handler.setFormatter(formatter)
    logger.handlers = [handler]