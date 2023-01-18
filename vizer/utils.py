import logging
logging.basicConfig(format="%(threadName)s %(asctime)s %(levelname)s %(message)s", level=logging.INFO)

def get_logger(name: str)-> logging.Logger:
    logger = logging.getLogger(name)
    return logger
