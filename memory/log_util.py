from logzero import logger
import traceback

def log_exception():
    detailed_exp = traceback.format_exc()
    logger.error(detailed_exp)
