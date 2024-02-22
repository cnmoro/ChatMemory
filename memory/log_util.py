import traceback
from logzero import logger

def log_exception():
    detailed_exp = traceback.format_exc()
    logger.error(detailed_exp)
