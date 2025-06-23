import logging

logger = logging.getLogger("resume-scorer")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s  %(name)s:%(filename)s:%(lineno)d %(message)s')
handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(handler)
