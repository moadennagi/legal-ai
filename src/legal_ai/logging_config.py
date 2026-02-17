import logging
import sys


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s -%(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("pipeline.log")],
    )


setup_logging()
