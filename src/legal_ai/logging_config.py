import logging
import sys
from legal_ai.settings import settings


def setup_logging(level: int = settings.log_level):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s -%(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("pipeline.log")],
    )


setup_logging()
