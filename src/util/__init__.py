from src import logging

logger = logging.getLogger(__name__)

logger.debug("initialized")

from .log_exceptions import log_exceptions
__all__ = ["log_exceptions"]