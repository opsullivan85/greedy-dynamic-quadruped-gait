import logging
logger = logging.getLogger(__name__)
del logging

logger.debug("initialized")

from .log_exceptions import log_exceptions
from .vectorpool import VectorPool
from .data_logging import data_logger, data_dir, DataType