
from src import get_logger
logger = get_logger()

logger.debug("initialized")

from src.sim2real.siminterface import SimInterface
from src.sim2real.abstractinterface import Sim2RealInterface
from src.sim2real.realinterface import RealInterface