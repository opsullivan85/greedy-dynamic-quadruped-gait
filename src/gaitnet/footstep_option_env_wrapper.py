
from isaaclab.managers import ManagerBasedRLEnv

class FootstepOptionEnvWrapper():
    """Modifies observations such that the terrain heightmap is replaced with footstep options.
    """

    def __init__(self, cfg, sim):