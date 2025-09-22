import argparse
import time

import numpy as np
import pygame
import torch
import torch.nn as nn

import src
from src.contactnet.debug import view_footstep_cost_map
from src.contactnet.train import QuadrupedDataset, QuadrupedModel
from src.contactnet.util import get_checkpoint_path, get_dataset_paths
from src import get_logger

logger = get_logger()


def main():
    while True:
        logger.info("WORKING")
    pygame.joystick.init()

    try:
        joystick = pygame.joystick.Joystick(0)
    except pygame.error as e:
        logger.fatal("Could not find valid controller")
        raise


    while True:
        axes = joystick.get_numaxes()
        logger.info(f"Number of axes: {axes}")

        for i in range(axes):
            axis = joystick.get_axis(i)
            logger.info(f"\tAxis {i} value: {axis:>6.3f}")



if __name__ == "__main__":
    main()