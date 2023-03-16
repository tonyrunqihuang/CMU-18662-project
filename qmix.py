import numpy as np
import torch
import torch.nn.functional as F
from agent import Agent
from buffer import Buffer


class QMIX:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, dim_info, capacity):
        pass