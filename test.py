import torch
import pickle
from core import ddpg
from core.mod_neuro_evo import SSNE

if __name__ == "__main__":
    actor1 = ddpg.Actor()