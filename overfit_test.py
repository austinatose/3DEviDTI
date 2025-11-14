import os
import torch
import pandas as pd
import numpy as np
from config.cfg import get_cfg_defaults
from dataset import MyDataset
from model import Model
from solver import Solver

cfg = get_cfg_defaults()
model = Model(cfg)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

solver = Solver(model, cfg, device=device, optim=torch.optim.Adam, loss_fn=dirichlet_loss, eval=None)
