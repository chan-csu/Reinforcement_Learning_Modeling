import cobra
from mimetypes import init
from turtle import color
import Toolkit as tk
import ToyModel as tm
import torch
import torch.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import time
import ray
import os
import seaborn  as sns
import matplotlib.pyplot as plt
import warnings
import json
import multiprocessing as mp
model_1=cobra.io.read_sbml_model("iAF1260.xml")
model_1.remove_reactions()
agent1=tk.Agent("agent1",
				model=,
				actor_network=tk.NN,
				critic_network=tk.NN,
				clip=0.1,
				lr_actor=0.0001,
				lr_critic=0.001,
				grad_updates=1,
				optimizer_actor=torch.optim.Adam,
				optimizer_critic=torch.optim.Adam,       
				observables=['agent1','agent2','S',"A","B"],
				actions=['EX_A_sp1','EX_B_sp1'],
				gamma=1,
				)

agent2=tk.Agent("agent2",
				model=tm.Toy_Model_NE_2,
				actor_network=tk.NN,
				critic_network=tk.NN,
				clip=0.1,
				lr_actor=0.0001,
				lr_critic=0.001,
				grad_updates=1,
				optimizer_actor=torch.optim.Adam,
				optimizer_critic=torch.optim.Adam,       
				observables=['agent1','agent2','S',"A","B"],
				actions=['EX_A_sp2','EX_B_sp2'],
				gamma=1
)