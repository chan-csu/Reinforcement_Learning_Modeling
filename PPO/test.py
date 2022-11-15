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
warnings.filterwarnings("ignore") 
agents=[]
with open('Results/Toy-NECOM_Facultative/agent1_2000.pkl','rb') as f:
    agent1 = pickle.load(f)
    agent1.observables=['agent1','agent2','S',"A","B"]
    agents.append(agent1)
with open('Results/Toy-NECOM_Facultative/agent2_2000.pkl','rb') as f:
    agent2 = pickle.load(f)
    agent2.observables=['agent1','agent2','S',"A","B"]
    agents.append(agent2)
env=tk.Environment(name="Toy-NECOM_Facultative",
                    agents=agents,
                    dilution_rate=0.01,
                    extracellular_reactions=[],
                    initial_condition={"S":100,"agent1":0.1,"agent2":0.1},
                    inlet_conditions={"S":100},
                    max_c={'S':100,
                           'agent1':10,  
                           'agent2':10,
                           'A':10,
                           'B':10,},
                            dt=0.1,
                            episode_time=100,
                            number_of_batches=5000,
                            episodes_per_batch=10,training=False)




env.reset()
concs=[]
rewards=[]
for i in range(1000):
    env.t=1000-i
    for agent in env.agents:
        agent.a,_=agent.get_actions(np.hstack([env.state[agent.observables],env.t]))
    s,r,a,_=env.step()
    concs.append(s)
    rewards.append(r)
concs=pd.DataFrame(concs,columns=env.species)
concs.plot()
plt.show(
)

    