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
import plotly.express as px
warnings.filterwarnings("ignore") 
agents=[]
# with open(os.path.join('Results','Results','Toy-NECOM','agent1_600.pkl'),'rb') as f:
#     agent1 = pickle.load(f)
#     agent1.observables=['agent1','agent2','S',"A","B"]
#     agent1.cov_var=torch.full(size=(len(agent1.actions),), fill_value=0.0001)
#     agent1.cov_mat = torch.diag(agent1.cov_var)
#     agents.append(agent1)
# with open(os.path.join('Results','Results','Toy-NECOM','agent2_600.pkl'),'rb') as f:
#     agent2 = pickle.load(f)
#     agent2.observables=['agent1','agent2','S',"A","B"]
#     agent2.cov_var=torch.full(size=(len(agent2.actions),), fill_value=0.0001)
#     agent2.cov_mat = torch.diag(agent2.cov_var)
#     agents.append(agent2)
# env=tk.Environment(name="Toy-NECOM_Facultative",
#                     agents=agents,
#                     dilution_rate=0.001,
#                     extracellular_reactions=[],
#                     initial_condition={"S":100,"agent1":0.1,"agent2":0.1},
#                     inlet_conditions={"S":0},
#                     max_c={'S':100,
#                            'agent1':10,  
#                            'agent2':10,
#                            'A':10,
#                            'B':10,},
#                             dt=0.1,
#                             episode_time=100,
#                             number_of_batches=5000,
#                             episodes_per_batch=10,training=False)

with open(os.path.join('Results','Results','Toy-Exoenzyme','agent1_4800.pkl'),'rb') as f:
    agent1 = pickle.load(f)
    agent1.observables=['agent1','Glc','Starch']
    agent1.cov_var=torch.full(size=(len(agent1.actions),), fill_value=0.0001)
    agent1.cov_mat = torch.diag(agent1.cov_var)
    agents.append(agent1)
env=tk.Environment(name="Toy-Exoenzyme",
                    agents=agents,
                    dilution_rate=0.0001,
                    initial_condition={"Glc":100,"agent1":0.1,"Starch":10},
                    inlet_conditions={"Starch":10},
                    extracellular_reactions=[{"reaction":{
                    "Glc":10,
                    "Starch":-0.1,},
                    "kinetics": (tk.general_kinetic,("Glc","Amylase"))}],
                    max_c={'Glc':100,
                           'agent1':10,  
                           'Starch':10,
                           },
                           dt=0.1,
                           episode_time=100,
                           number_of_batches=5000,
                           episodes_per_batch=10,
                           )
env.constants={}
env.reset()
concs=[]
rewards=[]
for i in range(1000):
    env.t=1000-i
    if i==900:
        print('pass')
        pass
    for agent in env.agents:
        agent.a,_=agent.get_actions(np.hstack([env.state[agent.observables],env.t]))
    s,r,a,_=env.step()
    concs.append(s)
    rewards.append(r)
concs=pd.DataFrame(concs,columns=env.species)

concs.to_csv(os.path.join('Results','Results','Toy-Exoenzyme','concs.csv'))
    