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
with open('Results/Toy-Exoenzyme/agent1_2000.pkl','rb') as f:
    agent = pickle.load(f)

agent.observables=['agent1', 'Glc', 'Starch']
agents=[agent]

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

env.reset()
concs=[]
for i in range(1000):
    s,_,a,_=env.step()
    concs.append(s)
concs=pd.DataFrame(concs,columns=env.species)
concs.plot()
plt.show(
)

    