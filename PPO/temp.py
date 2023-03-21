from mimetypes import init
from turtle import color
import Toolkit as tk
import Toy_Exoenzyme_mass_transfer as tm
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
import rich
import multiprocessing as mp
import cobra

NUM_CORES = 8
warnings.filterwarnings("ignore") 

toy_model_1=tm.ToyModel_SA_1.copy()
toy_model_2=tm.ToyModel_SA_2.copy()
toy_model_3=tm.ToyModel_SA_3.copy()
toy_model_4=tm.ToyModel_SA_4.copy()
toy_model_5=tm.ToyModel_SA_5.copy()


agent1=tk.Agent("agent1",
                model=toy_model_1,
                actor_network=tk.NN,
                critic_network=tk.NN,
                clip=0.1,
                lr_actor=0.0001,
                lr_critic=0.001,
                grad_updates=4,
                optimizer_actor=torch.optim.Adam,
                optimizer_critic=torch.optim.Adam,
                observables=['agent1','agent2','agent3','agent4','agent5' ,'Glc_1', 'Starch'],
                actions=["Amylase_1_e"],
                gamma=1,
                tau=0.1
                )

agent2=tk.Agent("agent2",
                model=toy_model_2,
                actor_network=tk.NN,
                critic_network=tk.NN,
                clip=0.1,
                lr_actor=0.0001,
                lr_critic=0.001,
                grad_updates=4,
                optimizer_actor=torch.optim.Adam,
                optimizer_critic=torch.optim.Adam,
                observables=['agent1','agent2','agent3','agent4','agent5' ,'Glc_2', 'Starch'],
                actions=["Amylase_2_e"],
                gamma=1,
                tau=0.1
                )



agent3=tk.Agent("agent3",
                model=toy_model_3,
                actor_network=tk.NN,
                critic_network=tk.NN,
                clip=0.1,
                lr_actor=0.0001,
                lr_critic=0.001,
                grad_updates=4,
                optimizer_actor=torch.optim.Adam,
                optimizer_critic=torch.optim.Adam,
				observables=['agent1','agent2','agent3','agent4','agent5' ,'Glc_3', 'Starch'],
                actions=["Amylase_3_e"],
                gamma=1,
                tau=0.1
)

agent4=tk.Agent("agent4",
                model=toy_model_4,
                actor_network=tk.NN,
                critic_network=tk.NN,
                clip=0.1,
                lr_actor=0.0001,
                lr_critic=0.001,
                grad_updates=4,
                optimizer_actor=torch.optim.Adam,
                optimizer_critic=torch.optim.Adam,
				observables=['agent1','agent2','agent3','agent4','agent5' ,'Glc_4', 'Starch'],
                actions=["Amylase_4_e"],
                gamma=1,
                tau=0.1
)

agent5=tk.Agent("agent5",
                model=toy_model_5,
                actor_network=tk.NN,
                critic_network=tk.NN,
                clip=0.1,
                lr_actor=0.0001,
                lr_critic=0.001,
                grad_updates=4,
                optimizer_actor=torch.optim.Adam,
                optimizer_critic=torch.optim.Adam,
				observables=['agent1','agent2','agent3','agent4','agent5' ,'Glc_5', 'Starch'],
                actions=["Amylase_5_e"],
                gamma=1,
                tau=0.1
)

agents=[agent1,agent2,agent3,agent4,agent5]

env=tk.Environment(name="Toy-Exoenzyme-Five-agents-mass-transfer-low",
                    agents=agents,
                    dilution_rate=0.0001,
                    initial_condition={"Glc":0,"Glc_1":20,"Glc_2":20,"Glc_3":20,"Glc_4":20,"Glc_5":20,"agent1":0.1,"agent2":0.1,'agent3':0.1,'agent4':0.1,'agent5':0.1,"Starch":10},
                    inlet_conditions={"Starch":10},
                    extracellular_reactions=[{"reaction":{
                    "Glc_1":10,
                    "Starch":-0.1,},
                    "kinetics": (tk.general_kinetic,("Glc_1","Amylase_1"))}
		            ,
		            {
                    "reaction":{
                    "Glc_2":10,
                    "Starch":-0.1,},
                    "kinetics": (tk.general_kinetic,("Glc_2","Amylase_2"))
                    ,}
		    ,
		    {
                    "reaction":{
                    "Glc_3":10,
                    "Starch":-0.1,},
                    "kinetics": (tk.general_kinetic,("Glc_3","Amylase_3"))
		    
            },
            {
                    "reaction":{
                    "Glc_4":10,
                    "Starch":-0.1,},
                    "kinetics": (tk.general_kinetic,("Glc_4","Amylase_4"))
                    
            },  
	    
            {   "reaction":{
                    "Glc_5":10,
                    "Starch":-0.1,},
                    "kinetics": (tk.general_kinetic,("Glc_5","Amylase_5"))
                    
            }  ,
            {   
                "reaction":{
                    "Glc_1":-1,
                    "Glc":1},
                "kinetics": (tk.mass_transfer,("Glc_1","Glc"))
            }
            ,
            {
                "reaction":{
                    "Glc_2":-1,
                    "Glc":1},
                "kinetics": (tk.mass_transfer,("Glc_2","Glc"))
            }
            ,
            {
                "reaction":{
                    "Glc_3":-1,
                    "Glc":1},
                "kinetics": (tk.mass_transfer,("Glc_3","Glc"))  
            },
            {
                "reaction":{
                    "Glc_4":-1,
                    "Glc":1},
                "kinetics": (tk.mass_transfer,("Glc_4","Glc"))
            }
            ,
            {
                "reaction":{ 
                    "Glc_5":-1,
                    "Glc":1},
                "kinetics": (tk.mass_transfer,("Glc_5","Glc"))
            }
					],
                    max_c={'Glc':100,
                           'agent1':10,  
                           'Starch':10,
                           },
                           dt=0.1,
                           episode_time=100,
                           number_of_batches=5000,
                           episodes_per_batch=int(NUM_CORES/2),
                           )

env.rewards={agent.name:[] for agent in env.agents}

if not os.path.exists(f"Results/{env.name}"):
	os.makedirs(f"Results/{env.name}")

for agent in env.agents:
	agent.model.solver="glpk"

for batch in range(env.number_of_batches):

	batch_obs,batch_acts, batch_log_probs, batch_rtgs=tk.rollout(env)
	for agent in env.agents:
		V, _= agent.evaluate(batch_obs[agent.name],batch_acts[agent.name])
		A_k = batch_rtgs[agent.name] - V.detach()   
		A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-5) 
		if batch==0:
			rich.print("[bold yellow] Hold on, bringing the creitc network to range...[/bold yellow]")
			err=51
			while err>50:
				V, _= agent.evaluate(batch_obs[agent.name],batch_acts[agent.name])
				critic_loss = nn.MSELoss()(V, batch_rtgs[agent.name])
				agent.optimizer_value_.zero_grad()
				critic_loss.backward()
				agent.optimizer_value_.step() 
				err=critic_loss.item()
			rich.print("[bold green] Done![/bold green]")
		else: 
			
			for _ in range(agent.grad_updates):                                                      
				V, curr_log_probs = agent.evaluate(batch_obs[agent.name],batch_acts[agent.name])
				ratios = torch.exp(curr_log_probs - batch_log_probs[agent.name])
				surr1 = ratios * A_k.detach()
				surr2 = torch.clamp(ratios, 1 - agent.clip, 1 + agent.clip) * A_k
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs[agent.name])
				agent.optimizer_policy_.zero_grad()
				actor_loss.backward(retain_graph=False)
				agent.optimizer_policy_.step()
				agent.optimizer_value_.zero_grad()
				critic_loss.backward()
				agent.optimizer_value_.step()                                                            
	
	if batch%500==0:
	
		with open(f"Results/{env.name}/{env.name}_{batch}.pkl", 'wb') as f:
			pickle.dump(env, f)
		with open(f"Results/{env.name}/observations_{batch}.pkl", 'wb') as f:
			pickle.dump(batch_obs,f)		
		with open(f"Results/{env.name}/actions_{batch}.pkl", 'wb') as f:
			pickle.dump(batch_acts,f)		
      		
	print(f"Batch {batch} finished:")
	for agent in env.agents:
		print(f"{agent.name} return is:  {np.mean(env.rewards[agent.name][-env.episodes_per_batch:])}")	