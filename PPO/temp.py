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
from itertools import combinations

INIT_VAR=0.01

NUM_CORES = 8
warnings.filterwarnings("ignore") 

model_base = cobra.io.read_sbml_model("iJO1366.xml")
medium = model_base.medium.copy()
test_model = model_base.copy()

knockouts_gene_names = [
    "tyrA",
    "pheA",
]

exchange_reactions = {
    "tyrA": "EX_tyr__L_e",
    "pheA": "EX_phe__L_e",
    # "argA": "EX_arg__L_e",
    # "hisB": "EX_his__L_e",
    # "leuB": "EX_leu__L_e",
    # "cysE": "EX_cys__L_e",
    # "glyA": "EX_gly_e",
    # "serA": "EX_ser__L_e",
    # "thrC": "EX_thr__L_e",
    # "trpC": "EX_trp__L_e",
    # "ilvA": "EX_ile__L_e",
    # "lysA": "EX_lys__L_e",
    # "metA": "EX_met__L_e",
    # "proA": "EX_pro__L_e",
}

exchange_species = {}
exchange_mets = {}
for i in exchange_reactions.items():
    exchange_mets[i[0]] = list(test_model.reactions.get_by_id(i[1]).metabolites.keys())[
        0
    ].id

gene_ids = {}
for ko_gene in knockouts_gene_names:
    for gene in test_model.genes:
        if gene.name == ko_gene:
            gene_ids[ko_gene] = gene.id
            break


knockouts = set()
for i in combinations(knockouts_gene_names, 2):
    if set(i) not in knockouts:
        knockouts.add(frozenset(i))

unique_knockouts = [tuple(i) for i in knockouts]

ic={
    key.lstrip("EX_"):10000 for key,val in model_base.medium.items() 
}

ic['glc__D_e']=500
ic['agent1']=0.01
ic['agent2']=0.01
for ko in unique_knockouts:
    model1 = model_base.copy()
    model2 = model_base.copy()
    model1.remove_reactions(model1.genes.get_by_id(gene_ids[ko[0]]).reactions)
    model2.remove_reactions(model2.genes.get_by_id(gene_ids[ko[1]]).reactions)
    model1.exchange_reactions = tuple([model1.reactions.index(i) for i in model1.exchanges])
    model2.exchange_reactions = tuple([model2.reactions.index(i) for i in model2.exchanges])
    model1.biomass_ind=model1.reactions.index("BIOMASS_Ec_iJO1366_core_53p95M")
    model2.biomass_ind=model2.reactions.index("BIOMASS_Ec_iJO1366_core_53p95M")
    model1.solver = "gurobi"
    model2.solver = "gurobi"
    if model1.optimize().objective_value != 0 or model2.optimize().objective_value != 0:
        rich.print(f"[yellow]Skipping {ko} because at least one organism can grow without auxotrophy")
        continue
    else:
        rich.print(f"[green]Non of the KOs can grow without auxotrophy: Running {ko}")
    ko_name = ko[0] + "_" + ko[1]
    agent1 = tk.Agent(
        "agent1",
        model=model1,
        actor_network=tk.NN,
        critic_network=tk.NN,
        clip=0.1,
        lr_actor=0.0001,
        lr_critic=0.001,
        actor_var=0.5,
        grad_updates=4,
        optimizer_actor=torch.optim.Adam,
        optimizer_critic=torch.optim.Adam,
        observables=[
            "agent1",
            "agent2",
            "glc__D_e",
            *[i.replace("EX_", "") for i in exchange_reactions.values()]
        ],
        actions=[i for i in exchange_reactions.values()],
        gamma=1,
    )
    agent2 = tk.Agent(
        "agent2",
        model=model2,
        actor_network=tk.NN,
        critic_network=tk.NN,
        clip=0.1,
        lr_actor=0.0001,
        lr_critic=0.001,
        grad_updates=4,
        actor_var=0.5,
        optimizer_actor=torch.optim.Adam,
        optimizer_critic=torch.optim.Adam,
        observables=[
            "agent1",
            "agent2",
            "glc__D_e",
            *[i.replace("EX_", "") for i in exchange_reactions.values()]
        ],
        actions=[i for i in exchange_reactions.values()],
        gamma=1,
    )

    env = tk.Environment(
        ko_name,
        agents=[agent1, agent2],
        dilution_rate=0,
        extracellular_reactions=[],
        initial_condition=ic,
        inlet_conditions={},
        max_c={},
        dt=0.2,
        episode_time=20,  ##TOBECHANGED
        number_of_batches=5000,  ##TOBECHANGED
        episodes_per_batch=4,
    )

    env.rewards = {agent.name: [] for agent in env.agents}
    # for agent in env.agents:
    #     agent.cov_var = torch.full(size=(len(agent.actions),), fill_value=INIT_VAR)
    #     agent.cov_mat = torch.diag(agent.cov_var)
    #     ### bring the actor network to range
    #     random_states=torch.rand(10000, len(agent.observables)+1)*100
    #     labels=torch.zeros(10000, len(agent.actions))
    #     loss=1
    #     while loss>0.0001:
    #         agent.optimizer_policy_.zero_grad()
    #         outs=agent.actor_network_(random_states)
    #         loss=torch.nn.functional.mse_loss(outs,labels)
    #         loss.backward()
    #         agent.optimizer_policy_.step()
    #         print(loss.item())



    if not os.path.exists(f"Results/{env.name}"):
        os.makedirs(f"Results/{env.name}")
### The next block will train the actor network to output -1 for all actions, so that 
### the agents start like FBA

    for batch in range(env.number_of_batches):
        batch_obs,batch_acts, batch_log_probs, batch_rtgs=tk.rollout(env)
        for agent in env.agents:
            V, _= agent.evaluate(batch_obs[agent.name],batch_acts[agent.name])
            A_k = batch_rtgs[agent.name] - V.detach()   
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-5) 
            if batch==0:
                rich.print("[bold yellow] Hold on, bringing the networks to range...[/bold yellow]")
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
		# Gradualy reduce the variance of the action distribution
        if batch%100==0:
            for agent in env.agents:
                agent.actor_var=agent.actor_var*0.9	
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