from distutils.log import warn
import cobra
import torch
import numpy as np
import random
import torch
import torch.nn as nn
from collections import deque,namedtuple
from torch.distributions import MultivariateNormal

import ray
import pandas as pd

class NN(nn.Module):
    """
    This is a base class for all networks created in this algorithm
    """
    def __init__(self,input_dim,output_dim,hidden_dim=64,n_hidden=1,activation=nn.ReLU):
        super(NN,self).__init__()
        self.inlayer=nn.Sequential(nn.Linear(input_dim,hidden_dim),activation())
        self.hidden=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),activation(),
                                  nn.Linear(hidden_dim,hidden_dim),activation(),
                                  nn.Linear(hidden_dim,hidden_dim),activation(),
                                  nn.Linear(hidden_dim,hidden_dim),activation(),
                                  nn.Linear(hidden_dim,hidden_dim),activation(),
                                  nn.Linear(hidden_dim,hidden_dim),activation(),
                                  nn.Linear(hidden_dim,hidden_dim),activation(),
                                  nn.Linear(hidden_dim,hidden_dim),activation(),
                                  nn.Linear(hidden_dim,hidden_dim),activation(),)
        self.output=nn.Linear(hidden_dim,output_dim)
    
    def forward(self, obs):
        out=self.inlayer(obs)
        out=self.hidden(out)
        out=self.output(out)
        return out

def rollout(self):
  # Batch data
    batch_obs = {}             # batch observations
    batch_acts = {}            # batch actions
    batch_log_probs = {}      # log probs of each action
    batch_rews = {}           # batch rewards
    batch_rtgs = {}            # batch rewards-to-go
    batch_lens = {}            # episodic lengths in batch
        


class Environment:
    """ An environment is a collection of the following:
        agents: a list of objects of class Agent
        extracellular reactions: a list of dictionaries. This list should look like this:
        {"reaction":{
            "a":1,
            "b":-1,
            "c":1
        },
        "kinetics": (lambda x,y: x*y,("a","b")),))}
     
    """
    def __init__(self,
                name:str,
                agents:list,
                extracellular_reactions:list[dict],
                initial_condition:dict,
                inlet_conditions:dict,
                batch_per_episode:int=1000,
                number_of_episodes:int=100,
                dt:float=0.1,
                dilution_rate:float=0.05,
                min_c:dict={},
                max_c:dict={},
                batch_iter=1
                
                ) -> None:
        self.name=name
        self.agents = agents
        self.num_agents = len(agents)
        self.extracellular_reactions = extracellular_reactions
        self.dt = dt
        self.batch_per_episode = batch_per_episode
        self.number_of_episodes=number_of_episodes
        self.dilution_rate = dilution_rate
        self.mapping_matrix=self.resolve_exchanges()
        self.species=self.extract_species()
        self.resolve_extracellular_reactions(extracellular_reactions)
        self.initial_condition =np.zeros((len(self.species),))
        for key,value in initial_condition.items():
            self.initial_condition[self.species.index(key)]=value
        self.inlet_conditions = np.zeros((len(self.species),))
        for key,value in inlet_conditions.items():
            self.inlet_conditions[self.species.index(key)]=value
        self.min_c = np.zeros((len(self.species),))
        self.max_c = np.ones((len(self.species),))
        self.batch_iter=batch_iter
        for key,value in max_c.items():
            self.max_c[self.species.index(key)]=value
        self.set_observables()
        self.set_networks()
        self.reset()
        print("Environment {} created successfully!.".format(self.name))

    
    def resolve_exchanges(self)->dict:
        """ Determines the exchange reaction mapping for the community."""
        models=[agent.model for agent in self.agents]
        return Build_Mapping_Matrix(models)
    
    def extract_species(self)->list:
        """ Determines the extracellular species in the community before extracellula reactions."""
        species=[ag.name for ag in self.agents]
        species.extend(self.mapping_matrix["Ex_sp"])
        return species

    def resolve_extracellular_reactions(self,extracellular_reactions:list[dict])->list[dict]:
        """ Determines the extracellular reactions for the community."""
        species=[]
        [species.extend(list(item["reaction"].keys())) for item in extracellular_reactions]
        new_species=[item for item in species if item not in self.species]
        if len(new_species)>0:
            warn("The following species are not in the community: {}".format(new_species))
            self.species.extend(new_species)
    
    
    def reset(self):
        """ Resets the environment to its initial state."""
        self.state = self.initial_condition.copy()
    
    def step(self):
        """ Performs a single step in the environment."""
        self.temp_actions=[]
        self.state[self.state<0]=0
        dCdt = np.zeros(self.state.shape)
        Sols = list([0 for i in range(len(self.agents))])
        for i,M in enumerate(self.agents):
            for index,item in enumerate(self.mapping_matrix["Ex_sp"]):
                if self.mapping_matrix['Mapping_Matrix'][index,i]!=-1:
                    M.model.reactions[self.mapping_matrix['Mapping_Matrix'][index,i]].upper_bound=100
                    M.model.reactions[self.mapping_matrix['Mapping_Matrix'][index,i]].lower_bound=-M.general_uptake_kinetics(self.state[index+len(self.agents)])


            for index,flux in enumerate(M.actions):

                if M.a[index]<0:
                
                    M.model.reactions[M.actions[index]].lower_bound=max(M.a[index],M.model.reactions[M.actions[index]].lower_bound)
                    # M.model.reactions[M.actions[index]].lower_bound=M.a[index]*M.model.reactions[M.actions[index]].lower_bound

                else:
                    M.model.reactions[M.actions[index]].lower_bound=min(M.a[index],10)
                
                # M.model.reactions[M.actions[index]].upper_bound=M.model.reactions[M.actions[index]].lower_bound+0.00001



            Sols[i] = self.agents[i].model.optimize()
            # self.agents[i].feasibility_optimizer_.zero_grad()
            if Sols[i].status == 'infeasible':
                self.agents[i].reward=-1
                dCdt[i] = 0
                # pred=self.agents[i].feasibility_network_(torch.cat([torch.FloatTensor(self.state[self.agents[i].observables]),torch.FloatTensor(self.agents[i].a)]))
                # l=cross_entropy_loss(pred,torch.FloatTensor([0,1]))
            
            else:
                dCdt[i] += Sols[i].objective_value*self.state[i]
                self.agents[i].reward =Sols[i].objective_value*self.state[i]
                # pred=self.agents[i].feasibility_network_(torch.cat([torch.FloatTensor(self.state[self.agents[i].observables]),torch.FloatTensor(self.agents[i].a)]))
                # l=cross_entropy_loss(pred,torch.FloatTensor([1,0]))
            # l.backward()
            # self.agents[i].feasibility_optimizer_.step()
        # Handling the exchange reaction balances in the community
        self.temp_actions=[[Sols[j].fluxes.iloc[i] for i in self.agents[i].actions] for j in range(len(self.agents))]
        for i in range(self.mapping_matrix["Mapping_Matrix"].shape[0]):
        
            for j in range(len(self.agents)):

                if self.mapping_matrix["Mapping_Matrix"][i, j] != -1:
                    if Sols[j].status == 'infeasible':
                        dCdt[i+len(self.agents)] += 0
                    else:
                        dCdt[i+len(self.agents)] += Sols[j].fluxes.iloc[self.mapping_matrix["Mapping_Matrix"]
                                                    [i, j]]*self.state[j]
        
        # Handling extracellular reactions

        for ex_reaction in self.extracellular_reactions:
            rate=ex_reaction["kinetics"][0](*[self.state[self.species.index(item)] for item in ex_reaction["kinetics"][1]])
            for metabolite in ex_reaction["reaction"].keys():
                dCdt[self.species.index(metabolite)]+=ex_reaction["reaction"][metabolite]*rate
        dCdt+=self.dilution_rate*(self.inlet_conditions-self.state)
        C=self.state.copy()
        self.state += dCdt*self.dt
        Cp=self.state.copy()
        return C,list(i.reward for i in self.agents),list(i.a for i in self.agents),Cp


    @ray.remote
    def _step_p(self,C):
        """ Performs a single step in the environment."""
        dCdt = np.zeros(C.shape)
        Sols = list([0 for i in range(len(self.agents))])
        for i,M in enumerate(self.agents):
            
            for index,item in enumerate(self.mapping_matrix["Ex_sp"]):
                if self.mapping_matrix['Mapping_Matrix'][index,i]!=-1:
                    M.model.reactions[self.mapping_matrix['Mapping_Matrix'][index,i]].upper_bound=10
                    M.model.reactions[self.mapping_matrix['Mapping_Matrix'][index,i]].lower_bound=-M.general_uptake_kinetics(C[index+len(self.agents)])    
            
            for index,flux in enumerate(M.actions): 
                if M.a[index]<0:
                
                    M.model.reactions[M.actions[index]].lower_bound=max(M.a[index],-10)
                    # M.model.reactions[M.actions[index]].lower_bound=M.a[index]*M.model.reactions[M.actions[index]].lower_bound    
                else:
                    M.model.reactions[M.actions[index]].lower_bound=min(M.a[index],20)

                # M.model.reactions[M.actions[index]].upper_bound=M.model.reactions[M.actions[index]].lower_bound+0.000001    
            Sols[i] = self.agents[i].model.optimize()
            if Sols[i].status == 'infeasible':
                self.agents[i].reward= 0
                dCdt[i] = 0
            else:
                dCdt[i] += Sols[i].objective_value*C[i]
                self.agents[i].reward =Sols[i].objective_value
            

        # Handling the exchange reaction balances in the community  
        for i in range(self.mapping_matrix["Mapping_Matrix"].shape[0]):
        
            for j in range(len(self.agents)):   
                if self.mapping_matrix["Mapping_Matrix"][i, j] != -1:
                    if Sols[j].status == 'infeasible':
                        dCdt[i+len(self.agents)] += 0
                    else:
                        dCdt[i+len(self.agents)] += Sols[j].fluxes.iloc[self.mapping_matrix["Mapping_Matrix"]
                                                    [i, j]]*C[j]

        for ex_reaction in self.extracellular_reactions:
            rate=ex_reaction["kinetics"][0](*[C[self.species.index(item)] for item in ex_reaction["kinetics"][1]])
            for metabolite in ex_reaction["reaction"].keys():
                dCdt[self.species.index(metabolite)]+=ex_reaction["reaction"][metabolite]*rate
        dCdt+=self.dilution_rate*(self.inlet_conditions-C)
        Cp=C + dCdt*self.dt
        Cp[Cp<0]=0
        return C,list(i.reward for i in self.agents),list(i.a for i in self.agents),Cp

    def generate_random_c(self,size:int):
        """ Generates a random initial condition for the environment."""
        return np.random.uniform(low=self.min_c, high=self.max_c, size=(size,len(self.species))).T

    def batch_step(self,C:np.ndarray):
        """ Performs a batch of steps in the environment in parallel.
        This is just an experimental feature and is not yet implemented.
        C is a m*n where m is the number of species in the system and n 
        is the number of parallel steps.
        """
        batch_episodes=[]
        for batch in range(C.shape[1]):
            batch_episodes.append(Environment._step_p.remote(self,C[:,batch]))
        batch_episodes = ray.get(batch_episodes)

        return batch_episodes

    def set_observables(self):
        """ Sets the observables for the agents in the environment."""
        for agent in self.agents:
            agent.observables=[self.species.index(item) for item in agent.observables]

    def set_networks(self):
        """ Sets the networks for the agents in the environment."""
        for agent in self.agents:
            agent.actor_network_=agent.actor_network(len(agent.observables),len(agent.actions))
            agent.critic_network_=agent.critic_network(len(agent.observables),len(agent.actions))
            agent.optimizer_value_ = agent.optimizer_critic(agent.critic_network_.parameters(), lr=agent.lr_critic)
            agent.optimizer_policy_ = agent.optimizer_actor(agent.actor_network_.parameters(), lr=agent.lr_actor)
    
class Agent:
    """ Any microbial agent will be an instance of this class.
    """
    def __init__(self,
                name:str,
                model:cobra.Model,
                actor_network:NN,
                critic_network:NN,
                optimizer_critic:torch.optim.Adam,
                optimizer_actor:torch.optim.Adam,
                actions:list[str],
                observables:list[str],
                gamma:float,
                clip:float=0.01,
                grad_updates:int=1,
                epsilon:float=0.01,
                lr_actor:float=0.001,
                lr_critic:float=0.001,
                buffer_sample_size:int=500,
                tau:float=0.001,
                alpha:float=0.001) -> None:

        self.name = name
        self.model = model
        self.optimizer_critic = optimizer_critic
        self.optimizer_actor = optimizer_actor
        self.gamma = gamma
        self.observables = observables
        self.actions = [self.model.reactions.index(item) for item in actions]
        self.observables = observables
        self.epsilon = epsilon
        self.general_uptake_kinetics=lambda C: 20*(C/(C+20))
        self.tau = tau
        self.clip = clip
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.buffer_sample_size = buffer_sample_size
        self.R=0
        self.grad_updates = grad_updates
        self.alpha = alpha
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.cov_var = torch.full(size=(len(self.actions),), fill_value=0.1)
        self.cov_mat = torch.diag(self.cov_var)
   
    def get_actions(self,observation:np.ndarray):
        """ 
        Gets the actions and their probabilities for the agent.
        """
        mean = self.actor_network_(torch.tensor(observation, dtype=torch.float32)).detach()
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()
   
    def evaluate(self, batch_obs,batch_obs_next,batch_acts):
        V = self.critic_network_(batch_obs).squeeze()
        VP=self.critic_network_(batch_obs_next).squeeze()
        mean = self.actor_network_(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs ,VP
    
    def compute_rtgs(self, batch_rews):

        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs


        
def Build_Mapping_Matrix(Models:list[cobra.Model])->dict:
    """
    Given a list of COBRA model objects, this function will build a mapping matrix for all the exchange reactions.

    """

    Ex_sp = []
    Temp_Map={}
    for model in Models:
        
        
        if not hasattr(model,"Biomass_Ind"):
            raise Exception("Models must have 'Biomass_Ind' attribute in order for the DFBA to work properly!")
        
        
        for Ex_rxn in model.exchanges :
            if Ex_rxn!=model.reactions[model.Biomass_Ind]:
                if list(Ex_rxn.metabolites.keys())[0].id not in Ex_sp:
                    Ex_sp.append(list(Ex_rxn.metabolites.keys())[0].id)
                if list(Ex_rxn.metabolites.keys())[0].id in Temp_Map.keys():
                   Temp_Map[list(Ex_rxn.metabolites.keys())[0].id][model]=Ex_rxn
                else:
                     Temp_Map[list(Ex_rxn.metabolites.keys())[0].id]={model:Ex_rxn}

    Mapping_Matrix = np.zeros((len(Ex_sp), len(Models)), dtype=int)
    for i, id in enumerate(Ex_sp):
        for j, model in enumerate(Models):
            if model in Temp_Map[id].keys():
                Mapping_Matrix[i, j] = model.reactions.index(Temp_Map[id][model].id)
            else:
                Mapping_Matrix[i, j] = -1
    return {"Ex_sp": Ex_sp, "Mapping_Matrix": Mapping_Matrix}
@ray.remote
def simulate(env,episodes=200,steps=1000):
    """ Simulates the environment for a given number of episodes and steps."""
    env.rewards=np.zeros((len(env.agents),episodes))
    env.record=[]
    for episode in range(episodes):
        env.reset()
        env.episode=episode

        for agent in env.agents:
            agent.rewards=[]
        C=[]
        episode_len=steps
        for ep in range(episode_len):
            env.t=episode_len-ep
            s,r,a,sp=env.step()
            for ind,ag in enumerate(env.agents):
                ag.rewards.append(r[ind])
                # ag.optimizer_reward_.zero_grad()
                # # r_pred=ag.reward_network_(torch.FloatTensor(np.hstack([s[ag.observables],env.t])), torch.FloatTensor(a[ind]))
                # # r_loss=nn.MSELoss()(r_pred,torch.FloatTensor(np.expand_dims(np.array(r[ind]),0)))
                # # r_loss.backward()
                # ag.optimizer_reward_.step()
                ag.optimizer_value_.zero_grad()
                Qvals = ag.critic_network_(torch.FloatTensor(np.hstack([s[ag.observables],env.t])), torch.FloatTensor(a[ind]))
                next_actions = ag.actor_network_(torch.FloatTensor(np.hstack([sp[ag.observables],env.t-1])))
                if env.t==1:
                    next_Q = torch.FloatTensor([0])
                else:
                    next_Q = ag.critic_network_.forward(torch.FloatTensor(np.hstack([sp[ag.observables],env.t-1])), next_actions.detach())
                Qprime = torch.FloatTensor(np.expand_dims(np.array(r[ind]),0))+ag.gamma*next_Q
                critic_loss=nn.MSELoss()(Qvals,Qprime.detach())
                critic_loss.backward()
                ag.optimizer_value_.step()
                ag.optimizer_policy_.zero_grad()
                policy_loss = -ag.critic_network_(torch.FloatTensor(np.hstack([s[ag.observables],env.t])), ag.actor_network_(torch.FloatTensor(np.hstack([s[ag.observables],env.t])))).mean()
                policy_loss.backward()
                ag.optimizer_policy_.step()
            C.append(env.state.copy())
            env.record.append(np.hstack([env.state.copy(),np.reshape(np.array(env.temp_actions),(-1))]))

        # pd.DataFrame(C,columns=env.species).to_csv("Data.csv")

        for ag_ind,agent in enumerate(env.agents):
            print(episode)
            print(np.sum(agent.rewards))
            env.rewards[ag_ind,episode]=np.sum(agent.rewards)
    return env.rewards.copy(),env.record.copy()



def rollout(env):
    batch_obs = {key.name:[] for key in env.agents}
    batch_obs_next={key.name:[] for key in env.agents}
    batch_acts = {key.name:[] for key in env.agents}
    batch_log_probs = {key.name:[] for key in env.agents}
    batch_rews = {key.name:[] for key in env.agents}
    for step in range(env.batch_iter):
        obs = env.state.copy()
        for agent in env.agents:   
            action, log_prob = agent.get_actions(obs[agent.observables])
            agent.a=action
            agent.log_prob=log_prob         
        obs, rew,_,obs_p = env.step()
        for m,agent in enumerate(env.agents):
            batch_obs[agent.name].append(obs[agent.observables])
            batch_obs_next[agent.name].append(obs_p[agent.observables])
            batch_acts[agent.name].append(agent.a)
            batch_log_probs[agent.name].append(agent.log_prob)
            batch_rews[agent.name].append(rew[m])
    
    for agent in env.agents:

        batch_obs[agent.name] = torch.tensor(batch_obs[agent.name], dtype=torch.float)
        batch_acts[agent.name] = torch.tensor(batch_acts[agent.name], dtype=torch.float)
        batch_log_probs[agent.name] = torch.tensor(batch_log_probs[agent.name], dtype=torch.float)
        batch_rews[agent.name]= torch.tensor(batch_rews[agent.name], dtype=torch.float)                                                            # ALG STEP 4
        batch_obs_next[agent.name] = torch.tensor(batch_obs_next[agent.name], dtype=torch.float)
    return batch_obs,batch_obs_next,batch_acts, batch_log_probs, batch_rews

