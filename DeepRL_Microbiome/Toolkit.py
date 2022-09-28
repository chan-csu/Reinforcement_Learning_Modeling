from distutils.log import warn
import cobra
import torch
import numpy as np
import random
import torch
import torch.nn as nn
from collections import deque,namedtuple
import ray

class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state):
        experience = (state, action, np.array([reward]), next_state)
        self.buffer.appendleft(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            
        
        return state_batch, action_batch, reward_batch, next_state_batch

    def __len__(self):
        return len(self.buffer)




class DDPGActor(nn.Module):

    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 30),nn.Tanh(),
            nn.Linear(30,30),nn.Tanh(),
            nn.Linear(30,30),nn.Tanh(),
            nn.Linear(30, act_size))

    def forward(self, x):
       return self.net(x)

class DDPGCritic(nn.Module):       
    
    def __init__(self, obs_size, act_size):

        super(DDPGCritic, self).__init__()
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 30),nn.Tanh(),
            nn.Linear(30,30),nn.Tanh(),         
            nn.Linear(30,30),nn.Tanh(),     
            nn.Linear(30,20),
            
            )


        self.out_net = nn.Sequential(
                       nn.Linear(20 + act_size, 30),nn.Tanh(),
                       nn.Linear(30,30),nn.Tanh(), 
                       nn.Linear(30, 1),
                       )
    
    def forward(self, x, a):
        obs = self.obs_net(x)           
        return self.out_net(torch.cat([obs, a],dim=1))




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
                dt:float=0.1,
                dilution_rate:float=0.1,
                min_c:dict={},
                max_c:dict={},
                
                ) -> None:
        self.name=name
        self.agents = agents
        self.num_agents = len(agents)
        self.extracellular_reactions = extracellular_reactions
        self.dt = dt
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
        species=[item["reaction"].keys() for item in extracellular_reactions]
        new_species=[item for item in species if item not in self.species]
        if len(new_species)>0:
            warn("The following species are not in the community: {}".format(new_species))
            print("Adding the following species to the community: {}".format(new_species))
            self.species.extend(new_species)
    
    
    def reset(self):
        """ Resets the environment to its initial state."""
        self.state = self.initial_condition.copy()
    
    def step(self):
        """ Performs a single step in the environment."""
        
        dCdt = np.zeros(self.state.shape)
        Sols = list([0 for i in range(len(self.agents))])
        for i,M in enumerate(self.agents):
            M.a=M._actor_network(torch.FloatTensor([self.state[M.observables]])).detach().numpy()[0]
            if random.random()<M.epsilon:
                M.a+=np.random.uniform(low=-1, high=1,size=len(M.actions))

            else:

                pass
            
            for index,item in enumerate(self.mapping_matrix["Ex_sp"]):
                if self.mapping_matrix['Mapping_Matrix'][index,i]!=-1:
                    M.model.reactions[self.mapping_matrix['Mapping_Matrix'][index,i]].upper_bound=100
                    M.model.reactions[self.mapping_matrix['Mapping_Matrix'][index,i]].lower_bound=-M.general_uptake_kinetics(self.state[index+len(self.agents)])


            for index,flux in enumerate(M.actions):

                if M.a[index]<0:
                
                    M.model.reactions[M.actions[index]].lower_bound=max(M.a[index],M.model.reactions[M.actions[index]].lower_bound)
                    # M.model.reactions[M.actions[index]].lower_bound=M.a[index]*M.model.reactions[M.actions[index]].lower_bound

                else:
                    M.model.reactions[M.actions[index]].lower_bound=min(M.a[index],M.model.reactions[M.actions[index]].upper_bound)

                M.model.reactions[M.actions[index]].upper_bound=M.model.reactions[M.actions[index]].lower_bound+0.000001


            Sols[i] = self.agents[i].model.optimize()
            if Sols[i].status == 'infeasible':
                self.agents[i].reward= 0
                dCdt[i] = 0
            else:
                dCdt[i] += Sols[i].objective_value*self.state[i]
                self.agents[i].reward =Sols[i].objective_value
                
        # Handling the exchange reaction balances in the community

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
        self.state += dCdt*self.dt
        return self.state,(i.reward for i in self.agents),(i.a for i in self.agents)


    @ray.remote
    def _step_p(self,C):
        """ Performs a single step in the environment."""
        dCdt = np.zeros(C.shape)
        Sols = list([0 for i in range(len(self.agents))])
        for i,M in enumerate(self.agents):
            M.a=M._actor_network(torch.FloatTensor([C[M.observables]])).detach().numpy()[0]
            if random.random()<M.epsilon:
                M.a+=np.random.uniform(low=-1, high=1,size=len(M.actions))  
            else:   
                pass
            
            for index,item in enumerate(self.mapping_matrix["Ex_sp"]):
                if self.mapping_matrix['Mapping_Matrix'][index,i]!=-1:
                    M.model.reactions[self.mapping_matrix['Mapping_Matrix'][index,i]].upper_bound=100
                    M.model.reactions[self.mapping_matrix['Mapping_Matrix'][index,i]].lower_bound=-M.general_uptake_kinetics(C[index+len(self.agents)])    
            for index,flux in enumerate(M.actions): 
                if M.a[index]<0:
                
                    M.model.reactions[M.actions[index]].lower_bound=max(M.a[index],M.model.reactions[M.actions[index]].lower_bound)
                    # M.model.reactions[M.actions[index]].lower_bound=M.a[index]*M.model.reactions[M.actions[index]].lower_bound    
                else:
                    M.model.reactions[M.actions[index]].lower_bound=min(M.a[index],M.model.reactions[M.actions[index]].upper_bound) 
                M.model.reactions[M.actions[index]].upper_bound=M.model.reactions[M.actions[index]].lower_bound+0.000001    
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
            agent._actor_network=agent.actor_network(len(agent.observables),len(agent.actions))
            agent._critic_network=agent.critic_network(len(agent.observables),len(agent.actions))
            agent._target_actor_network=agent.actor_network(len(agent.observables),len(agent.actions))
            agent._target_critic_network=agent.critic_network(len(agent.observables),len(agent.actions))


class Agent:
    """ Any microbial agent will be an instance of this class.
    """
    def __init__(self,
                name:str,
                model:cobra.Model,
                actor_network:DDPGActor,
                critic_network:DDPGCritic,
                actions:list[str],
                observables:list[str],
                buffer:Memory,
                gamma:float,
                update_batch_size:int,
                epsilon:float=0.01) -> None:

        self.name = name
        self.buffer = buffer
        self.model = model
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.gamma = gamma
        self.update_batch_size = update_batch_size
        self.observables = observables
        self.actions = [self.model.reactions.index(item) for item in actions]
        self.observables = observables
        self.epsilon = epsilon
        self.general_uptake_kinetics=lambda C: 20*(C/(C+20))


        
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

