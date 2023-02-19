import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
import cobra
import pandas as pd
from torch.distributions import MultivariateNormal,Normal
import ray
import time
from warnings import warn

DEVICE=torch.device('cuda')

class NN(nn.Module):
    """
    This is a base class for all networks created in this algorithm
    """
    def __init__(self,input_dim,output_dim,hidden_dim=20,activation=nn.Tanh ):
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



class Model:
    """This class is a substitute for cobra model. It is used to completely remove LP solver from the solution.
    This is well suited for RL as it makes the algorithm rely on matrix operations only and not rely on external solvers.
    """
    def __init__(self, lb:torch.FloatTensor, ub:torch.FloatTensor,nullspace:torch.FloatTensor,primer:torch.FloatTensor=None):
        self.lb = lb.to(DEVICE)
        self.ub = ub.to(DEVICE)
        self.nullspace=nullspace.to(DEVICE)
        if primer is None:
            self.control=torch.rand(self.nullspace.shape[0]).to(DEVICE)
        else:
            self.control=torch.FloatTensor([torch.dot(primer,self.nullspace[i,:])/torch.dot(self.nullspace[i,:],self.nullspace[i,:]) for i in range(self.nullspace.shape[0])]).to(DEVICE)


def calculate_flux(model:Model):
    sol_raw=torch.matmul(model.control,model.nullspace)
    sols=torch.clip(sol_raw,model.lb,model.ub)
    res=torch.sum(torch.abs(sols-sol_raw))
    return sols,res

def parse_cobra_model(model:cobra.Model,primer:pd.DataFrame=None):
    """This function takes a cobra model and returns a Model object.
    """
    lb = torch.FloatTensor([r.lower_bound for r in model.reactions])
    ub = torch.FloatTensor([r.upper_bound for r in model.reactions])
    nullspace = torch.FloatTensor(linalg.null_space(cobra.util.array.create_stoichiometric_matrix(model))).t()
    if primer is not None:
        primer=torch.FloatTensor(primer.fluxes.to_numpy()).to(DEVICE)
    return Model(lb,ub,nullspace,primer)


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
                number_of_batches:int=100,
                dt:float=0.1,
                episode_time:float=1000,
                dilution_rate:float=0.05,
                episodes_per_batch:int=10,
                training:bool=True,
                
                ) -> None:
        self.name=name
        self.agents = agents
        self.num_agents = len(agents)
        self.extracellular_reactions = extracellular_reactions
        self.dt = dt
        self.episode_length = int(episode_time/dt)
        self.episodes_per_batch=episodes_per_batch
        self.number_of_batches=number_of_batches
        self.batch_per_episode = batch_per_episode
        self.dilution_rate = dilution_rate
        self.training=training
        self.mapping_matrix=self.resolve_exchanges()
        self.species=self.extract_species()
        self.resolve_extracellular_reactions(extracellular_reactions)
        self.initial_condition =np.zeros((len(self.species),))
        for key,value in initial_condition.items():
            self.initial_condition[self.species.index(key)]=value
        self.inlet_conditions = np.zeros((len(self.species),))
        for key,value in inlet_conditions.items():
            self.inlet_conditions[self.species.index(key)]=value
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
        dCdt=np.zeros((len(self.species),))

        for i,M in enumerate(self.agents):
            for index,item in enumerate(self.mapping_matrix["Ex_sp"]):
                if self.mapping_matrix['Mapping_Matrix'][index,i]!=-1:
                    M._model.lb[self.mapping_matrix['Mapping_Matrix'][index,i]]=-M.general_uptake_kinetics(self.state[index+len(self.agents)])
            M._model.control=torch.tensor(M.a).to(DEVICE)
            M.fluxes,M.res=calculate_flux(M._model)
            M.reward=torch.matmul(M.reward_vect,M.fluxes)-M.res
            

        for i in range(self.mapping_matrix["Mapping_Matrix"].shape[0]):
        
            for j in range(len(self.agents)):
                if self.mapping_matrix["Mapping_Matrix"][i, j] != -1:
                    if self.agents[j].res > 1:
                        dCdt[i+len(self.agents)] += 0
                    else:
                  
                        dCdt[i+len(self.agents)] += self.agents[j].fluxes[self.mapping_matrix["Mapping_Matrix"]
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
        return C,list(i.reward.cpu() for i in self.agents),list(i.a for i in self.agents),Cp


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
        if self.training==True:
            for agent in self.agents:
                agent.actor_network_=agent.actor_network(len(agent.observables)+1,agent._model.control.shape[0])
                agent.critic_network_=agent.critic_network(len(agent.observables)+1,1)
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
                prime_solution:pd.DataFrame,
                optimizer_critic:torch.optim.Adam,
                optimizer_actor:torch.optim.Adam,
                reward_vect:np.ndarray,
                observables:list[str],
                gamma:float,
                clip:float=0.01,
                grad_updates:int=1,
                actor_var:float=0.1,
                epsilon:float=0.01,
                lr_actor:float=0.001,
                lr_critic:float=0.001,
                buffer_sample_size:int=500,
                tau:float=0.001,
                alpha:float=0.001) -> None:

        self.name = name
        self.model = model
        self._model=parse_cobra_model(model,prime_solution)
        self.optimizer_critic = optimizer_critic
        self.optimizer_actor = optimizer_actor
        self.gamma = gamma
        self.observables = observables
        self.observables = observables
        self.epsilon = epsilon
        self.general_uptake_kinetics=general_uptake
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
        self.actor_var=actor_var
        self.cov_var = torch.full(size=(self._model.control.shape), fill_value=0.0001)
        self.cov_mat = torch.diag(self.cov_var)
        self.reward_vect = reward_vect.to(DEVICE)
   
    def get_actions(self,observation:np.ndarray):
        """ 
        Gets the actions and their probabilities for the agent.
        """
        mean = self.actor_network_(torch.tensor(observation, dtype=torch.float32)).detach()
        dist = Normal(mean, self.actor_var)
        # dist=torch.distributions.Uniform(low=mean-0.001, high=mean+0.001)
        action = dist.sample()
        log_prob =torch.sum(dist.log_prob(action))
        return action.detach().numpy(), log_prob.detach().numpy()
   
    def evaluate(self, batch_obs,batch_acts):
        V = self.critic_network_(batch_obs).squeeze()
        mean = self.actor_network_(batch_obs)
        dist = Normal(mean, self.actor_var)
        # dist=torch.distributions.Uniform(low=mean-0.001, high=mean+0.001)
        log_probs = torch.sum(dist.log_prob(batch_acts),dim=1)
        return V, log_probs 
    
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

def rollout(env):
    batch_obs={key.name:[] for key in env.agents}
    batch_acts={key.name:[] for key in env.agents}
    batch_log_probs={key.name:[] for key in env.agents}
    batch_rews = {key.name:[] for key in env.agents}
    batch_rtgs = {key.name:[] for key in env.agents}
    batch=[]
    for ep in range(env.episodes_per_batch):
        batch.append(run_episode_single(env))
        # batch.append(run_episode.remote(env))
    # batch = ray.get(batch)
    for ep in range(env.episodes_per_batch):
        for ag in env.agents:
            batch_obs[ag.name].extend(batch[ep][0][ag.name])
            batch_acts[ag.name].extend(batch[ep][1][ag.name])
            batch_log_probs[ag.name].extend(batch[ep][2][ag.name])
            batch_rews[ag.name].append(batch[ep][3][ag.name])
    batch

    for ag in env.agents:
        env.rewards[ag.name].extend(list(np.sum(np.array(batch_rews[ag.name]),axis=1)))

    
    for agent in env.agents:

        batch_obs[agent.name] = torch.tensor(batch_obs[agent.name], dtype=torch.float)
        batch_acts[agent.name] = torch.tensor(batch_acts[agent.name], dtype=torch.float)
        batch_log_probs[agent.name] = torch.tensor(np.array(batch_log_probs[agent.name]), dtype=torch.float)
        batch_rtgs[agent.name] = agent.compute_rtgs(batch_rews[agent.name]) 
    return batch_obs,batch_acts, batch_log_probs, batch_rtgs

@ray.remote
def run_episode(env):
    """ Runs a single episode of the environment. """
    batch_obs = {key.name:[] for key in env.agents}
    batch_acts = {key.name:[] for key in env.agents}
    batch_log_probs = {key.name:[] for key in env.agents}
    episode_rews = {key.name:[] for key in env.agents}
    env.reset()
    episode_len=env.episode_length
    for ep in range(episode_len):
        env.t=episode_len-ep
        obs = env.state.copy()
        for agent in env.agents:   
            action, log_prob = agent.get_actions(np.hstack([obs[agent.observables],env.t]))
            agent.a=action
            agent.log_prob=log_prob
        s,r,a,sp=env.step()
        for ind,ag in enumerate(env.agents):
            batch_obs[ag.name].append(np.hstack([s[ag.observables],env.t]))
            batch_acts[ag.name].append(a[ind])
            batch_log_probs[ag.name].append(ag.log_prob)
            episode_rews[ag.name].append(r[ind])
    return batch_obs,batch_acts, batch_log_probs, episode_rews



def run_episode_single(env):
    """ Runs a single episode of the environment. """
    batch_obs = {key.name:[] for key in env.agents}
    batch_acts = {key.name:[] for key in env.agents}
    batch_log_probs = {key.name:[] for key in env.agents}
    episode_rews = {key.name:[] for key in env.agents}
    env.reset()
    episode_len=env.episode_length
    for ep in range(episode_len):
        env.t=episode_len-ep
        obs = env.state.copy()
        for agent in env.agents:   
            action, log_prob = agent.get_actions(np.hstack([obs[agent.observables],env.t]))
            agent.a=action
            agent.log_prob=log_prob
        s,r,a,sp=env.step()
        for ind,ag in enumerate(env.agents):
            batch_obs[ag.name].append(np.hstack([s[ag.observables],env.t]))
            batch_acts[ag.name].append(a[ind])
            batch_log_probs[ag.name].append(ag.log_prob)
            episode_rews[ag.name].append(r[ind])
    return batch_obs,batch_acts, batch_log_probs, episode_rews




def general_kinetic(x,y):
    return 0.1*x*y/(10+x)
def general_uptake(c):
    return 20*(c/(c+20))






if __name__ == "__main__":
    cmodel=cobra.io.read_sbml_model("iAF1260.xml")
    model=parse_cobra_model(cmodel)
