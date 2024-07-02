"""This module contains a variety of kinetic models to be used"""
import numpy as np
import scipy.integrate as integrate
import plotly.express as px
import pandas as pd
import numba
import torch
from torch.distributions import Normal
from typing import Iterable
import ray 
EPS=1e-10
class Network(torch.nn.Module):
    def __init__(self, input_dim:int, output_dim:int, hidden_layers:tuple[int], activation:torch.nn.Module):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(self.input_dim,self.hidden_layers[0]))
        for i in range(1,len(self.hidden_layers)):
            self.layers.append(torch.nn.Linear(self.hidden_layers[i-1],self.hidden_layers[i]))
        self.layers.append(torch.nn.Linear(self.hidden_layers[-1],self.output_dim))
        self.relu=torch.nn.ReLU()
        
    def forward(self,x:torch.FloatTensor)->torch.FloatTensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.relu(self.layers[-1](x))
        

TOY_REACTIONS = [
    "S_import",
    "S_to_I1",
    "I1_to_P",
    "S_to_NTP",
    "NTP_to_NA",
    "I1_to_Li",
    "I1_to_AA",
    "AA_to_e",
    "P_export",
    "AA_and_li_to_W",
    "e_to_t1",
    "e_to_e1",
    "e_to_e2",
    "e_to_e3",
    "e_to_e4",
    "e_to_e5",
    "e_to_e6",
    "e_to_e7",
    "e_to_e8",
    "e_to_t2",
    ]

TOY_SPECIES = [
    "S_env",
    "S",
    "I1",
    "P",
    "P_env",
    "NTP",
    "NA",
    "Li",
    "AA",
    "e",
    "t1",
    "t2",
    "e1",
    "e2",
    "e3",
    "e4",
    "e5",
    "e6",
    "e7",
    "e8",
    "E",
    "W"
]
    
class Shape:
    """Objects of this class represent the shape of a cell"""
    def __init__(self, name:str,dimensions:tuple[dict[str,float]]):
        self.name = name
        self.dimensions = dimensions
        for key, value in dimensions.items():
            setattr(self, key, value)
    
    def __str__(self) -> str:
        dim_print=""
        for key, value in self.__dict__.items():
            dim_print += f"{key} = {value}\n"
        return f"{self.name} shape with dimensions:\n{dim_print}"
    
    def __repr__(self) -> str:
        return f"Shape({self.name},{self.dimensions})"
    
    @property
    def volume(self)->float:
        pass
    
    @property
    def area(self)->float:
        pass
    
    def calculate_differentials(self,dv:float)->dict[str,float]:
        pass
    def set_dim_from_volume(self,v:float)->None:
        pass
    

class Sphere(Shape):
    """Objects of this class represent a sphere. 
    NOTE: This is hollow sphere. So, the constructor takes in a dictionary with the following
    keys: r,t. r is the radius of the sphere and t is the thickness of the sphere.
    The assumption in this class is that the thickness is uniform and throughout the sphere.
    """
    def __init__(self, dimensions:dict[str,float])->None:
        if {"r","t"} != set(dimensions.keys()):
            raise ValueError("Sphere shape requires only parameter r")
        super().__init__("Sphere", dimensions)
    
    @property
    def volume(self)->float:
        return 4/3*np.pi*self.r**3
    
    @property
    def area(self)->float:
        return 4*np.pi*self.r**2
    
    def set_dimensions(self,dimensions:dict[str,float])->None:
        self.dimensions = dimensions
        for key, value in dimensions.items():
            setattr(self, key, value)
    
    def get_dimensions(self)->dict[str,float]:
        return {key:getattr(self,key) for key in self.dimensions.keys()}
        
    
    def set_dim_from_volume(self,v:float)->None:
        self.r = (3*v/(4*np.pi))**(1/3)

    def calculate_differentials(self,dv:float)->dict[str,float]:
        """V=4*pi*[3rt^3+3t^2r+t^3] this equation should be used:
        dV=4*pi*[6rt+3t^2]*dr
        dr=dv/(4*pi*[6rt+3t^2])
        """
        diffs={k:0 for k in self.dimensions.keys()}
        diffs.update({"r":dv/(4*np.pi*(6*self.r*self.t + 3*self.t**2))})
        return diffs
    
class Cell:
    """Objects of this class represent the biological function of a Cell"""
    def __init__(self,
                 name:str,
                 stoichiometry:callable,
                 ode_sys:callable, 
                 parameters:dict,
                 reactions:list,
                 compounds:list,
                 shape:Shape,
                 observable_env_states:list[str],
                 controlled_params:list,
                 initial_conditions:dict[str,float],
                 gamma:float=1,
                 grad_updates:int=10
                 ):
        self.name = name
        self.stoichiometry = stoichiometry
        self.ode_sys = ode_sys
        self.parameters = parameters
        self.shape=shape
        self.reactions = reactions
        self.compounds = compounds
        if set(controlled_params).issubset(set(self.parameters.keys())):
            self.controlled_params=controlled_params
        else:
            raise Exception("The controlled parameters should be a subset of parameters")
        
        self.state_variables = self.get_state_variables()
        self.number_index=self.state_variables.index("number")
        self.volume_index=self.state_variables.index("volume")
        self.gamma=gamma
        self.kinetics={}
        self.observable_states = [ind for ind,i in enumerate(self.state_variables) if not i.endswith("_env") or i in observable_env_states]
        self._set_policy()
        self._set_value()
        for dim,val in self.shape.get_dimensions().items():
            initial_conditions[dim]=val
        initial_conditions["volume"]=self.shape.volume
        initial_conditions["number"]=initial_conditions.setdefault("number",1)
        self._initial_state=tuple([initial_conditions.get(i,0) for i in self.state_variables])
        self.env_metabolites=[ind for ind,i in enumerate(self.state_variables) if i.endswith("_env")]
        self.environment=None
        self.reset()
        self.grad_updates=grad_updates
        
        
        
    
    @property
    def initial_state(self)->np.ndarray:
        return np.array(self._initial_state,dtype=np.float64)
    
    @property
    def number(self)->float:
        return self.state[self.number_index]
    
    @property
    def volume(self)->float:
        return self.state[self.volume_index]
        
    def get_state_variables(self,include_time:bool=True)->list:
        """
        This method returns the state variables of the cell. The state variables are the compounds and the shape variables\
        
        """
        state_variables = self.compounds.copy()
        shape_variables = [key for key in self.shape.dimensions.keys()]
        state_variables.extend(shape_variables)
        state_variables.append("volume")
        state_variables.append("number")
        if include_time:
            state_variables.append("time_env")
        return state_variables
    
    
    def can_double(self,state:np.ndarray)->bool:
        if self.volume>=self.parameters["split_volume"]:
            return True
        else:
            return False
    
    def update_parameters(self,new_params:dict)->None:
        self.parameters.update(new_params)
    
    def decide(self)->tuple[torch.FloatTensor]:
        outs=self.policy(torch.FloatTensor(self.state.take(self.observable_states))).view(len(self.controlled_params),2)
        dist=Normal(outs[:,0],outs[:,1]+EPS)
        self.actions=dist.sample()
        self.log_prob =torch.sum(dist.log_prob(self.actions)).detach()
        return self.actions,self.log_prob
    
    def evaluate(self,
                 batch_states:torch.FloatTensor,
                 batch_actions:torch.FloatTensor)->tuple[torch.FloatTensor]:
        outs=self.policy(batch_states).view(-1,len(self.controlled_params),2)
        dist=Normal(outs[:,:,0],outs[:,:,1]+EPS)
        log_prob = torch.sum(dist.log_prob(batch_actions),dim=1)
        v=self.value(batch_states)
        return v,log_prob

    
    def _decision_to_params(self,decision:torch.FloatTensor)->dict:
        return dict(zip(self.controlled_params,decision))
    
    def _set_policy(self)->None:
        self.policy = Network(len(self.observable_states),len(self.controlled_params)*2,(10,10),torch.nn.ReLU())
    
    def _set_value(self)->None:
        self.value = Network(len(self.observable_states),1,(10,10),torch.nn.ReLU())
    
    def reset(self)->None:
        self.state=self.initial_state
        return
    
    @property
    def state(self)->np.ndarray:
        return self._state
    
    @state.setter
    def state(self,state:np.ndarray)->None:
        self._state=state
    
    @property
    def reward(self)->float:
        return self.number

    def process_data(self,data:dict[str,np.ndarray])->None:
        data["s`"]=torch.FloatTensor(data["s`"])
        data["r"]=torch.FloatTensor(data["r"])
        data["a"]=torch.FloatTensor(data["a"])
        data["s"]=torch.FloatTensor(data["s"])
        
    
    
class Environment:
    def __init__(self, 
                 name:str, 
                 cells:Iterable[Cell],
                 extra_states:Iterable[str],
                 initial_conditions:dict[str,float],
                 controllers:dict[str,callable],
                 time_step:float=0.1
                ):
        self.name = name
        self.cells = cells
        for cell in self.cells:
            cell.environment=self
        self.environment_vars=self.resolve_env_vars(extra_state=extra_states)
        self.initial_conditions = initial_conditions
        for c_states,func in controllers.items():
            self.initial_conditions[c_states]=func(self.get_state_from_initial_conditions())
        for compound in set(self.environment_vars).difference(set(initial_conditions.keys())):
            self.initial_conditions[compound]=0
        self.state=self.get_state_from_initial_conditions()
        self.controllers=controllers
        self.time_step=time_step
    
    def resolve_env_vars(self,extra_state:Iterable[str])->list:
        env_vars=[]
        env_vars_mapping={}
        for cell in self.cells:
            env_vars_mapping[cell.name]={}
            env_comps=[i for i in enumerate(cell.state_variables) if i[1].endswith("_env")]
            for ind,comp in env_comps:
                env_vars_mapping[cell.name][comp]=ind
            env_vars.extend([i[1] for i in env_comps])
        env_vars=[i.name for i in self.cells]+sorted(list(set(env_vars)-set("time_env")))+extra_state+["time_env"] # Makes sure that time_env is the last variable and only appears once
        self.env_vars_mapping=env_vars_mapping
        return env_vars
    
    def get_state_from_initial_conditions(self)->dict[str,float]:
        return {key:self.initial_conditions.get(key,0) for key in self.environment_vars}
        
    def pass_env_states(self)->None:
        """A critical method to communicate environment states to the cells. This method updates the states of the cells with the environment states."""
        for cell in self.cells:
            for key,value in self.env_vars_mapping[cell.name].items():
                cell.state[value]=self.state[key]
        return
    def reset(self)->None:
        self.state=self.get_state_from_initial_conditions()
        for c_states,func in self.controllers.items():
            self.initial_conditions[c_states]=func(self.state)
        for cell in self.cells:
            cell.reset()
        self.pass_env_states()
        return
    
    def step(self)->dict[str,np.ndarray]:
        ### to update the information of the agents about the environment
        self.pass_env_states()
        previous_states={cell.name:cell.state.take(cell.observable_states) for cell in self.cells}
        rewards={}
        actions={}
        ddt_collections={}
        for cell in self.cells:
            dydt=cell.ode_sys(self.state['time_env'],cell.state,cell)
            cell.state+=dydt*self.time_step
            rewards[cell.name]=cell.reward
            actions[cell.name]=cell.actions
            for comp,ind in self.env_vars_mapping[cell.name].items():
                ddt_collections[comp]=ddt_collections.get(comp,0)+dydt[ind]
        
        for key in ddt_collections:
            self.state[key]+=ddt_collections[key]
        
        self.state['time_env']+=self.time_step
        
        for c_states,func in self.controllers.items():
            self.state[c_states]=func(self.state)   
        
        self.pass_env_states()

        return ({cell.name:cell.state.take(cell.observable_states) for cell in self.cells},rewards,actions,previous_states,{cell.name:cell.log_prob for cell in self.cells})
                
class Trainer:
    """This is a class to train the agents in a given environment"""
    def __init__(self,
                 env:Environment,
                 episodes_per_batch:int,
                 steps_per_episode:int,
                 number_of_batches:int,
                 save_every:int,
                 save_path:str,
                 parallel_framework:str="ray",
                 ):
    
        self.env=env
        self.episodes_per_batch=episodes_per_batch
        self.steps_per_episode=steps_per_episode
        self.number_of_batches=number_of_batches
        self.save_every=save_every
        self.save_path=save_path
        self.parallel_framework=parallel_framework
        

    
    def run_batch(self)->dict[dict]:
        if self.parallel_framework=="ray":
            results=ray.get([run_episode_ray.remote(self.env,self.steps_per_episode) for i in range(self.episodes_per_batch)])
        elif self.parallel_framework=="native":
            pass
        return results
    
    def train(self):
        for i in range(self.number_of_batches):
            res=self.run_batch()
            data={agent.name:{} for agent in self.env.cells}
            for episode in res:
                for step in episode:
                    for agent in self.env.cells:
                        data[agent.name].setdefault("r",[]).append(step[1][agent.name])
                        data[agent.name].setdefault("a",[]).append(step[2][agent.name])
                        data[agent.name].setdefault("s",[]).append(step[3][agent.name])
                        data[agent.name].setdefault("log_prob",[]).append(step[4][agent.name])
                for agent in self.env.cells:
                    data[agent.name].setdefault("rtgs",[]).append(calculate_rtgs(data[agent.name]["r"][-len(episode):],agent.gamma))
        
            for agent in self.env.cells:
                data[agent.name]["r"]=torch.FloatTensor(np.array(data[agent.name]["r"]))
                data[agent.name]["a"]=torch.FloatTensor(np.array(data[agent.name]["a"]))
                data[agent.name]["s"]=torch.FloatTensor(np.array(data[agent.name]["s"]))
                data[agent.name]["rtgs"]=torch.FloatTensor(np.hstack(data[agent.name]["rtgs"]))
                data[agent.name]["log_prob"]=torch.FloatTensor(np.array(data[agent.name]["log_prob"]))
                for _ in range(agent.grad_updates):
                    v,lps=agent.evaluate(data[agent.name]["s"],data[agent.name]["a"])
                    a_k=data[agent.name]["rtgs"].unsqueeze(dim=1)-v.detach()
                    a_k=(a_k - a_k.mean()) / (a_k.std() + 1e-5)
                    ratios = torch.exp( lps - data[agent.name]["log_prob"])
                    surr1 = ratios * a_k
                    surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * a_k
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = torch.nn.MSELoss()(v,data[agent.name]["rtgs"])
                    self.optimizer_policy_.zero_grad()
                    actor_loss.backward(retain_graph=False)
                    agent.optimizer_policy_.step()
                    agent.optimizer_value_.zero_grad()
                    critic_loss.backward()
                    agent.optimizer_value_.step()
                    
                    

                    
            
            

                
                
                           

                
                
def calculate_rtgs(rewards:np.ndarray,gamma:float)->np.ndarray:
    rtgs=[]
    rtg=0
    for i in reversed(rewards):
        rtg=i+gamma*rtg
        rtgs.insert(0,rtg)
    return rtgs
    
    
    
def run_episode(env:Environment,num_steps:int)->list[tuple[dict]]:
    env.reset()
    return [env.step() for i in range(num_steps)]   

@ray.remote
def run_episode_ray(env:Environment,num_steps:int)->list[tuple[dict]]:
    env.reset()
    return [env.step() for i in range(num_steps)]


class Kinetic:
    """Objects of this class represent a kinetic model"""
    def __init__(self, name:str, parameters:dict)->None:
        self.name = name
        for key, value in parameters.items():
            setattr(self, key, value)
    
    def __str__(self) -> str:
        param_print=""
        for key, value in self.__dict__.items():
            param_print += f"{key} = {value}\n"
        return f"{self.name} kinetic model with parameters:\n{param_print}"

        
class Hill(Kinetic):
    """Objects of this class represent a Hill kinetic model"""
    def __init__(self,parameters:dict)->None:
        if {"n", "k"} != set(parameters.keys()):
            raise ValueError("Hill kinetic model requires only parameters n and k")
        super().__init__("Hill", parameters)

    def __call__(self,x)->float:
        return x**self.n/(self.k**self.n + x**self.n)

class MichaelisMenten(Kinetic):
    """Objects of this class represent a Michaelis Menten kinetic model"""
    def __init__(self, parameters:dict)->None:
        if {"k", "kcat"} != set(parameters.keys()):
            raise ValueError("Michaelis Menten kinetic model requires only parameters k and kcat")
        super().__init__("MichaekisMenten", parameters)

    def __call__(self,x:float)->float:
        return self.kcat*x/(self.k + x)

class PingPong(Kinetic):
    """Objects of this class represent a Ping Pong kinetic model"""
    def __init__(self, parameters:dict)->None:
        if {"ka", "kb","kab","vm"} != set(parameters.keys()):
            raise ValueError("Ping Pong kinetic model requires only parameters ka and kb and kab and vm")
        super().__init__("PingPong", parameters)

    def __call__(self,a:float,b:float)->float:
        return self.vm*a*b/(self.ka*a + self.kb*b + self.ka*self.kb)
    
class Linear(Kinetic):
    """Objects of this class represent a Linear kinetic model"""
    def __init__(self, parameters:dict)->None:
        if {"k"} != set(parameters.keys()):
            raise ValueError("Linear kinetic model requires only parameters k")
        super().__init__("Linear", parameters)

    def __call__(self,x:float)->float:
        return self.k*x

def toy_model_stoichiometry(model:Cell)->np.ndarray:
    """This function returns the stoichiometry of the toy model. Here is a look at the governing rules:
    r[1]: S -> S_in                     ::: r=PingPong(ka1, kb1, kab1, vm1)([S]   ,[t1])
    r[2]: S_in -> p21 I_1 + p22 E       ::: r=PingPong(ka2, kb2, kab2, vm2)([S_in],[e2])
    r[3]: I_1 -> p31 P + p32 E          ::: r=PingPong(ka3, kb3, kab3, vm3)([I_1] ,[e3])
    r[4]: S_in + r41 E ->  p42 NTP      ::: r=PingPong(ka4, kb4, kab4, vm4)([S_in],[e4])
    r[5]: NTP -> p51 NA                 ::: r=PingPong(ka5, kb5, kab5, vm5)([NTP] ,[e5])
    r[6]: I_1 + r61 E -> p62 Li         ::: r=PingPong(ka6, kb6, kab6, vm6)([I_1] ,[e6])
    r[7]: I_1 + r71 E -> p72 AA         ::: r=PingPong(ka7, kb7, kab7, vm7)([I_1] ,[e7])
    r[8]: AA + r81 E -> p82 e           ::: r=PingPong(ka8, kb8, kab8, vm8)([AA]  ,[e8])
    r[9]: P->P_out                      ::: r=Hill(n1,k1)([t2])
    r[10]: r101 AA + r102 Li -> W       ::: r=PingPong(ka10, kb10, kab10, vm10)([AA],[Li]) -> Should be very fast    
    _________________________________________________________________________________________________________
    e=e1+e2+e3+e4+e5+e6+e7+e8+t1       
    #Q: is this okay?

    r[]:e->t1
    r[]:e->e1
    r[]:e->e2
    r[]:e->e3
    r[]:e->e4
    r[]:e->e5
    r[]:e->e6
    r[]:e->e7
    r[]:e->e8
    r[]:e->t2
    ____________________________________________________________________________________________
    
    """
    s=np.zeros((len(model.state_variables),len(model.reactions)))
    s[[model.state_variables.index("S_env"),model.state_variables.index("S")],model.reactions.index("S_import")] = [-1,1]
    s[list(map(model.state_variables.index,["S",
    "I1", "E"])),model.reactions.index("S_to_I1")] = [-1,model.parameters["p21"],model.parameters["p22"]]
    
    s[list(map(model.state_variables.index,["I1","P","E"])),model.reactions.index("I1_to_P")] = [-1,model.parameters["p31"],model.parameters["p32"]]
    
    s[list(map(model.state_variables.index,["S","E","NTP"])),model.reactions.index("S_to_NTP")] = [-1,model.parameters["r41"],model.parameters["p42"]]
    
    s[list(map(model.state_variables.index,["NTP","NA"])),model.reactions.index("NTP_to_NA")] = [-1,model.parameters["p51"]]
    
    s[list(map(model.state_variables.index,["I1","E","Li"])),model.reactions.index("I1_to_Li")] = [-1,model.parameters["r61"],model.parameters["p62"]]
    
    s[list(map(model.state_variables.index,["I1","E","AA"])),
      model.reactions.index("I1_to_AA")] = [-1,model.parameters["r71"],model.parameters["p72"]]
    
    s[list(map(model.state_variables.index,["AA","E","e"])),
      model.reactions.index("AA_to_e")] = [-1,model.parameters["r81"],model.parameters["p82"]]
    
    s[[model.state_variables.index("P"),model.state_variables.index("P_env")],model.reactions.index("P_export")] = [-1,1]
    
    s[list(map(model.state_variables.index,["AA","Li","W"])),
      model.reactions.index("AA_and_li_to_W")] = [model.parameters["r101"],model.parameters["r102"],1]
    
    s[list(map(model.state_variables.index,["e","t1"])),model.reactions.index("e_to_t1")] = [-1,1]
    s[list(map(model.state_variables.index,["e","e1"])),model.reactions.index("e_to_e1")] = [-1,1]
    s[list(map(model.state_variables.index,["e","e2"])),model.reactions.index("e_to_e2")] = [-1,1]
    s[list(map(model.state_variables.index,["e","e3"])),model.reactions.index("e_to_e3")] = [-1,1]
    s[list(map(model.state_variables.index,["e","e4"])),model.reactions.index("e_to_e4")] = [-1,1]
    s[list(map(model.state_variables.index,["e","e5"])),model.reactions.index("e_to_e5")] = [-1,1]
    s[list(map(model.state_variables.index,["e","e6"])),model.reactions.index("e_to_e6")] = [-1,1]
    s[list(map(model.state_variables.index,["e","e7"])),model.reactions.index("e_to_e7")] = [-1,1]
    s[list(map(model.state_variables.index,["e","e8"])),model.reactions.index("e_to_e8")] = [-1,1]
    s[list(map(model.state_variables.index,["e","t2"])),model.reactions.index("e_to_t2")] = [-1,1]
    return s

def toy_model_ode(t:float, y:np.ndarray, model:Cell)->np.ndarray:
    ### First we update the dimensions of the cell
    y[y<0]=0
    model.shape.set_dimensions({key:y[model.state_variables.index(key)] for key in model.shape.dimensions.keys()})
    y[model.volume_index]=model.shape.volume
    
    if model.can_double(y):
        model.shape.set_dim_from_volume(model.shape.volume/2)
        for dim,val in model.shape.get_dimensions().items():
            y[model.state_variables.index(dim)] = val
        
        for i in model.cell_metabolites:
            y[model.state_variables.index(i)]=y[model.state_variables.index(i)]/2
        
        model.state[model.number_index]=model.state[model.number_index]*2
                
        
    ### Now we calculate the fluxes for each reaction
    actions=model.decide()[0]
    model.parameters.update(dict(zip(model.controlled_params,actions)))
    fluxes = np.zeros(len(model.reactions))
    fluxes[model.reactions.index("S_import")] = model.kinetics.setdefault("S_import",
                                                                          PingPong({"ka": model.parameters["ka1"],
                                                                                    "kb": model.parameters["kb1"],
                                                                                    "kab":model.parameters["kab1"],
                                                                                    "vm": model.parameters["vm1"]}))\
                                                                            (y[model.state_variables.index("S_env")],
                                                                             y[model.state_variables.index("t1")]/model.shape.area)*model.shape.area
    
    fluxes[model.reactions.index("S_to_I1")] = model.kinetics.setdefault("S_to_I1",
                                                                        PingPong({"ka":  model.parameters["ka2"],
                                                                                  "kb":  model.parameters["kb2"],
                                                                                  "kab": model.parameters["kab2"],
                                                                                  "vm":  model.parameters["vm2"]}))\
                                                                            (y[model.state_variables.index("S")]/model.shape.volume,
                                                                             y[model.state_variables.index("e2")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("I1_to_P")] = model.kinetics.setdefault("I1_to_P",
                                                                          PingPong({"ka": model.parameters["ka3"],
                                                                                    "kb": model.parameters["kb3"],
                                                                                    "kab":model.parameters["kab3"],
                                                                                    "vm": model.parameters["vm3"]}))\
                                                                             (y[model.state_variables.index("I1")]/model.shape.volume,
                                                                              y[model.state_variables.index("e3")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("S_to_NTP")] = model.kinetics.setdefault("S_to_NTP",
                                                                            PingPong({"ka":  model.parameters["ka4"], 
                                                                                     "kb":  model.parameters["kb4"], 
                                                                                     "kab": model.parameters["kab4"], 
                                                                                     "vm":  model.parameters["vm4"]}))\
                                                                                (y[model.state_variables.index("S")]/model.shape.volume,
                                                                                y[model.state_variables.index("e4")]/model.shape.volume)*model.shape.volume*y[model.state_variables.index("E")]
    
    fluxes[model.reactions.index("NTP_to_NA")] = model.kinetics.setdefault("NTP_to_NA",
                                                                            PingPong({"ka":  model.parameters["ka5"],
                                                                                      "kb":   model.parameters["kb5"],
                                                                                      "kab":  model.parameters["kab5"],
                                                                                      "vm":   model.parameters["vm5"]}))\
                                                                                (y[model.state_variables.index("NTP")]/model.shape.volume,
                                                                                 y[model.state_variables.index("e5")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("I1_to_Li")] = model.kinetics.setdefault("I1_to_Li",
                                                                            PingPong({"ka":  model.parameters["ka6"],
                                                                                      "kb":  model.parameters["kb6"],
                                                                                      "kab": model.parameters["kab6"],
                                                                                      "vm":  model.parameters["vm6"]}))\
                                                                                (y[model.state_variables.index("I1")]/model.shape.volume,
                                                                                 y[model.state_variables.index("e6")]/model.shape.volume)*model.shape.volume*y[model.state_variables.index("E")]
    
    fluxes[model.reactions.index("I1_to_AA")] = model.kinetics.setdefault("I1_to_AA",
                                                                            PingPong({  "ka": model.parameters["ka7"],
                                                                                        "kb": model.parameters["kb7"],
                                                                                        "kab":model.parameters["kab7"],
                                                                                        "vm": model.parameters["vm7"]}))\
                                                                                (y[model.state_variables.index("I1")]/model.shape.volume,
                                                                                 y[model.state_variables.index("e7")]/model.shape.volume)*model.shape.volume*y[model.state_variables.index("E")]
    
    fluxes[model.reactions.index("AA_to_e")] = model.kinetics.setdefault("AA_to_e",
                                                                        PingPong({"ka":   model.parameters["ka8"],
                                                                                  "kb":   model.parameters["kb8"],
                                                                                  "kab":  model.parameters["kab8"],
                                                                                  "vm":   model.parameters["vm8"]}))\
                                                                            (y[model.state_variables.index("AA")]/model.shape.volume,
                                                                             y[model.state_variables.index("e8")]/model.shape.volume)*model.shape.volume*y[model.state_variables.index("E")]
                                                                            
    fluxes[model.reactions.index("P_export")] = model.kinetics.setdefault("P_export",
                                                                        PingPong({
                                                                            "ka": model.parameters["ka9"],
                                                                            "kb": model.parameters["kb9"],
                                                                            "kab":model.parameters["kab9"],
                                                                            "vm": model.parameters["vm9"]}))(y[model.state_variables.index("P")]/model.shape.volume,
                                                                            y[model.state_variables.index("t2")]/model.shape.area)*model.shape.area
                                                                            
    fluxes[model.reactions.index("AA_and_li_to_W")] = model.kinetics.setdefault("AA_and_li_to_W",
                                                                                 PingPong({ "ka": model.parameters["ka10"],
                                                                                           "kb": model.parameters["kb10"],
                                                                                           "kab":model.parameters["kab10"],
                                                                                           "vm": model.parameters["vm10"]}))\
                                                                                  (y[model.state_variables.index("AA")]/model.shape.volume,
                                                                                    y[model.state_variables.index("Li")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("e_to_t1")] = model.kinetics.setdefault("e_to_t1",
                                                                        Linear({"k":model.parameters["k_t1"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("e_to_e1")] = model.kinetics.setdefault("e_to_e1",
                                                                        Linear({"k":model.parameters["k_e1"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume

    fluxes[model.reactions.index("e_to_e2")] = model.kinetics.setdefault("e_to_e2",
                                                                        Linear({"k":model.parameters["k_e2"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("e_to_e3")] = model.kinetics.setdefault("e_to_e3",
                                                                        Linear({"k":model.parameters["k_e3"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("e_to_e4")] = model.kinetics.setdefault("e_to_e4",
                                                                        Linear({"k":model.parameters["k_e4"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume

    fluxes[model.reactions.index("e_to_e5")] = model.kinetics.setdefault("e_to_e5",
                                                                        Linear({"k":model.parameters["k_e5"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("e_to_e6")] = model.kinetics.setdefault("e_to_e6",
                                                                        Linear({"k":model.parameters["k_e6"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("e_to_e7")] = model.kinetics.setdefault("e_to_e7",
                                                                        Linear({"k":model.parameters["k_e7"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("e_to_e8")] = model.kinetics.setdefault("e_to_e8",
                                                                        Linear({"k":model.parameters["k_e8"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume
    
    fluxes[model.reactions.index("e_to_t2")] = model.kinetics.setdefault("e_to_t2",
                                                                        Linear({"k":model.parameters["k_t2"]}))\
                                                                            (y[model.state_variables.index("e")]/model.shape.volume)*model.shape.volume

                                                                         
    v=np.matmul(model.stoichiometry(model),fluxes) 
    dvdt=model.parameters["lipid_density"]*v[model.state_variables.index("W")]
    for dim in model.shape.dimensions.keys():
        v[model.state_variables.index(dim)] = model.shape.calculate_differentials(dvdt)[dim]
    
    for i in model.env_metabolites:
        v[i]*=model.number
    return v

def forward_euler(ode:callable,initial_conditions:np.ndarray,t:np.ndarray,args:tuple)->np.ndarray:
    """This function solves an ordinary differential equation using the forward Euler method"""
    y=np.zeros((len(initial_conditions),len(t)))
    y[:,0]=initial_conditions
    dt=t[1]-t[0]
    for i in range(1,len(t)):
        y[:,i]=y[:,i-1]+dt*ode(t[i-1],y[:,i-1],*args)
    return y

if __name__ == "__main__":
    s=Sphere({"r":0.1,"t":0.5})
    cell=Cell("Toy Model",
              toy_model_stoichiometry,
              toy_model_ode,
              {"ka1":100,
               "kb1":10,
               "kab1":1,
               "vm1":100,
               "ka2":1,
               "kb2":1,
               "kab2":1,
               "vm2":1,
               "ka3":1,
               "kb3":1,
               "kab3":1,
               "vm3":1,
               "ka4":1,
               "kb4":1,
               "kab4":1,
               "vm4":1,
               "ka5":1,
               "kb5":1,
               "kab5":1,
               "vm5":1,
               "ka6":1,
               "kb6":1,
               "kab6":1,
               "vm6":0.1,
               "ka7":1,
               "kb7":1,
               "kab7":1,
               "vm7":1,
               "ka8":1,
               "kb8":1,
               "kab8":1,
               "vm8":1,
               "ka9":1,
               "kb9":1,
               "kab9":1,
               "vm9":1,
               "ka10":1,
               "kb10":1,
               "kab10":1,
               "vm10":1,
               "k_t1":10,
               "k_e1":1,
               "k_e2":1,
               "k_e3":1,
               "k_e4":1,
               "k_e5":1,
               "k_e6":10,
               "k_e7":1,
               "k_e8":1,
               "k_t2":1,
               "p21":1,
               "p22":1,
               "p31":1,
               "p32":1,
               "r41":-1,
               "p42":1,
               "p51":1,
               "r61":-1,
               "p62":1,
               "r71":-1,
               "p72":1,
               "r81":-1,
               "p82":1,
               "r101":-1,
               "r102":-1,
               "lipid_density":0.05,
               "split_volume":0.04,
               },
              TOY_REACTIONS,
              TOY_SPECIES,
              s,
              controlled_params=["k_e1",
                                 "k_e2",
                                 "k_e3",
                                 "k_e4",
                                 ],
              observable_env_states=["S_env","time_env","P_env"],
              initial_conditions={i:0.1 for i in TOY_SPECIES}
              
              )
    
    def S_controller(state:dict[str,float])->float:
        amplitude=10
        period=10
        return amplitude*np.abs(np.sin(2*np.pi*state["time_env"]/period))
         
    env=Environment(name="Toy Model Environment",
                    cells=[cell],
                    initial_conditions={"S_env":10},
                    extra_states=[],
                    controllers={"S_env":S_controller},time_step=0.01)
    
    trainer=Trainer(env,8,1000,10,10,"./",parallel_framework="ray")
    trainer.train()
    




        
    