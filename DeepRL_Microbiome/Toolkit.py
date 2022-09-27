from distutils.log import warn
import cobra
import torch
import numpy as np
import random
class Simulation:
    pass 




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
                initial_condition:np.ndarray,
                dt:float=0.1,
                dilution_rate:float=0.1,
                
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
        if initial_condition.shape[0]!=len(self.species):
            raise ValueError("The initial condition does not match the number of extracellular species in the community.")
        else:
            self.initial_condition = initial_condition
            self.reset()

    
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
            M.a=M.actor_network(torch.FloatTensor([self.state[M.observables]])).detach().numpy()[0]
            if random.random()<M.epsilon:
                M.a+=np.random.uniform(low=-1, high=1,size=len(M.actions))

            else:

                pass
            
            for index,item in enumerate(self.mapping_matrix["Ex_sp"]):
                if self.mapping_matrix['Mapping_Matrix'][index,i]!=-1:
                    M.reactions[self.mapping_matrix['Mapping_Matrix'][index,i]].upper_bound=100
                    M.reactions[self.mapping_matrix['Mapping_Matrix'][index,i]].lower_bound=-general_uptake_kinetics(C[index+len(Models)])


        for index,flux in enumerate(M.actions):

            if M.a[index]<0:
            
                M.reactions[M.actions[index]].lower_bound=max(M.a[index],M.reactions[M.actions[index]].lower_bound)
                # M.reactions[M.actions[index]].lower_bound=M.a[index]*M.reactions[M.actions[index]].lower_bound
    
            else:
                M.reactions[M.actions[index]].lower_bound=min(M.a[index],M.reactions[M.actions[index]].upper_bound)



            Sols[i] = self.agents[i].optimize()

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
                
        self.state += dCdt*self.dt
        return self.state,(i.reward for i in self.agents),(i.a for i in self.agents)


    def batch_step(self):
        """ Performs a batch of steps in the environment in parallel.
        This is just an experimental feature and is not yet implemented."""
        pass


class Agent:
    """ Any microbial agent will be an instance of this class.
    """
    def __init__(self,
                name:str,
                model:cobra.Model,
                actor_network:torch.nn.modules,
                critic_network:torch.nn.modules,
                gamma:float,
                update_batch_size:int) -> None:
        self.name = name
        self.model = model
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.gamma = gamma
        self.update_batch_size = update_batch_size
        
def Build_Mapping_Matrix(Models):
    """
    Given a list of COBRA model objects, this function will build a mapping matrix

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
