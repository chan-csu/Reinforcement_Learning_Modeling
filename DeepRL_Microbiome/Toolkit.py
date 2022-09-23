from distutils.log import warn
import cobra
import torch
import numpy as np

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
                agents:list,
                extracellular_reactions:list[dict],
                dt:float=0.1,
                ) -> None:
        self.agents = agents
        self.num_agents = len(agents)
        self.extracellular_reactions = extracellular_reactions
        self.mapping_matrix=self.resolve_exchanges()
        self.species=self.extract_species()
        self.resolve_extracellular_reactions(extracellular_reactions)

    
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
        warn("The following species are not in the community: {}".format(new_species))
        print("Adding the following species to the community: {}".format(new_species))
        self.species.extend(new_species)



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
