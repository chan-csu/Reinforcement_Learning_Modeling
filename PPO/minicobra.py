import cobra
import numpy as np
from functools import cached_property
import docplex

class Model:
    """A simple model similar to cobra model that is supposed to 
    run faster with gurobipy. This is not a good substitute for cobra,
    however, it avoids most of the validations and interfaces that cobra uses
    with the goal of making it faster."""

    def __init__(self,name:str,reactions:list[cobra.Reaction],metabolites:list[cobra.Metabolite],objective=str):
        self.reactions = reactions
        self.metabolites = metabolites
        self._objective=objective
        self.reactionids=   [reaction.id for reaction in self.reactions]
        self.metaboliteids= [metabolite.id for metabolite in self.metabolites]
        self.Biomass_Ind=self.objective
        self.lb=np.array([reaction.lower_bound for reaction in self.reactions],dtype="float32")
        self.ub=np.array([reaction.upper_bound for reaction in self.reactions],dtype="float32")
        self.s=self._s()
        self.exchanges=self._exchanges()
        self.name=name


    def remove_reactions(self,reactions):
        """Remove reactions from the model"""
        for reaction in reactions:
            self.reactions.remove(reaction)
        

    
    def add_reactions(self,reactions):
        """Add reactions to the model"""
        for reaction in reactions:
            self.reactions.append(reaction)
    
    
    def _s(self):
        """Update the stoichiometric matrix based on metabolites and reactions"""
        s=np.zeros((len(self.metabolites),len(self.reactions)))
        for i,reaction in enumerate(self.reactions):
            for metabolite in reaction.metabolites:
                s[self.metabolites.index(metabolite),i]=reaction.metabolites[metabolite]
        return s

    
    def objective(self,value):
        self._objective=value

    def _init_pulp_model(self):
        """Initializes a LP model for pulp based on the stoichiometric matrix and initial lb and ub"""
        model = pulp.LpProblem(f"{self.name}", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", self.reactionids, lowBound=self.lb, upBound=self.ub, cat=pulp.LpContinuous)


    
    def _exchanges(self):
        """Return the exchange reactions"""
        ex_rxn_inds=list(np.where(self.s.sum(axis=0)==-1)[0])
        return([self.reactions[i] for i in ex_rxn_inds ])


# class Solution:

#     def __init__(self,model:Model):
#         self.=model
#         self._solution=model.optimize()

            


