import cobra
import numpy as np
from functools import cached_property
from gurobipy import Model as gurobiModel
from gurobipy import GRB

class Model:
    """A simple model similar to cobra model that is supposed to 
    run faster with gurobipy. This is not a good substitute for cobra,
    however, it avoids most of the validations and interfaces that cobra uses
    with the goal of making it faster."""

    def __init__(self,reactions:list[cobra.Reaction],metabolites:list[cobra.Metabolite],objective=str):
        self.reactions = reactions
        self.metabolites = metabolites
        self._objective=objective
        self.reactionids=   [reaction.id for reaction in self.reactions]
        self.metaboliteids= [metabolite.id for metabolite in self.metabolites]
        self.empty_gb_model=gurobiModel()
        self.Biomass_Ind=self.objective
        self.lb=np.array([reaction.lower_bound for reaction in self.reactions],dtype="float32")
        self.ub=np.array([reaction.upper_bound for reaction in self.reactions],dtype="float32")
        self.s=self.s()


    def remove_reactions(self,reactions):
        """Remove reactions from the model"""
        for reaction in reactions:
            self.reactions.remove(reaction)
        

    
    def add_reactions(self,reactions):
        """Add reactions to the model"""
        for reaction in reactions:
            self.reactions.append(reaction)
    
    
    def s(self):
        """Update the stoichiometric matrix based on metabolites and reactions"""
        s=np.zeros((len(self.metabolites),len(self.reactions)))
        for i,reaction in enumerate(self.reactions):
            for metabolite in reaction.metabolites:
                s[self.metabolites.index(metabolite),i]=reaction.metabolites[metabolite]
        self.s=s
        return s

    @property
    def objective(self):
        return self.reactions.index(self._objective)

    @objective.setter
    def objective(self,value):
        self._objective=value

    def optimize(self):
        """Optimize the model"""
        model=self.empty_gb_model.copy()
        model.setParam('OutputFlag',False)
        x=model.addMVar(shape=self.s.shape[1],lb=self.lb,ub=self.ub,vtype=GRB.CONTINUOUS,name="x")
        b=np.zeros((self.s.shape[0],))
        model.addMConstr(self.s,x,"=",b)
        model.setObjective(x[self.objective],GRB.MAXIMIZE)
        model.optimize()
        return model

    @property
    def exchanges(self):
        """Return the exchange reactions"""
        ex_rxn_inds=list(np.where(self.s.sum(axis=0)==-1)[0])
        return([self.reactions[i] for i in ex_rxn_inds ])


# class Solution:

#     def __init__(self,model:Model):
#         self.=model
#         self._solution=model.optimize()

            


