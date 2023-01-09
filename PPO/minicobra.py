import cobra
import numpy as np

class Model:
    """A simple model similar to cobra model that is supposed to 
    run faster with gurobipy. This is not a good substitute for cobra,
    however, it avoids most of the validations and interfaces that cobra uses
    with the goal of making it faster."""

    def __init__(self,reactions:list[cobra.Reaction],metabolites:list[cobra.Metabolite],objective=str):
        self.reactions = reactions
        self.metabolites = metabolites
        self.objective = self.index_objective(objective)
        self.update_stoichiometric_matrix()


    def remove_reactions(self,reactions):
        """Remove reactions from the model"""
        for reaction in reactions:
            self.reactions.remove(reaction)

    
    def add_reactions(self,reactions):
        """Add reactions to the model"""
        for reaction in reactions:
            self.reactions.append(reaction)

    def update_stoichiometric_matrix(self):
        """Update the stoichiometric matrix based on metabolites and reactions"""
        s=np.zeros((len(self.reactions),len(self.metabolites)))
        for i,reaction in enumerate(self.reactions):
            for metabolite in reaction.metabolites:
                s[i,self.metabolites.index(metabolite)]=reaction.metabolites[metabolite]
        self.s=s

    def index_objective(self,objective):
        """finds the index of the objective function"""
        return self.reactions.index(objective)

            

if __name__=='__main__':
    model=cobra.io.read_sbml_model("iAF1260.xml")
    mini_model=Model(model.reactions,model.metabolites)
    mini_model.update_stoichiometric_matrix()
    mini_model.s