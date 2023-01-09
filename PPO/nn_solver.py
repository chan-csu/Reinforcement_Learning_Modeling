import torch.nn as nn
import torch
import minicobra as mc
import numpy as np
import cobra

class IsFeasible(nn.Module):

    def __init__(self,inshape:int,hiddenshape:int):
        super(IsFeasible,self).__init__()
        self.inshape=inshape
        self.fc1=nn.Linear(inshape,hiddenshape)
        self.fc2=nn.Sequential(
            nn.Linear(hiddenshape,hiddenshape),nn.ReLU(),
            nn.Linear(hiddenshape,hiddenshape),nn.ReLU(),
            nn.Linear(hiddenshape,hiddenshape),nn.ReLU(),
            nn.Linear(hiddenshape,hiddenshape),nn.ReLU(),
            nn.Linear(hiddenshape,hiddenshape),nn.ReLU(),
        )
        self.fc3=nn.Linear(hiddenshape,1)
        self.logsoftmax=nn.LogSoftmax(dim=1)

    def forward(self,lb,ub):
        x=torch.cat((lb,ub),dim=1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        x=self.logsoftmax(x)
        return x


class LPRegressor(nn.Module):
    
    def __init__(self,num_rxns:int,hiddenshape:int):
        super(LPRegressor,self).__init__()
        self.inshape=num_rxns*2
        self.outshape=num_rxns
        self.fc1=nn.Linear(self.inshape,hiddenshape)
        self.fc2=nn.Sequential(
            nn.Linear(hiddenshape,hiddenshape),nn.ReLU(),
            nn.Linear(hiddenshape,hiddenshape),nn.ReLU(),
            nn.Linear(hiddenshape,hiddenshape),nn.ReLU(),
            nn.Linear(hiddenshape,hiddenshape),nn.ReLU(),
            nn.Linear(hiddenshape,hiddenshape),nn.ReLU(),
        )
        self.fc3=nn.Linear(hiddenshape,self.outshape)


    def forward(self,lb,ub):
        x=torch.cat((lb,ub),dim=1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x



def batch_generator(model:mc.Model,batch_size:int=64,scope:tuple[int,int]=(-100,100)):
    num_reactions=len(model.reactions)
    while True:
        lb=np.zeros((batch_size,num_reactions))
        ub=np.zeros((batch_size,num_reactions))
        feasible=np.zeros((batch_size,1))
        fluxes=np.zeros((batch_size,num_reactions))
        for i in range(batch_size):
            lb_=np.random.uniform(scope[0],scope[1],(1,num_reactions))
            ub_=np.random.uniform(lb_,scope[1],(1,num_reactions))
            lb[i,:]=lb_
            ub[i,:]=ub_
            model.lb=lb_
            model.ub=ub_
            gb_sol=model.optimize()
            if gb_sol.status==2:
                feasible[i]=1
                fluxes[i,:]=gb_sol.x
        
        yield lb,ub,feasible



    
class Solver:
    def __init__(self,feasiblity_classifier:IsFeasible,lp_regressor:LPRegressor):
        self.feasiblity_classifier=feasiblity_classifier
        self.lp_regressor=lp_regressor
    
    def optimize(self,model):
        if np.any(model.ub<model.lb):
            raise ValueError("Upper bounds must be greater than lower bounds")
        feasible=self.feasiblity_classifier(model)
        if feasible:
            return Solution("optimal",self.lp_regressor(model.lb,model.ub).detach.numpy())
        else:
            return Solution("infeasible",np.zeros((len(model.reactions),)))


class Solution:

    def __init__(self,status:str,fluxes:np.ndarray):

        self.status=status
        self.fluxes=fluxes


if __name__=="__main__":
    cbmodel=cobra.io.read_sbml_model("iAF1260.xml")
    mcmodel=mc.Model(reactions=cbmodel.reactions,metabolites=cbmodel.metabolites,objective="BIOMASS_Ec_iAF1260_core_59p81M")
    batch=batch_generator(mcmodel)
    counter=0
    while True:
        lb,ub,feasible=next(batch)
        print(counter)
        if np.any(feasible==0):
            print("Encountered infeasible problem")
            print(feasible)
        counter+=1
    
    