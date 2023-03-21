from cobra import Model, Reaction, Metabolite
import cobra
import numpy as np

ToyModel_SA_1 = Model('Toy_Model1')

### S_Uptake ###

S_Uptake = Reaction('Glc_1_e')
S = Metabolite('Glc_1', compartment='c')
S_Uptake.add_metabolites({S: -1})
S_Uptake.lower_bound = -20
S_Uptake.upper_bound = 0
ToyModel_SA_1.add_reactions([S_Uptake])

### ADP Production From Catabolism ###

ATP_Cat = Reaction('ATP_Cat')
ADP = Metabolite('ADP', compartment='c')
ATP = Metabolite('ATP', compartment='c')
S_x = Metabolite('S_x', compartment='c')
ATP_Cat.add_metabolites({ADP: -1, S: -1, S_x: 1, ATP: 1})
ATP_Cat.lower_bound = 0
ATP_Cat.upper_bound = 1000
ToyModel_SA_1.add_reactions([ATP_Cat])

### ATP Maintenance ###

ATP_M = Reaction('ATP_M')
ATP_M.add_metabolites({ATP: -1, ADP: 1})
ATP_M.lower_bound = 1
ATP_M.upper_bound = 100
ToyModel_SA_1.add_reactions([ATP_M])

### Biomass Production ###

X = Metabolite('X', compartment='c')
X_Production = Reaction('X_Production')
X_Production.add_metabolites({S_x: -1, ATP: -100, ADP: 100, X: 1})
X_Production.lower_bound = 0
X_Production.upper_bound = 1000
ToyModel_SA_1.add_reactions([X_Production])

### Biomass Release ###

X_Release = Reaction('X_Ex')
X_Release.add_metabolites({X: -1})
X_Release.lower_bound = 0
X_Release.upper_bound = 1000
ToyModel_SA_1.add_reactions([X_Release])

### Metabolism stuff ###

P = Metabolite('P', compartment='c')
P_Prod = Reaction('P_Prod')
P_Prod.add_metabolites({S_x: -1, ATP: 1, ADP: -1, P: 0.1})
P_Prod.lower_bound = 0
P_Prod.upper_bound = 1000
ToyModel_SA_1.add_reactions([P_Prod])

### Product Release ###

P_out = Reaction('P_e')
P_out.add_metabolites({P: -1})
P_out.lower_bound = 0
P_out.upper_bound = 1000
ToyModel_SA_1.add_reactions([P_out])
ToyModel_SA_1.objective = 'X_Ex'

### Amylase Production ###
Amylase_Prod = Reaction('Amylase_Prod')
Amylase_1 = Metabolite('Amylase_1', compartment='c')
Amylase_Prod.add_metabolites({S_x: -1, ATP: -1, ADP: 1, Amylase_1: 0.1})
Amylase_Prod.lower_bound = 0
Amylase_Prod.upper_bound = 1000
ToyModel_SA_1.add_reactions([Amylase_Prod])

### Amylase Exchange ###
Amylase_Ex = Reaction('Amylase_1_e')
Amylase_Ex.add_metabolites({Amylase_1: -1})
Amylase_Ex.lower_bound = 0
Amylase_Ex.upper_bound = 1000
ToyModel_SA_1.add_reactions([Amylase_Ex])

ToyModel_SA_1.biomass_ind=4
ToyModel_SA_1.exchange_reactions=tuple([ToyModel_SA_1.reactions.index(i) for i in ToyModel_SA_1.exchanges])

ToyModel_SA_2 = Model('Toy_Model2')
S_Uptake = Reaction('Glc_2_e')
S = Metabolite('Glc_2', compartment='c')
S_Uptake.add_metabolites({S: -1})
S_Uptake.lower_bound = -20
S_Uptake.upper_bound = 0
ToyModel_SA_2.add_reactions([S_Uptake])

### ADP Production From Catabolism ###
ATP_Cat = Reaction('ATP_Cat')
ADP = Metabolite('ADP', compartment='c')
ATP = Metabolite('ATP', compartment='c')
S_x = Metabolite('S_x', compartment='c')
ATP_Cat.add_metabolites({ADP: -1, S: -1, S_x: 1, ATP: 1})
ATP_Cat.lower_bound = 0
ATP_Cat.upper_bound = 1000
ToyModel_SA_2.add_reactions([ATP_Cat])

### ATP Maintenance ###
ATP_M = Reaction('ATP_M')
ATP_M.add_metabolites({ATP: -1, ADP: 1})
ATP_M.lower_bound = 1
ATP_M.upper_bound = 100
ToyModel_SA_2.add_reactions([ATP_M])

### Biomass Production ###
X = Metabolite('X', compartment='c')
X_Production = Reaction('X_Production')
X_Production.add_metabolites({S_x: -1, ATP: -100, ADP: 100, X: 1})
X_Production.lower_bound = 0
X_Production.upper_bound = 1000
ToyModel_SA_2.add_reactions([X_Production])

### Biomass Release ###
X_Release = Reaction('X_Ex')
X_Release.add_metabolites({X: -1})
X_Release.lower_bound = 0
X_Release.upper_bound = 1000
ToyModel_SA_2.add_reactions([X_Release])

### Metabolism stuff ###
P = Metabolite('P', compartment='c')
P_Prod = Reaction('P_Prod')
P_Prod.add_metabolites({S_x: -1, ATP: 1, ADP: -1, P: 0.1})
P_Prod.lower_bound = 0
P_Prod.upper_bound = 1000
ToyModel_SA_2.add_reactions([P_Prod])

### Product Release ###
P_out = Reaction('P_e')
P_out.add_metabolites({P: -1})
P_out.lower_bound = 0
P_out.upper_bound = 1000
ToyModel_SA_2.add_reactions([P_out])
ToyModel_SA_2.objective = 'X_Ex'

### Amylase Production ###
Amylase_Prod = Reaction('Amylase_Prod')
Amylase_2 = Metabolite('Amylase_2', compartment='c')
Amylase_Prod.add_metabolites({S_x: -1, ATP: -1, ADP: 1, Amylase_2: 0.1})
Amylase_Prod.lower_bound = 0
Amylase_Prod.upper_bound = 1000
ToyModel_SA_2.add_reactions([Amylase_Prod])

### Amylase Exchange ###
Amylase_Ex = Reaction('Amylase_2_e')
Amylase_Ex.add_metabolites({Amylase_2: -1})
Amylase_Ex.lower_bound = 0
Amylase_Ex.upper_bound = 1000
ToyModel_SA_2.add_reactions([Amylase_Ex])

ToyModel_SA_2.biomass_ind=4
ToyModel_SA_2.exchange_reactions=tuple([ToyModel_SA_2.reactions.index(i) for i in ToyModel_SA_2.exchanges])

ToyModel_SA_3 = Model('Toy_Model3')
S_Uptake = Reaction('Glc_3_e')
S = Metabolite('Glc_3', compartment='c')
S_Uptake.add_metabolites({S: -1})
S_Uptake.lower_bound = -20
S_Uptake.upper_bound = 0
ToyModel_SA_3.add_reactions([S_Uptake])

### ADP Production From Catabolism ###
ATP_Cat = Reaction('ATP_Cat')
ADP = Metabolite('ADP', compartment='c')
ATP = Metabolite('ATP', compartment='c')
S_x = Metabolite('S_x', compartment='c')
ATP_Cat.add_metabolites({ADP: -1, S: -1, S_x: 1, ATP: 1})
ATP_Cat.lower_bound = 0
ATP_Cat.upper_bound = 1000

ToyModel_SA_3.add_reactions([ATP_Cat])

### ATP Maintenance ###
ATP_M = Reaction('ATP_M')
ATP_M.add_metabolites({ATP: -1, ADP: 1})
ATP_M.lower_bound = 1
ATP_M.upper_bound = 100
ToyModel_SA_3.add_reactions([ATP_M])

### Biomass Production ###
X = Metabolite('X', compartment='c')
X_Production = Reaction('X_Production')
X_Production.add_metabolites({S_x: -1, ATP: -100, ADP: 100, X: 1})  
X_Production.lower_bound = 0
X_Production.upper_bound = 1000
ToyModel_SA_3.add_reactions([X_Production])

### Biomass Release ###
X_Release = Reaction('X_Ex')
X_Release.add_metabolites({X: -1})
X_Release.lower_bound = 0
X_Release.upper_bound = 1000
ToyModel_SA_3.add_reactions([X_Release])

### Metabolism stuff ###
P = Metabolite('P', compartment='c')
P_Prod = Reaction('P_Prod')
P_Prod.add_metabolites({S_x: -1, ATP: 1, ADP: -1, P: 0.1})
P_Prod.lower_bound = 0
P_Prod.upper_bound = 1000
ToyModel_SA_3.add_reactions([P_Prod])

### Product Release ###
P_out = Reaction('P_e')
P_out.add_metabolites({P: -1})
P_out.lower_bound = 0
P_out.upper_bound = 1000
ToyModel_SA_3.add_reactions([P_out])
ToyModel_SA_3.objective = 'X_Ex'

### Amylase Production ###
Amylase_Prod = Reaction('Amylase_Prod')
Amylase_3 = Metabolite('Amylase_3', compartment='c')
Amylase_Prod.add_metabolites({S_x: -1, ATP: -1, ADP: 1, Amylase_3: 0.1})
Amylase_Prod.lower_bound = 0
Amylase_Prod.upper_bound = 1000
ToyModel_SA_3.add_reactions([Amylase_Prod])

### Amylase Exchange ###
Amylase_Ex = Reaction('Amylase_3_e')
Amylase_Ex.add_metabolites({Amylase_3: -1})
Amylase_Ex.lower_bound = 0
Amylase_Ex.upper_bound = 1000
ToyModel_SA_3.add_reactions([Amylase_Ex])

ToyModel_SA_3.biomass_ind=4
ToyModel_SA_3.exchange_reactions=tuple([ToyModel_SA_3.reactions.index(i) for i in ToyModel_SA_3.exchanges])

ToyModel_SA_4 = Model('Toy_Model4')
S_Uptake = Reaction('Glc_4_e')
S = Metabolite('Glc_4', compartment='c')  
S_Uptake.add_metabolites({S: -1})
S_Uptake.lower_bound = -20
S_Uptake.upper_bound = 0
ToyModel_SA_4.add_reactions([S_Uptake])

### ADP Production From Catabolism ###
ATP_Cat = Reaction('ATP_Cat')
ADP = Metabolite('ADP', compartment='c')
ATP = Metabolite('ATP', compartment='c')
S_x = Metabolite('S_x', compartment='c')
ATP_Cat.add_metabolites({ADP: -1, S: -1, S_x: 1, ATP: 1})
ATP_Cat.lower_bound = 0
ATP_Cat.upper_bound = 1000

ToyModel_SA_4.add_reactions([ATP_Cat])

### ATP Maintenance ###
ATP_M = Reaction('ATP_M')
ATP_M.add_metabolites({ATP: -1, ADP: 1})
ATP_M.lower_bound = 1
ATP_M.upper_bound = 100
ToyModel_SA_4.add_reactions([ATP_M])

### Biomass Production ###
X = Metabolite('X', compartment='c')
X_Production = Reaction('X_Production')
X_Production.add_metabolites({S_x: -1, ATP: -100, ADP: 100, X: 1})
X_Production.lower_bound = 0
X_Production.upper_bound = 1000
ToyModel_SA_4.add_reactions([X_Production])

### Biomass Release ###
X_Release = Reaction('X_Ex')
X_Release.add_metabolites({X: -1})
X_Release.lower_bound = 0
X_Release.upper_bound = 1000
ToyModel_SA_4.add_reactions([X_Release])

### Metabolism stuff ###
P = Metabolite('P', compartment='c')
P_Prod = Reaction('P_Prod')
P_Prod.add_metabolites({S_x: -1, ATP: 1, ADP: -1, P: 0.1})
P_Prod.lower_bound = 0
P_Prod.upper_bound = 1000
ToyModel_SA_4.add_reactions([P_Prod])

### Product Release ###
P_out = Reaction('P_e')
P_out.add_metabolites({P: -1})
P_out.lower_bound = 0
P_out.upper_bound = 1000
ToyModel_SA_4.add_reactions([P_out])
ToyModel_SA_4.objective = 'X_Ex'

### Amylase Production ###
Amylase_Prod = Reaction('Amylase_Prod')
Amylase_4 = Metabolite('Amylase_4', compartment='c')

Amylase_Prod.add_metabolites({S_x: -1, ATP: -1, ADP: 1, Amylase_4: 0.1})
Amylase_Prod.lower_bound = 0
Amylase_Prod.upper_bound = 1000
ToyModel_SA_4.add_reactions([Amylase_Prod])

### Amylase Exchange ###
Amylase_Ex = Reaction('Amylase_4_e')
Amylase_Ex.add_metabolites({Amylase_4: -1})
Amylase_Ex.lower_bound = 0
Amylase_Ex.upper_bound = 1000
ToyModel_SA_4.add_reactions([Amylase_Ex])

ToyModel_SA_4.biomass_ind=4
ToyModel_SA_4.exchange_reactions=tuple([ToyModel_SA_4.reactions.index(i) for i in ToyModel_SA_4.exchanges])

ToyModel_SA_5 = Model('Toy_Model5')
S_Uptake = Reaction('Glc_5_e')
S = Metabolite('Glc_5', compartment='c')
S_Uptake.add_metabolites({S: -1})
S_Uptake.lower_bound = -20
S_Uptake.upper_bound = 0
ToyModel_SA_5.add_reactions([S_Uptake])

### ADP Production From Catabolism ###
ATP_Cat = Reaction('ATP_Cat')
ADP = Metabolite('ADP', compartment='c')
ATP = Metabolite('ATP', compartment='c')
S_x = Metabolite('S_x', compartment='c')
ATP_Cat.add_metabolites({ADP: -1, S: -1, S_x: 1, ATP: 1})
ATP_Cat.lower_bound = 0
ATP_Cat.upper_bound = 1000

ToyModel_SA_5.add_reactions([ATP_Cat])

### ATP Maintenance ###
ATP_M = Reaction('ATP_M')

ATP_M.add_metabolites({ATP: -1, ADP: 1})
ATP_M.lower_bound = 1
ATP_M.upper_bound = 100
ToyModel_SA_5.add_reactions([ATP_M])

### Biomass Production ###
X = Metabolite('X', compartment='c')
X_Production = Reaction('X_Production')

X_Production.add_metabolites({S_x: -1, ATP: -100, ADP: 100, X: 1})
X_Production.lower_bound = 0
X_Production.upper_bound = 1000
ToyModel_SA_5.add_reactions([X_Production])

### Biomass Release ###
X_Release = Reaction('X_Ex')
X_Release.add_metabolites({X: -1})
X_Release.lower_bound = 0
X_Release.upper_bound = 1000
ToyModel_SA_5.add_reactions([X_Release])

### Metabolism stuff ###
P = Metabolite('P', compartment='c')
P_Prod = Reaction('P_Prod')

P_Prod.add_metabolites({S_x: -1, ATP: 1, ADP: -1, P: 0.1})
P_Prod.lower_bound = 0
P_Prod.upper_bound = 1000
ToyModel_SA_5.add_reactions([P_Prod])

### Product Release ###
P_out = Reaction('P_e')
P_out.add_metabolites({P: -1})
P_out.lower_bound = 0
P_out.upper_bound = 1000
ToyModel_SA_5.add_reactions([P_out])
ToyModel_SA_5.objective = 'X_Ex'

### Amylase Production ###
Amylase_Prod = Reaction('Amylase_Prod')
Amylase_5 = Metabolite('Amylase_5', compartment='c')

Amylase_Prod.add_metabolites({S_x: -1, ATP: -1, ADP: 1, Amylase_5: 0.1})
Amylase_Prod.lower_bound = 0
Amylase_Prod.upper_bound = 1000
ToyModel_SA_5.add_reactions([Amylase_Prod])

### Amylase Exchange ###
Amylase_Ex = Reaction('Amylase_5_e')
Amylase_Ex.add_metabolites({Amylase_5: -1})
Amylase_Ex.lower_bound = 0
Amylase_Ex.upper_bound = 1000
ToyModel_SA_5.add_reactions([Amylase_Ex])

ToyModel_SA_5.biomass_ind=5
ToyModel_SA_5.exchange_reactions=tuple([ToyModel_SA_5.reactions.index(i) for i in ToyModel_SA_5.exchanges])
