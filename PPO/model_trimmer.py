import cobra
from cobra.flux_analysis import flux_variability_analysis
import pickle
import scipy.sparse.linalg as spla
model=cobra.io.read_sbml_model("iAF1260.xml")
model_=model.copy()
for reaction in model_.exchanges:
    reaction.lower_bound=-1000
if __name__=="__main__":
    sol_fake=flux_variability_analysis(model_,model_.reactions,fraction_of_optimum=0.0,processes=8)
    can_be_removed=sol_fake[(sol_fake["maximum"]<0.0000001)&(sol_fake["minimum"]>-0.0000001)].index
    print("Number of reactions removed: ",len(can_be_removed))
    model.remove_reactions(can_be_removed)
    cobra.io.write_sbml_model(model,"iAF1260_trimmed.xml")
    