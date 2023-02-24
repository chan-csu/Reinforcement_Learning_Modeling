import cobra
from cobra.flux_analysis import flux_variability_analysis
import pickle
import scipy.sparse.linalg as spla


def trimmer(model:cobra.Model,report=True,tol=1e-6):
    model_=model.copy()
    for reaction in model_.exchanges:
        reaction.lower_bound=-1000
    sol_fake=flux_variability_analysis(model_,model_.reactions,fraction_of_optimum=0.0)
    can_be_removed=sol_fake[(sol_fake["maximum"]<tol)&(sol_fake["minimum"]>-tol)].index
    model.remove_reactions(can_be_removed)
    if report:
        print(f"Removed {len(can_be_removed)} reactions")
    return model

model_base = cobra.io.read_sbml_model("iML1515.xml")

if __name__=="__main__":
    trimmer(model_base)