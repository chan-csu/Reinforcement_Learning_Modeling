import cobra
import os
Main_dir = os.path.dirname(os.path.abspath(__file__))
Model = cobra.io.read_sbml_model(Main_dir+'/IJO1366_AP.xml')
S=cobra.util.array.constraint_matrices(Model, array_type='dense')
S