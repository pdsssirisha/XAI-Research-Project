# Building a new EBM model
############## load packages ############
# import packages
import pandas as pd
from interpret.perf import ROC
from interpret import show
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

############## create EBM model #############
ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)

############## visualizations #############

# Generate global explanability visuals
global_exp=ebm.explain_global()
show(global_exp)

# Generate local explanability visuals
ebm_local = ebm.explain_local(X, y)
show(ebm_local)

# Generate EDA visuals 
hist = ClassHistogram().explain_data(X_train, y_train, name = 'Train Data')
show(hist)

# Package it all in one Dashboard , see image below
show([hist, ebm_local, ebm_perf,global_exp], share_tables=True)