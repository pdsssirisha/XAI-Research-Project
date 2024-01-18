# Utilizing our same xgb_mod model object created above
############## load packages ############
# import packages
import numpy as np
import pandas as pd
import xgboost as xgb
import datatable as dt # data table factory
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import lime
import eli5
from eli5 import show_weights
from eli5 import show_prediction
from eli5.sklearn import PermutationImportance

############## visualizations #############
# Generate  global importances - ['weight', 'gain', 'cover', 'total_gain', 'total_cover'] - options for importance type
show_weights(xgb_mod,importance_type = 'total_gain')

# Local level explanation 
eli5.show_prediction(xgb_mod, X_test.iloc[1],
                    feature_names=list(X.columns),
                    show_feature_values=True)

# permutation based importance 
# sorted(sklearn.metrics.SCORERS.keys()) to check for options for scoring in Permutation importance
perm = PermutationImportance(xgb_mod,scoring="roc_auc_ovr_weighted")
perm.fit(X_test, y_test)
eli5.show_weights(perm, feature_names=list(X.columns))