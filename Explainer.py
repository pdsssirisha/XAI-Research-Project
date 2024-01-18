# Utlizing our same xgb_model object created above

# import the packages 
import numpy as np
import pandas as pd
import xgboost as xgb
import datatable as dt # data table factory
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import explainerdashboard as expdb
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard import InlineExplainer
from explainerdashboard.custom import (ImportancesComposite,
                                       IndividualPredictionsComposite,
                                       WhatIfComposite,
                                       ShapDependenceComposite,
                                       ShapInteractionsComposite,
                                       DecisionTreesComposite)

# Create the explainer object
explainer = ClassifierExplainer(xgb_mod, X_test, y_test,model_output='logodds')

# Create individual component plants using Inexplainer

ie = InlineExplainer(explainer)

# SHAP overview
ie.shap.overview()

# SHAP interactions
ie.shap.interaction_dependence()

# Model Stats
ie.classifier.model_stats()

# SHAP contribution
ie.shap.contributions_graph()

# SHAP dependence
ie.shap.dependence()