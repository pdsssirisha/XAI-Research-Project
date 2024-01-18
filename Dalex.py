# Utilizing our same xgb_mod model object created above
############## load packages ############
import numpy as np
import pandas as pd
import xgboost as xgb
import dalex as dx
import datatable as dt # data table factory
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import shap
import ipywidgets
from ipywidgets import IntProgress
import statsmodels
import lime
import warnings
import flask
import flask_cors
import requests
explainer = dx.Explainer(xgb_mod,X,y) # create explainer from Dalex

############## visualizations #############
# Generate importance plot showing top 30
explainer.model_parts().plot(max_vars=30)

# Generate ROC curve for xgboost model object
explainer.model_performance(model_type='classification').plot(geom='roc')

# Generate breakdown plot
explainer.predict_parts(X.iloc[79, :]).plot(max_vars=15)

# Generate SHAP plot 
explainer.predict_parts(X.iloc[79, :],type="shap").plot(min_max=[0,1],max_vars=15)

# Generate breakdown interactions plot 
explainer.predict_parts(X.iloc[79, :], type='break_down_interactions').plot(max_vars=20)

# Generate residual plots
explainer.model_performance(model_type = 'classification').plot()

# Generate PDP plots for all variables 
explainer.model_profile(type = 'partial', label="pdp").plot()

# Generate Accumulated Local Effects plots for all variables 
explainer.model_profile(type = 'ale', label="pdp").plot()

# Generate Individual Conditional Expectation plots for worst texture variable 
explainer.model_profile(type = 'conditional', label="conditional",variables="worst texture")

# Generate lime breakdown plot
explainer.predict_surrogate(X.iloc[[79]]).plot()

####### start Arena dashboard #############
# create empty Arena
arena=dx.Arena()

# push created explainer
arena.push_model(explainer)

# push whole test dataset (including target column)
arena.push_observations(X_test)

# run server on port 9294
arena.run_server(port=9291)