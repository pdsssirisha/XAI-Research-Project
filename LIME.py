# Utilizing our same xgb_mod model object created above
# Import pacakages
import lime
import lime.lime_tabular
import numpy as np
import xgboost

############## create explainer ###########
# we use the dataframes splits created above for SHAP
explainer = 
lime.lime_tabular.LimeTabularExplainer(X_test.to_numpy(), feature_names=X_test.columns, class_names=['0','1'], verbose=True)

############## visualizations #############
exp = explainer.explain_instance(X_np[79], xgb_mod.predict_proba, num_features=20)
exp.show_in_notebook(show_table=True)