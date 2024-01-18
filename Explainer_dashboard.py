db = ExplainerDashboard(explainer, 
                        title="Breast cancer Explainer", # defaults to "Model Explainer"
                        shap_interaction=False, # you can switch off tabs with bools
                        )
db.run(port=8050)