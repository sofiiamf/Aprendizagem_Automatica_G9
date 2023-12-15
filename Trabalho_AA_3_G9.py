#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the data using pickle
X_train, X_ivs, y_train, col_names = pickle.load(open("drd2_data.pickle", "rb"))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# ### Scalling

# In[ ]:


# Scale the data using PowerTransformer
scaler = PowerTransformer()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_ivs_scaled = scaler.transform(X_ivs)


# ### Parameter Definition

# In[ ]:


MaxDepths = [1, 3, 5, 10]
MinSamplesLeafs = [1, 3, 5, 10]
NNeighbors = [1, 3, 5, 7, 9]
Weight = ["uniform","distance"]
Metric=['minkowski','euclidean','manhattan']
NEstimators = [1, 3, 5, 7]
gammas = [1e-1, 1e-2, 1e-3,1,10]
Cs = [1, 10, 100, 1e3]
alphas=[0.0001,0.1,1,10]
epsilons=[0.5,1.5]
activations=["relu","logistic","tanh"]


# ### GridSearchCV Parameter Grid Definition

# In[ ]:


param_gridDT = {"max_depth": MaxDepths, "min_samples_leaf": MinSamplesLeafs}
param_gridLR = {}
param_gridLassoR = {"alpha": alphas}
param_gridRidgeR = {"alpha": alphas}
param_gridKNNR = {"n_neighbors": NNeighbors, "weights": Weight, "metric": Metric}
param_gridSVMR = {"C": Cs,"epsilon":epsilons,"gamma":gammas}
param_gridRF = {"n_estimators": NEstimators, "max_depth": MaxDepths, "min_samples_leaf": MinSamplesLeafs}
param_gridMPR = {"alpha": alphas,"activation":activations}


# ### Trainig and Evaluation of Regression Models

# In[ ]:


models = {
    "Linear Regression": (LinearRegression(), param_gridLR),
    "Decision Tree": (DecisionTreeRegressor(), param_gridDT),
    "Random Forest": (RandomForestRegressor(), param_gridRF),
    "Lasso Regression": (Lasso(max_iter=9999999),param_gridLassoR),
    "Ridge Regression": (Ridge(max_iter=9999999),param_gridRidgeR),
    "KNN Regression": (KNeighborsRegressor(),param_gridKNNR),
    "SVR Regression": (SVR(),param_gridSVMR),
    "Multilayer Perceptron Regression": (MLPRegressor(hidden_layer_sizes=[10,10],max_iter=500),param_gridMPR),
}

fig, axs = plt.subplots(nrows=len(models), figsize=(8, 6 * len(models)))


# ### Evaluation

# In[ ]:


for i, (model_name, (model, params)) in enumerate(models.items()):
    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    preds = best_model.predict(X_ivs_scaled)
    preds = preds.clip(0, 1)  # Clip predictions between 0 and 1

    # Evaluate the model
    rmse = mean_squared_error(y_test, best_model.predict(X_test_scaled))
    RVE = explained_variance_score(y_test,  best_model.predict(X_test_scaled))
    print(f"{model_name}: MSE = {rmse},RVE={RVE}")

    # Save the predictions to a text file
    with open(f"{model_name}_predictions.txt", "w") as file:
        for pred in preds:
            file.write(f"{pred}\n")
        
     # Print the best hyperparameters
    best_params = grid_search.best_params_
    print(f"Best Hyperparameters for {model_name}: {best_params}")
    
    # Plot GridSearchCV results for each model
    cv_results = grid_search.cv_results_
    params_range = [f"{param}" for param in cv_results['params']]
    test_scores = cv_results['mean_test_score']

    axs[i].plot(params_range, -test_scores, marker='o')
    axs[i].set_title(f'{model_name} - GridSearchCV Results')
    axs[i].set_xlabel('Hyperparameters')
    axs[i].set_ylabel('Negative Mean Squared Error (MSE)')

plt.tight_layout()
plt.show()

