"""Model Selection Helper Functions

This script is used as a module in the March_Madness_Predictions Jupyter notebooks.

The following functions are present:
    * init_knn
    * init_naive_bayes
    * init_logreg
    * init_svm
    * init_rf
    * get_cv_models

Requires a minimum of the 'numpy' and 'sklearn' libraries being present 
in your environment to run.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier


def init_naive_bayes(y):
    """Initialize Naive Bayes model

    Parameters
    -------
    y : list
        Historical tournament game target variables

    Returns
    -------
    list
        Collection of model, its parameters, and what CV search to perform
    """
    gnb = GaussianNB()
    gnb_params = {
        'priors': [None, list(y.value_counts(normalize=True))],
    }

    return ['Grid', gnb, gnb_params]


def init_logreg():
    """Initialize Logistic Regression model

    Returns
    -------
    list
        Collection of model, its parameters, and what CV search to perform
    """
    lr = LogisticRegression()
    lr_params = {
        'C': [10**i for i in range(-5, 6)],
        'random_state': [42],
    }

    return ['Grid', lr, lr_params]


def init_svm():
    """Initialize Support Vector Machine model

    Returns
    -------
    list
        Collection of model, its parameters, and what CV search to perform
    """
    svm = LinearSVC()
    svm_params = {
        'dual': [False],
        'C': [10**i for i in range(-5, 6)],
        'random_state': [42],
    }

    return ['Grid', svm, svm_params]


def init_rf():
    """Initialize Random Forest model

    Returns
    -------
    list
        Collection of model, its parameters, and what CV search to perform
    """
    rf = RandomForestClassifier()
    rf_params = {
        'n_estimators': np.arange(100, 301, 25),
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2**i for i in range(1, 6)],
        'min_samples_leaf': [2**i for i in range(1, 6)],
        'random_state': [42],
    }

    return ['Random', rf, rf_params]


def init_xgboost():
    xgb = XGBClassifier()
    xgb_params = {
        'n_estimators': np.arange(100, 301, 25),
        'learning_rate': np.arange(0.05, 0.31, 0.05),
        'subsample': np.arange(0.5, 1.01, 0.1),
        'verbosity': [0],
        'seed': [42],
    }

    """Consider applying XGBoost.cv once best model is selected from RandomizedSearchCV"""
    # model_cv = XGB_CV(params=params[2], data_dmatrix=DMatrix(data=X, label=y), num_boost_round=50, 
    # nfold=cross_vals, stratified=True, metrics=['auc', 'error'], 
    # early_stopping_rounds=10, as_pandas=True, seed=42)

    return ['XGBoost', xgb, xgb_params]


def get_cv_models(y):
    """Get all models

    Parameters
    -------
    y : list
        Historical tournament game target variables

    Returns
    -------
    dict
        Dictionary of all models upon which to perform CV search
    """
    cv_models = {
        'Naive Bayes': init_naive_bayes(y),
        'LogReg': init_logreg(),
        'SVM': init_svm(),
        'Random Forest': init_rf(),
        # 'XGBoost': init_xgboost(),
    }

    return cv_models