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
        'C': [10**i for i in range(-3, 3)],
        'penalty': ['l1', 'l2'],
        'solver': ['saga', 'liblinear'],
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
        'penalty': ['l1', 'l2'],
        'C': [10**i for i in range(-3, 3)],
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
        'n_estimators': np.arange(100, 201, 25),
        'criterion': ['entropy'],
        'min_samples_split': [2**i for i in range(3, 7)],
        'min_samples_leaf': [2**i for i in range(1, 5)],
        'random_state': [42],
    }

    return ['Random', rf, rf_params]


def init_xgb():
    xgb = XGBClassifier()
    xgb_params = {
        'n_estimators': np.arange(150, 301, 25),
        'learning_rate': [0.01, 0.025, 0.05, 0.1],
        'subsample': np.arange(0.4, 0.7, 0.1),
        'tree_method': ['gpu_hist'],
        'sampling_method': ['gradient_based'],
        'lambda': [100],
        'eval_metric': ['error'],
        'verbosity': [0],
        'seed': [42],
    }

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
        'XGBoost': init_xgb(),
    }

    return cv_models