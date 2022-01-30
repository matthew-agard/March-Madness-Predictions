from inspect import Parameter
import numpy as np
import pandas as pd
from xgboost import DMatrix, train as xgb_train, cv as xgb_cv

class XGBoostCV(object):
    def __init__(
        self, iterations, params, cross_vals, metrics, boost_rounds=100, early_stopping_rounds=10,
        random_state=42, stratified=True, pandas=True
    ):
        self.iterations = iterations
        self.params = params
        self.cross_vals = cross_vals
        # self.boost_rounds = boost_rounds
        # self.early_stopping_rounds = early_stopping_rounds
        self.metrics = metrics
        self.stratified = stratified
        self.pandas = pandas
        # self.random_state = np.random.seed(random_state)
        self.cv_results_ = None
        self.best_params_ = None
        self.best_estimator_ = None


    def fit(self, X, y):
        self.data_matrix = DMatrix(data=X, label=y)

        for iter in range(self.iterations):
            rand_params = {key: np.random.choice(self.params[key]) for key in self.params.keys()}

            print(rand_params)

            performance_df = xgb_cv(params=rand_params, dtrain=self.data_matrix, nfold=self.cross_vals, stratified=self.stratified, 
                                    num_boost_round=rand_params['n_estimators'], early_stopping_rounds=rand_params['n_estimators'] // 10,
                                    metrics=self.metrics, as_pandas=self.pandas)

            return performance_df


    def get_best_estimator(self):
        self.best_estimator_ = xgb_train(params=self.best_params_, dtrain=self.data_matrix, 
                                    evals=(dval, 'val_set'), num_boost_round=250, 
                                    early_stopping_rounds=10)

        return self.best_estimator_