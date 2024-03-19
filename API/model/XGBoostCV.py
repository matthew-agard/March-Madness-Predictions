import pandas as pd
from random import choice, seed
from xgboost import DMatrix, train as xgb_train, cv as xgb_cv

class XGBoostCV(object):
    def __init__(
        self, iterations, params, cross_vals, metrics, random_state=42, stratified=True, pandas=True
    ):
        self.iterations = iterations
        self.params = params
        self.cross_vals = cross_vals
        self.metrics = metrics
        self.stratified = stratified
        self.pandas = pandas
        self.random_state = seed(random_state)
        self.cv_results_ = pd.DataFrame(
            columns=['mean_test_score', 'std_test_score']
        )
        self.best_params_ = {}
        self.best_estimator_ = None


    def random_param_select(self):
        rand_params = {key: choice(self.params[key]) for key in self.params.keys()}
        return rand_params


    def set_best_estimator(self, model_tracker):
        best_iter = self.cv_results_['mean_test_score'].idxmax()

        best_boost_round = model_tracker.loc[best_iter, 'Best_Model_Iteration']
        self.best_params_ = model_tracker.loc[best_iter, 'Params']
        self.best_params_['n_estimators'] = best_boost_round

        self.best_estimator_ = xgb_train(params=self.best_params_, dtrain=self.data_matrix,
                                            num_boost_round=best_boost_round)      


    def fit(self, X, y):
        self.data_matrix = DMatrix(data=X, label=y)
        model_tracker = pd.DataFrame(columns=['Best_Model_Iteration', 'Params'])

        for iter in range(self.iterations):
            rand_params = self.random_param_select()

            performance_df = xgb_cv(params=rand_params, dtrain=self.data_matrix, nfold=self.cross_vals, 
                                    stratified=self.stratified, num_boost_round=rand_params['n_estimators'], 
                                    early_stopping_rounds=rand_params['n_estimators'] // 10,
                                    metrics=self.metrics, as_pandas=self.pandas)

            model_tracker.loc[iter] = [len(performance_df), rand_params]
            self.cv_results_.loc[iter] = performance_df.iloc[-1][
                ['test-error-mean', 'test-error-std']
            ].values.tolist()

        self.cv_results_['mean_test_score'] = 1-self.cv_results_['mean_test_score']

        self.set_best_estimator(model_tracker)