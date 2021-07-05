import numpy as np
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from time import time

REGRESSORS = [SVR, KNeighborsRegressor, MLPRegressor, RandomForestRegressor]

class combinedRegressor():
    def __init__(self, training_time_penalty = 0.001):
        self.training_time_penalty = training_time_penalty
        self.weights = None
        for r in REGRESSORS:
            setattr(self, r.__name__ + '_params', None)
        self.regressor = None
    
    def random(training_time_penalty = 0.001):
        random = combinedRegressor(training_time_penalty = training_time_penalty)
        estimators = []
        random.weights = { r : np.random.lognormal(mean=np.log(1)) for r in REGRESSORS}
        for r in REGRESSORS:
            setattr(random, r.__name__ + '_params', combinedRegressor.random_params(r.__name__))
            estimators.append(r(**getattr(random, r.__name__ + '_params')))
        weight_sum = sum(random.weights.values())
        for r in REGRESSORS:
            random.weights[r] /= weight_sum
        random.regressor = VotingRegressor(
            estimators=[(REGRESSORS[i].__name__, e) for i, e in enumerate(estimators) if list(random.weights.values())[i] != 0],
            weights=[v for v in random.weights.values() if v != 0]
        )
        return random

    def random_params(kind):
        params = {}
        if kind == 'SVR':
            params['tol'] =  np.random.lognormal(mean=np.log(1e-3))
            params['C'] = np.random.lognormal(mean=np.log(1))
            params['epsilon'] = np.random.lognormal(mean=np.log(1e-1))
        if kind == 'KNeighborsRegressor':
            params['n_neighbors'] = 1 + np.random.binomial(40, 0.1)
            params['leaf_size'] = 10 + np.random.binomial(80, 0.25)
            params['p'] = 1 + np.random.binomial(5, 0.2)
        if kind ==  'MLPRegressor':
            length = 1 + np.random.binomial(10, 0.1)
            params['hidden_layer_sizes'] = tuple(10 + np.random.binomial(900, 0.1) for i in range(length))
            params['alpha'] = np.random.lognormal(mean=np.log(1e-4))
            params['epsilon'] = np.random.lognormal(mean=np.log(0.9))
            params['max_iter'] = 100 + np.random.binomial(900, 0.3)
        if kind == 'RandomForestRegressor':
            params['n_estimators'] = 10 + np.random.binomial(900, 0.1)
            params['min_samples_split'] = 2 + np.random.binomial(50, 0.01)
        return params

    def neighbor_params(kind, old_params):
        params = {}
        if kind == 'SVR':
            params['tol'] =  np.random.lognormal(mean=np.log(old_params['tol']), sigma=0.2)
            params['C'] = np.random.lognormal(mean=np.log(old_params['C']), sigma=0.2)
            params['epsilon'] = np.random.lognormal(mean=np.log(old_params['epsilon']), sigma=0.2)
        if kind == 'KNeighborsRegressor':
            params['n_neighbors'] = max(1, old_params['n_neighbors'] + np.random.choice([-1, 0, 1]))
            params['leaf_size'] = max(1, old_params['leaf_size'] + np.random.choice([-1, 1])*(np.random.randint(1, 11)))
            params['p'] = max(1, old_params['p'] + np.random.choice([-1, 0, 1]))
        if kind ==  'MLPRegressor':
            length_mod = np.random.choice([-1, 0, 1]) if len(old_params['hidden_layer_sizes']) > 1 else np.random.choice([0, 1])
            length = max(1, len(old_params['hidden_layer_sizes']) + length_mod)
            if length_mod == 0:    
                params['hidden_layer_sizes'] = tuple(max(10, i + np.random.choice([-1, 1])*(np.random.randint(1, 11))) for i in old_params['hidden_layer_sizes'])
            if length_mod == 1:
                n = np.random.randint(0, len(old_params['hidden_layer_sizes']) + 1)
                params['hidden_layer_sizes'] = old_params['hidden_layer_sizes'][0:n] + (10 + np.random.binomial(900, 0.1), ) + old_params['hidden_layer_sizes'][n:len(old_params['hidden_layer_sizes'])]
            if length_mod == -1:
                n = np.random.randint(0, len(old_params['hidden_layer_sizes']))
                params['hidden_layer_sizes'] = old_params['hidden_layer_sizes'][0:n] + old_params['hidden_layer_sizes'][n+1:len(old_params['hidden_layer_sizes'])]
            params['alpha'] = np.random.lognormal(mean=np.log(old_params['alpha']), sigma=0.2)
            params['epsilon'] = np.random.lognormal(mean=np.log(old_params['epsilon']), sigma=0.2)
            params['max_iter'] = max(100, old_params['max_iter'] + np.random.choice([-1, 1])*(np.random.randint(1, 100)))
        if kind == 'RandomForestRegressor':
            params['n_estimators'] = max(1, old_params['n_estimators'] + np.random.choice([-1, 1])*(np.random.randint(1, 11)))
            params['min_samples_split'] = max(2, old_params['min_samples_split'] + np.random.choice([-1, 0, 1]))
        return params

    def neighbor(self):
        neighbor = combinedRegressor(training_time_penalty=self.training_time_penalty)
        neighbor.weights = dict.copy(self.weights)
        for r in REGRESSORS:
            setattr(neighbor, r.__name__ + '_params', dict.copy(getattr(self, r.__name__ + '_params')))
        if np.random.choice(2):
            r = np.random.choice(REGRESSORS)
            if neighbor.weights[r] == 0:
                neighbor.weights[r] = 0.1
            if not all([v == 0 for k, v in neighbor.weights.items() if k != r]):
                neighbor.weights[r] = max(0, neighbor.weights[r] + np.random.normal(scale=0.2))
            weight_sum = sum(neighbor.weights.values())
            for r in REGRESSORS:
                neighbor.weights[r] /= weight_sum
        else:
            r = np.random.choice(list(filter(lambda r: neighbor.weights[r] != 0, REGRESSORS)))
            setattr(neighbor, r.__name__ + '_params', combinedRegressor.neighbor_params(r.__name__, getattr(neighbor, r.__name__ + '_params')))
        estimators = []
        for r in REGRESSORS:
            estimators.append(r(**getattr(neighbor, r.__name__ + '_params')))
        neighbor.regressor = VotingRegressor(
            estimators=[(REGRESSORS[i].__name__, e) for i, e in enumerate(estimators) if list(neighbor.weights.values())[i] != 0], 
            weights=[v for v in neighbor.weights.values() if v != 0]
        )
        return neighbor

    @ignore_warnings(category=ConvergenceWarning)
    def sa_score(self, X, y):
        start = time()
        score = np.average(cross_val_score(self.regressor, X, y, cv=3))
        return  score - (time() - start) * self.training_time_penalty
