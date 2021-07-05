from math import exp, log
from combinedRegressor import combinedRegressor
from time import time
from random import random
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class autoML():
    def __init__(self, fitting_time = 60, training_time_penalty = 0.001, verbose=False):
        self.fitting_time = fitting_time
        self.training_time_penalty = training_time_penalty
        self.verbose = verbose
        self.regressor = None

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X, y):
        start = time()
        T = 1/log(2)
        current = combinedRegressor.random(training_time_penalty=self.training_time_penalty)
        current_score = current.sa_score(X, y)
        best = current
        best_score = current_score
        if self.verbose:
            i = 0
        while self.fitting_time - (time() - start) > 0:
            if self.verbose:
                i+= 1
                print(
                    f"Iteration {i},\t{self.fitting_time - (time() - start):.1f}s remaining\n"
                    f"Current: {current_score:.4f}\tBest: {best_score:.4f}"
                )
            neighbor = current.neighbor()
            neighbor_score = neighbor.sa_score(X, y)
            if neighbor_score > current_score:
                current = neighbor
                current_score = neighbor_score
            else:
                r = random()
                if r < 0 if T == 0 else exp((neighbor_score - current_score)/T):
                    current = neighbor
                    current_score = neighbor_score
            if current_score > best_score:
                best = current
                best_score = current_score
            T = 1/(log(2) * 10**(4*(time() - start)/self.fitting_time))

        self.regressor = best.regressor
        self.regressor.fit(X, y)
    
    @ignore_warnings(category=ConvergenceWarning)
    def predict(self, X):
        return self.regressor.predict(X)

    @ignore_warnings(category=ConvergenceWarning)
    def score(self, X, y):
        return self.regressor.score(X, y)

    def get_params(self, deep=False):     
        return {'fitting_time' : self.fitting_time, 'training_time_penalty' : self.training_time_penalty, 'verbose' : self.verbose}

    def set_params(self, params):
        self.fitting_time = params['fitting_time']
        self.training_time_penalty = params['training_time_penalty']
        self.verbose = params['verbose']
