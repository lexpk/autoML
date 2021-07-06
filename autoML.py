from numpy import exp
from combinedRegressor import CombinedRegressor
from defaultConfiguration import default_configuration
from time import time
from random import random

class AutoML():
    def __init__(self, configuration = default_configuration, fitting_time = 60, training_time_penalty = 0.001, verbose=False):
        self.configuration = configuration
        self.fitting_time = fitting_time
        self.training_time_penalty = training_time_penalty
        self.verbose = verbose
        self.regressor = None

    def fit(self, X, y): 
        start = time()
        T = 1
        current = CombinedRegressor(
            configuration = self.configuration,
            training_time_penalty=self.training_time_penalty
        )
        current_score = current.score(X, y)
        best = current
        i = 0
        loop_start = time()
        loop_fitting_time = (self.fitting_time - (time() - start))*0.8
        while self.fitting_time - (time() - start) > 0:
            if self.verbose and i % 5 == 0:
                print(
                    f"Iteration {i},\t{self.fitting_time - (time() - start):.1f}s remaining\n"
                    f"Current: {current.max_score:.4f}\tBest: {best.max_score:.4f}"
                )
            neighbor = current.neighbor(temperature=T)
            neighbor_score = neighbor.score(X, y)
            if neighbor_score > current_score:
                current = neighbor
                current_score = neighbor_score
            else:
                r = random()
                if r < exp((neighbor_score - current_score)/T):
                    current = neighbor
                    current_score = neighbor_score
            if current.max_score > best.max_score:
                best = current
            i+=1
            avg_time = (time() - loop_start)/i
            alpha = exp(-9*avg_time/loop_fitting_time)
            T = alpha**i

        self.regressor = best.best
        self.regressor.fit(X, y)
    
    def predict(self, X):
        return self.regressor.predict(X)

    def score(self, X, y):
        return self.regressor.score(X, y)

    def get_params(self, deep=False):     
        return {
            'configuration' : self.configuration,
            'fitting_time' : self.fitting_time,
            'training_time_penalty' : self.training_time_penalty,
            'verbose' : self.verbose
        }

    def set_params(self, params):
        self.configuration = params['configuration']
        self.fitting_time = params['fitting_time']
        self.training_time_penalty = params['training_time_penalty']
        self.verbose = params['verbose']
