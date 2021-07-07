from numpy import exp
from combinedRegressor import CombinedRegressor
from defaultConfiguration import default_configuration
from time import time
from random import random

class AutoML():
    def __init__(self, configuration = default_configuration, fitting_time = 60, n_jobs = None, verbose=False):
        self.configuration = configuration
        self.fitting_time = fitting_time
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.regressor = None

    def fit(self, X, y): 
        start = time()
        T = 1
        current = CombinedRegressor(
            configuration = self.configuration,
            n_jobs = self.n_jobs
        )
        current_score = current.score(X, y, verbose=self.verbose)
        best = current.best
        i = 0
        loop_start = time()
        loop_fitting_time = (self.fitting_time - (time() - start))*0.8
        while self.fitting_time - (time() - start) > 0:
            if self.verbose and i:
                print(
                    f"Iteration {i},\t{self.fitting_time - (time() - start):.1f}s remaining\n"
                    f"Current: {current.best['score']:.4f}\tBest: {best['score']:.4f}\n"
                )
            neighbor = current.neighbor(temperature=T)
            neighbor_score = neighbor.score(X, y, verbose=self.verbose)
            if neighbor_score > current_score:
                current = neighbor
                current_score = neighbor_score
            else:
                r = random()
                if r < exp((neighbor_score - current_score)/T):
                    current = neighbor
                    current_score = neighbor_score
            if current.best['score'] > best['score']:
                best = current.best
            i+=1
            avg_time = (time() - loop_start)/i
            alpha = exp(-10*avg_time/loop_fitting_time)
            T = alpha**i
        self.regressor = best['regressor'].fit(X, y)
        return self.regressor
    
    def predict(self, X):
        return self.regressor.predict(X)

    def score(self, X, y):
        return self.regressor.score(X, y)

    def get_params(self, deep=False):     
        return {
            'configuration' : self.configuration,
            'fitting_time' : self.fitting_time,
            'n_jobs' : self.n_jobs,
            'verbose' : self.verbose
        }

    def set_params(self, params):
        self.configuration = params['configuration']
        self.fitting_time = params['fitting_time']
        self.n_jobs = params['n_jobs']
        self.verbose = params['verbose']
