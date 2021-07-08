from numpy.random import default_rng
from numpy import inf, exp, log, sqrt, average
from sklearn.model_selection import cross_val_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class CombinedRegressor():
    def __init__(self, configuration, n_jobs = None, verbose=False, initialize_random=True):
        self.configuration = configuration
        self.n = len(self.configuration)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.regressors = []
        self.changed = []
        self.max_score = -inf
        if initialize_random:
            for i in range(self.n):
                self.regressors.append(dict())
                self.regressors[i]['params'] = dict()
                for p in self.configuration[i]['params']:
                    param = self.configuration[i]['params'][p]
                    if param['kind'] in ['float', 'int']:
                        temp = max(param['min'], min(param['max'], param['generator']()))
                        self.regressors[i]['params'][p] = int(temp + 0.5) if param['kind'] == 'int' else temp
                    if param['kind'] in ['enum', 'const']:
                        self.regressors[i]['params'][p] = param['generator']()
                for p in self.configuration[i]['params']:
                    param = self.configuration[i]['params'][p]
                    if param['kind'] in ['computed'] :
                        self.regressors[i]['params'][p] = param['generator']([self.regressors[i]['params'][x] for x in param['arguments']])
            self.regressors = [
                {
                    'params' : self.regressors[i]['params'],
                    'regressor' : r['regressor'](**{k : v for k, v in self.regressors[i]['params'].items() if not self.configuration[i]['params'][k]['hidden']}),
                    'score' : None
                } for i, r in enumerate(self.configuration)
            ]      
            self.changed = list(range(self.n))

    def neighbor(self, temperature = 1):
        neighbor = CombinedRegressor(
            self.configuration,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            initialize_random=False
        )
        neighbor.n = self.n
        rng = default_rng()
        bias = -log(sqrt(temperature))
        weights = [exp((r['score'] - self.best['score'])*(bias**4)) for r in self.regressors]
        weights = [w/sum(weights) for w in weights]
        chosen = rng.choice(neighbor.n, p = weights)
        neighbor.regressors = []
        for i in range(self.n):
            if i != chosen:
                neighbor.regressors.append(self.regressors[i])
            else:
                neighbor.regressors.append(dict())
                neighbor.regressors[i]['params'] = dict()
                for p in neighbor.configuration[i]['params']:
                    param = neighbor.configuration[i]['params'][p]
                    if param['kind'] in ['float', 'int']:
                        temp = max(param['min'], min(param['max'],
                            rng.normal(
                                loc = 1/(1+bias**2) * param['generator']() + (1 - 1/(1+bias**2)) * self.regressors[i]['params'][p],
                                scale = (param['max'] - param['min'])/max(1, 5*(bias**2))
                            )
                        ))
                        neighbor.regressors[i]['params'][p] = int(temp + 0.5) if param['kind'] == 'int' else temp
                    if param['kind'] in ['enum']:
                        neighbor.regressors[i]['params'][p] = self.regressors[i]['params'][p] if rng.choice(2, p = [1 - 1/(1+bias**2), 1/(1+bias**2)]) else param['generator']()
                    if param['kind'] in ['const']:
                        neighbor.regressors[i]['params'][p] = param['generator']()
                for p in neighbor.configuration[i]['params']:
                    param = neighbor.configuration[i]['params'][p] 
                    if param['kind'] in ['computed'] :
                        neighbor.regressors[i]['params'][p] = param['generator']([neighbor.regressors[i]['params'][x] for x in param['arguments']]) 
                neighbor.regressors[i]['regressor'] = make_pipeline(StandardScaler(), neighbor.configuration[i]['regressor'](
                    **{k : v for k, v in neighbor.regressors[i]['params'].items() if not neighbor.configuration[i]['params'][k]['hidden']}
                ))
                neighbor.regressors[i]['score'] = -inf
                neighbor.changed.append(i)
        return neighbor

    def score(self, X, y, verbose=False):
        while self.changed:
            i = self.changed.pop()
            self.regressors[i]['score'] = average(cross_val_score(
                self.regressors[i]['regressor'],
                X,
                y,
                cv=5,
                n_jobs=self.n_jobs,
            ))
            if verbose:
                print(
                    f"{self.regressors[i]['regressor']}"
                    f", score: {self.regressors[i]['score']:.4f}\n"
                )
        self.best = max(self.regressors, key=lambda x : x['score'])
        return sum([
            min(
                r['score'],
                r['score']**2
            ) for r in self.regressors
        ])
