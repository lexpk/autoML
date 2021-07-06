from numpy.random import default_rng
from numpy import inf, exp, log, sqrt, average
from sklearn.model_selection import cross_val_score

class CombinedRegressor():
    def __init__(self, configuration, training_time_penalty=0.001, verbose=False, initialize_random=True):
        self.configuration = configuration
        self.training_time_penalty = training_time_penalty
        self.verbose = verbose
        self.changed = []
        self.regressor = None
        self.max_score = -inf
        if initialize_random:
            for r in self.configuration:
                setattr(self, r['regressor'].__name__ + '_params', dict())
                for p in r['params']:
                    getattr(self, r['regressor'].__name__ + '_params')[p] = r['params'][p]['generator']() if r['params'][p]['kind'] == 'enum' else (
                        min(
                            r['params'][p]['max'],
                            max(
                                r['params'][p]['min'],
                                r['params'][p]['generator']() 
                            )
                        )
                    )
                setattr(self, r['regressor'].__name__,
                    r['regressor'](**getattr(self, r['regressor'].__name__ + '_params')
                ))
                self.changed.append(r['regressor'].__name__)

    def neighbor(self, temperature = 1):
        neighbor = CombinedRegressor(
            self.configuration,
            training_time_penalty=self.training_time_penalty,
            verbose=self.verbose,
            initialize_random=False
        )
        rng = default_rng()
        bias = sqrt(temperature)
        log_bias = -log(bias)
        weights = [exp((getattr(self, r['regressor'].__name__ + '_score') - self.max_score)*log_bias) for r in self.configuration]
        weights = [w/sum(weights) for w in weights]
        chosen = rng.choice(self.configuration, p = weights)
        for r in self.configuration:
             if r != chosen:
                setattr(neighbor, r['regressor'].__name__, getattr(self, r['regressor'].__name__))
                setattr(neighbor, r['regressor'].__name__ + '_params', getattr(self, r['regressor'].__name__ + '_params'))
                setattr(neighbor, r['regressor'].__name__ + '_score', getattr(self, r['regressor'].__name__ + '_score'))
        setattr(neighbor, chosen['regressor'].__name__ + '_params', dict())
        for p in chosen['params']:
            param = chosen['params'][p]
            if param['kind'] == 'float' or param['kind'] == 'int':
                temp = max(param['min'], min(param['max'],
                    rng.normal(
                        loc = (
                            bias * param['generator']() +
                            (1 - bias) * getattr(self, chosen['regressor'].__name__ + '_params')[p]
                        ),
                        scale = (param['max'] - param['min'])/max(1, 25*log_bias)
                    )
                ))
                getattr(neighbor, chosen['regressor'].__name__ + '_params')[p] = int(temp + 0.5) if param['kind'] == 'int' else temp
            if param['kind'] == 'enum':
                if rng.choice(2, p = [1 - bias, bias]):
                    getattr(neighbor, chosen['regressor'].__name__ + '_params')[p] = getattr(self, chosen['regressor'].__name__ + '_params')[p]
                else:
                    getattr(neighbor, chosen['regressor'].__name__ + '_params')[p] = param['generator']()
        setattr(neighbor, chosen['regressor'].__name__,
                chosen['regressor'](**getattr(self, chosen['regressor'].__name__ + '_params')
        ))
        neighbor.changed.append(chosen['regressor'].__name__)
        neighbor.max_score = self.max_score
        return neighbor


    def score(self, X, y):
        while self.changed:
            r = self.changed.pop()
            regressor = getattr(self, r)
            score = average(cross_val_score(regressor, X, y, cv=5, n_jobs=-1))
            setattr(self, r + '_score', score)
        self.max_score = max([
            getattr(self, r['regressor'].__name__ + '_score')
            for r in self.configuration        
        ])
        for r in self.configuration:
            if getattr(self, r['regressor'].__name__ + '_score') == self.max_score:
                self.best = getattr(self, r['regressor'].__name__)
        self.computed_score = sum([
            min(
                getattr(self, r['regressor'].__name__ + '_score'),
                getattr(self, r['regressor'].__name__ + '_score')**2
            ) for r in self.configuration
        ])
        return self.computed_score
