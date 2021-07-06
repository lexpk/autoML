from numpy.random import default_rng
from numpy import log
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

default_configuration_rng = default_rng()
default_configuration = [
    {
        'regressor' : SVR, 
        'params' : {
            'tol' : {
                'kind' : 'float',
                'generator' : lambda : default_configuration_rng.lognormal(mean=log(1e-3)),
                'min' : 1e-10,
                'max' :  1e-2
            },
            'C' : {
                'kind' : 'float',
                'generator' : lambda : default_configuration_rng.lognormal(mean=log(1)),
                'min' : 1e-5,
                'max' :  10
            },
            'epsilon' : {
                'kind' : 'float',
                'generator' : lambda : default_configuration_rng.lognormal(mean=log(1e-1)),
                'min' : 1e-10,
                'max' :  1
            },
        }
    },
    {
        'regressor' : KNeighborsRegressor, 
        'params' : {
            'n_neighbors' : {
                'kind' : 'int',
                'generator' : lambda : 1 + default_configuration_rng.binomial(40, 0.1),
                'min' : 1,
                'max' : 8,
            },
            'algorithm' : {
                'kind' : 'enum',
                'generator' : lambda : default_configuration_rng.choice(['ball_tree', 'kd_tree'])
            },
            'leaf_size' : {
                'kind' : 'int',
                'generator' : lambda : 10 + default_configuration_rng.binomial(80, 0.25),
                'min' : 10,
                'max' :  90
            },
        }
    },
    {
        'regressor' : RandomForestRegressor, 
        'params' : {
            'n_estimators' : {
                'kind' : 'int',
                'generator' : lambda : 10 + default_configuration_rng.binomial(900, 0.1),
                'min' : 10,
                'max' : 910
            },
            'criterion' : {
                'kind' : 'enum',
                'generator' : lambda : default_configuration_rng.choice(['mse', 'mae']),
            },
            'min_impurity_decrease' : {
                'kind' : 'float',
                'generator' : lambda : default_configuration_rng.lognormal(mean=log(1)) - 1,
                'min' : 0,
                'max' : 10
            },         
        }
    },
]