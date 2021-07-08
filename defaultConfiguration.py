from numpy.random import default_rng
from numpy import log
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

default_configuration_rng = default_rng()
default_configuration = [
    {   
        'name' : 'AdaBoostRegressor',
        'regressor' : AdaBoostRegressor, 
        'params' : {
            'n_estimators' : {
                'hidden' : False,
                'kind' : 'int',
                'generator' : lambda : default_configuration_rng.integers(low=50, high=500),
                'min' : 50,
                'max' :  500
            },
            'learning_rate' : {
                'hidden' : False,
                'kind' : 'float',
                'generator' : lambda : default_configuration_rng.lognormal(mean=log(0.1), sigma=0.2),
                'min' : 1e-2,
                'max' :  2
            },
            'loss' : {
                'hidden' : False,
                'kind' : 'enum',
                'generator' : lambda : default_configuration_rng.choice(["linear", "square", "exponential"]),
            },
        }
    },
    {
        'name' : 'SVR',
        'regressor' : SVR, 
        'params' : {
            'tol' : {
                'hidden' : False,
                'kind' : 'float',
                'generator' : lambda : default_configuration_rng.lognormal(mean=log(1e-3)),
                'min' : 1e-10,
                'max' :  1e-2
            },
            'C' : {
                'hidden' : False,
                'kind' : 'float',
                'generator' : lambda : default_configuration_rng.lognormal(mean=log(1)),
                'min' : 1e-5,
                'max' :  10
            },
            'epsilon' : {
                'hidden' : False,
                'kind' : 'float',
                'generator' : lambda : default_configuration_rng.lognormal(mean=log(1e-1)),
                'min' : 1e-10,
                'max' :  1
            },
        }
    },
    {
        'name' : 'KNeighborsRegressor',
        'regressor' : KNeighborsRegressor, 
        'params' : {
            'n_neighbors' : {
                'hidden' : False,
                'kind' : 'int',
                'generator' : lambda : 1 + default_configuration_rng.binomial(40, 0.1),
                'min' : 1,
                'max' : 8,
            },
            'algorithm' : {
                'hidden' : False,
                'kind' : 'enum',
                'generator' : lambda : default_configuration_rng.choice(['ball_tree', 'kd_tree'])
            },
            'leaf_size' : {
                'hidden' : False,
                'kind' : 'int',
                'generator' : lambda : 10 + default_configuration_rng.binomial(80, 0.25),
                'min' : 10,
                'max' :  90
            },
        }
    },
    {
        'name' : 'RandomForestRegressor',
        'regressor' : RandomForestRegressor, 
        'params' : {
            'n_estimators' : {
                'hidden' : False,
                'kind' : 'const',
                'generator' : lambda : 512,
            },
            'criterion' : {
                'hidden' : False,
                'kind' : 'enum',
                'generator' : lambda : default_configuration_rng.choice(['mse', 'mae']),
            },
            'min_impurity_decrease' : {
                'hidden' : False,
                'kind' : 'float',
                'generator' : lambda : default_configuration_rng.lognormal(mean=log(1), sigma=0.25) - 1,
                'min' : 0,
                'max' : 1
            },
            'min_samples_split' : {
                'hidden' : False,
                'kind' : 'int',
                'generator' : lambda : default_configuration_rng.lognormal(mean=log(2), sigma=2),
                'min' : 2,
                'max' : 10
            },
            'min_samples_leaf' : {
                'hidden' : False,
                'kind' : 'int',
                'generator' : lambda : default_configuration_rng.lognormal(mean=log(1), sigma=2),
                'min' : 1,
                'max' : 20
            },
            'max_features' : {
                'hidden' : False,
                'kind' : 'float',
                'generator' : lambda : 2 - default_configuration_rng.lognormal(mean=log(1), sigma=0.25),
                'min' : 0.1,
                'max' : 1
            }             
        }
    },
    {
        'name' : 'MLPRegressor',
        'regressor' : MLPRegressor, 
        'params' : {
            'hidden_layer_sizes' : {
                'hidden' : False,
                'kind' : 'computed',
                'arguments' : ['hidden_layer_depth', 'num_nodes_layer_0', 'num_nodes_layer_1', 'num_nodes_layer_2'],
                'generator' : lambda p : (p[1], p[2], p[3])[0:p[0]]
            },
            'hidden_layer_depth' : {
                'hidden' : True,
                'kind' : 'int',
                'generator' : lambda : default_configuration_rng.integers(1, 3),
                'min' : 1,
                'max' : 3
            },
            'num_nodes_layer_0' : {
                'hidden' : True,
                'kind' : 'int',
                'generator' : lambda : 16 + 16*default_configuration_rng.lognormal(mean=log(1)),
                'min' : 16,
                'max' : 128
            },
            'num_nodes_layer_1' : {
                'hidden' : True,
                'kind' : 'int',
                'generator' : lambda : 16 + 16*default_configuration_rng.lognormal(mean=log(1)),
                'min' : 16,
                'max' : 128
            },
            'num_nodes_layer_2' : {
                'hidden' : True,
                'kind' : 'int',
                'generator' : lambda : 16 + 16*default_configuration_rng.lognormal(mean=log(1)),
                'min' : 16,
                'max' : 128
            },
            'max_iter' : {
                'hidden' : False,
                'kind' : 'const',
                'generator' : lambda : 100000,
            },
            'activation' : {
                'hidden' : False,
                'kind' : 'enum',
                'generator' : lambda : default_configuration_rng.choice(['tanh', 'relu'])
            },
            'alpha' : {
                'hidden' : False,
                'kind' : 'float',
                'generator' : lambda : default_configuration_rng.lognormal(mean=1e-4), 
                'min' : 1e-7,
                'max' : 1e-1
            },
            'learning_rate_init' : {
                'hidden' : False,
                'kind' : 'float',
                'generator' : lambda : default_configuration_rng.lognormal(mean=1e-3),
                'min' : 1e-4,
                'max' : 0.5 
            },
        }
    },
]