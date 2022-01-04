'''Perfoms the optimization by utilizing the PyMoo algorithm component'''
from Problem import get_problem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize


class Explainer():
    def __init__(self, X, Y, model, seed=192):
        self._X = X
        self._Y = Y
        self._model = model
        self._seed = seed

    def explain_instance(self, sample, change_class, plausible, method, opt_params=None):
        '''
        Parameters:
            -method: string
            
            -method_params: {}, contain information for the optimization method
                    {termination: int, 
                     pop_size: int, 
                     n_offspring: int, 
                     sampling: string,
                     crossover: array-like,
                     mutation: array-like
                     }
            
            -x_orig: array-like
            
            -change_class: int
            
            -plausible: bool
        '''

        n_constr = 0

        if plausible:
            n_constr = 2
        else:
            n_constr = 1

        if method == 'NSGA2':
            params = None
            if method_params == None:
                params = {'termination': 100,
                          'pop_size': 40,
                          'n_offsprings': 20,
                          'sampling': 'real_random',
                          'crossover': ['real_sbx', 0.9, 15],
                          'mutation': ['real_pm', 20]}

            else:
                params = method_params

            termination = get_termination("n_gen", params['termination'])

            algorithm = NSGA2(pop_size=params['pop_size'],
                              n_offsprings=params['n_offsprings'],
                              sampling=get_sampling(params['sampling']),
                              crossover=get_crossover(params['crossover'][0],
                                                      prob=params['crossover'][1],
                                                      eta=params['crossover'][2]),
                              mutation=get_mutation(params['mutation'][0],
                                                    eta=params['mutation'][1]),
                              eliminate_duplicates=True)

            problem = get_problem(self._X, self._Y, self._model, sample,
                                  change_class, n_constr)

            res = minimize(problem,
                           algorithm,
                           termination,
                           seed=self._seed,
                           save_history=True,
                           verbose=True)

            return res

        else:
            NotImplementedError(
                'NSGA2 is currently the only utilized algorithm for ChWeiMO.')
