'''Perfoms the optimization by utilizing the PyMoo algorithm component'''
from chweimo.generate.problem import get_problem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize


class Optimizer():
    def __init__(self, X, Y, model, seed=192):
        self._X = X
        self._Y = Y
        self._model = model
        self._seed = seed

    def optimize_instance(self,
                          sample,
                          change_class,
                          plausible,
                          method,
                          opt_param):
        """
        Method for beginning optimization cycle

        Parameters
        ----------
        sample : np.ndarray
            DESCRIPTION.
        change_class : int32
            DESCRIPTION.
        plausible : bool
            DESCRIPTION.
        method : str
            DESCRIPTION.
        **opt_param : {}
            DESCRIPTION.

        Returns
        -------
        res : np.ndarray
            DESCRIPTION.

        """
        
        if method == 'NSGA2':
            opt_param_ = {'termination': 100,
                      'pop_size': 40,
                      'n_offsprings': 20,
                      'sampling': 'real_random',
                      'crossover': ['real_sbx', 0.9, 15],
                      'mutation': ['real_pm', 20]}
            
            for key in opt_param.keys():
                try:
                    opt_param_[key] = opt_param[key]
                except KeyError as e:
                    print(e)
                    print(f"Invalid opt_param for {key}, using default value")

            termination = get_termination("n_gen", opt_param_['termination'])

            algorithm = NSGA2(pop_size=opt_param_['pop_size'],
                              n_offsprings=opt_param_['n_offsprings'],
                              sampling=get_sampling(opt_param_['sampling']),
                              crossover=get_crossover(opt_param_['crossover'][0],
                                                      prob=opt_param_['crossover'][1],
                                                      eta=opt_param_['crossover'][2]),
                              mutation=get_mutation(opt_param_['mutation'][0],
                                                    eta=opt_param_['mutation'][1]),
                              eliminate_duplicates=True)

            problem = get_problem(self._X, self._Y, self._model, sample,
                                  change_class, plausible)

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
