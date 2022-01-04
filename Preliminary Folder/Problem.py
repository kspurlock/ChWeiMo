import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize

from DensityEstimator import GMMDensity


def bound_collector(X):
    xl_bounds = []
    xu_bounds = []

    try:
        for i in range(X.shape[1]):
            mini = np.min(X[:, [i]])
            maxi = np.max(X[:, [i]])

            xl_bounds.append(float(mini))
            xu_bounds.append(float(maxi))

    except IndexError:
        print('Bound collector: X needs reshaped to (1,-1)')

    return np.array(xl_bounds), np.array(xu_bounds)


def find_MAD(X):
    '''Method for scaling each feature based on MAD
    
        Parameters:
            -X is the feature vector of the training dataset'''

    MAD = []
    for j in range(X.shape[1]):
        mad_sample_wise = []
        avg = np.median(X[:, [j]])
        for i in range(X.shape[0]):
            val = abs(X-avg)
            mad_sample_wise.append(val)

        med = np.median(mad_sample_wise)
        
        if med == 0:
            med = 1
        
        MAD.append(med)

    return MAD


def L1_objective(x_prime, x_orig, MAD, use_mad=False):
    '''Goal is to have x_prime such that the difference between x_prime and x
        is as small as possible while allowing for desired model prediction
        
        Parameters:
            -x_prime: the new sample provided by the optimization pop
            -x_orig: original sample passed to explainer
    
    
        Manhattan distance between x' and x'''

    penalty = 0

    for i in range(x_prime.shape[0]):
        penalty += abs(x_orig[i] - x_prime[i])

        if use_mad:
            penalty = penalty/MAD[i]

    return -1/(penalty)


def pred_objective(x_prime, x_orig, model, change_class):
    '''Second objective, finds solutions with maximized prediction for yC
            between x' and x.
            
            Parameters:
                -x: solution pop of optimization
                -x_orig: original sample passed to explainer
            
        '''

    x_prime_proba = model.predict_proba(x_prime.reshape(1, -1)
                                        ).reshape(-1)[change_class]

    x_orig_proba = model.predict_proba(x_orig.reshape(1, -1)
                                       ).reshape(-1)[change_class]

    val = -1 * abs(x_prime_proba - x_orig_proba)

    return val


def min_prob(x_prime, model, change_class):
    '''Goal here is to assure that h(x') classified as yC based on probability
    
    
        Parameters:
            -x is the solutions of the optimization pop
            -change_class: changing x_prime to this class
    '''

    h_x = model.predict_proba(x_prime.reshape(
        1, -1)).reshape(-1,)[change_class]
    desired = .51

    return desired - h_x  # desired - h(x) <= 0 from desired <= h(x)


def plaus_constraint(x, gmm, change_class):
    density = gmm.find_new_density(x)

    max_med_density = gmm.med_densities[change_class]
    return density - max_med_density


class MyProblem(ElementwiseProblem):

    def __init__(self, x_orig, model, xl, xu, MAD, GMM, change_class, n_constr):
        super().__init__(n_var=x_orig.reshape(1, -1).shape[1],
                         n_obj=2,
                         n_constr=n_constr,
                         xl=xl,
                         xu=xu)

        self.x_orig = x_orig
        self.model = model
        self.MAD = MAD
        self.GMM = GMM
        self.change_class = change_class
        self.n_constr = n_constr

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = L1_objective(x, self.x_orig, self.MAD, use_mad=True)
        f2 = pred_objective(x, self.x_orig, self.model, self.change_class)

        g1 = min_prob(x, self.model, self.change_class)

        out["F"] = [f1, f2]

        out["G"] = None
        if self.n_constr == 2:
            g2 = plaus_constraint(x, self.GMM, self.change_class)
            out["G"] = [g1, g2]

        else:
            out["G"] = [g1]


def get_problem(X, Y, model, x_orig, change_class, n_constr):
    '''Performs bound collection,
        MAD generation,
        and GMM from training dataset'''

    xl, xu = bound_collector(X)
    MAD = find_MAD(X)

    GMM = None
    if n_constr == 2:
        GMM = GMMDensity()
        GMM.fit_class_gmm(X, Y)
        GMM.estimate_class_density(X, Y)

    else:
        GMM = None

    prob = MyProblem(x_orig, model, xl, xu, MAD, GMM, change_class, n_constr)

    return prob
