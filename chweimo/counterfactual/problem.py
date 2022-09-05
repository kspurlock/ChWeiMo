import numpy as np

from pymoo.core.problem import ElementwiseProblem

from chweimo.counterfactual.density_estimator import GMMDensity


def bound_collector(X):
    """Compute boundary of each m features in X

    Args:
        X (np.ndarray or list): n x m matrix

    Returns:
        xl_bounds (numpy.ndarray): m length array with minimum value for each feature
        xu_bounds (numpy.ndarray): m length array with maximum value for each feature
    """

    assert X.shape[0] > X.shape[1], "X must have more instances than features (may need to reshape)."

    # np.transpose offers support for various types (list, ndarrary, etc.)
    xl_bounds = np.array(
        [np.min(i) for _, i in enumerate(np.transpose(X))],
        dtype="float64")
    xu_bounds = np.array(
        [np.max(i) for _, i in enumerate(np.transpose(X))],
        dtype="float64")

    return (xl_bounds, xu_bounds)


def find_MAD(X):
    """Compute the Median Absolute Deviation (MAD) for each feature.

    Args:
        X (np.ndarray or list): n x m matrix

    Returns:
        MAD (np.ndarray): vector of m deviations from the median corrosponding to m features
    """

    MAD = [0 for _ in range(X.shape[1])]
    for j in range(X.shape[1]):
        mad_sample_wise = []
        avg = np.median(X[:, [j]])
        for i in range(X.shape[0]):
            val = abs(X-avg)
            mad_sample_wise.append(val)

        med = np.median(mad_sample_wise)

        if med == 0:
            med = 1

        MAD[j] = med
    
    return MAD


def L1_objective(x_prime, x_orig, MAD=False):
    """Objective described for minimization of the distance between x' and x.
        
    Args:
        x_prime (iterable): Potential solution (counterfactual) from the optimization
        x_orig (iterable): Original instance being explained
        MAD (iterable, optional): The median absolute deviations for each feature. Defaults to False.

    Returns:
        float: The cost for this objective
    """

    penalty = 0

    for i in range(x_prime.shape[0]):
        penalty += abs(x_orig[i] - x_prime[i])

        if isinstance(MAD, np.ndarray): # If this fails then MAD is set to False
            penalty = penalty/MAD[i]

    return -1/(penalty)


def pred_objective(x_prime, x_orig_proba, pred_proba_func, change_class):
    """Second objective, finds solutions with maximized prediction for yC
        between x' and x.

    Args:
        x_prime (iterable): Potential solution (counterfactual) from the optimization
        x_orig (iterable): Original instance being explained
        pred_proba_func (func): Function of learned model that supports class probabilities (Sklearn style)
        change_class (int): Desired class to change x_orig to

    Returns:
        float: The cost for this objective
    """
    
    x_prime_proba = pred_proba_func(x_prime.reshape(1, -1)
                                  ).reshape(-1)[change_class]

    """
    x_orig_proba = pred_proba_func(x_orig.reshape(1, -1)
                                 ).reshape(-1)[change_class]
    """
    
    cost = -1 * abs(x_prime_proba - x_orig_proba)

    return cost


def min_prob_constraint(x_prime, pred_proba_func, change_class):
    """Constraint to ensure that h(x') classified as yC based on probability.

    Args:
        x_prime (iterable): Potential solution (counterfactual) from the optimization
        pred_proba_func (func): Function of learned model that supports class probabilities (Sklearn style)
        change_class (int): Desired class to change x_orig to

    Returns:
        float: difference between desired probability and actual probability for class yC 
    """

    h_x = pred_proba_func(x_prime.reshape(
        1, -1)).reshape(-1,)[change_class]
    desired = .501

    return desired - h_x  # desired - h(x) <= 0 from desired <= h(x)


def plaus_constraint(x_prime, gmm, change_class):
    """Constraint to ensure that the potential counterfactual (x_prime)
        is feasible in the domain of the data.

    Args:
        x_prime (iterable): Potential solution (counterfactual) from the optimization
        gmm (object): Class-dependent density estimators
        change_class (int): Desired class to change x_orig to 

    Returns:
        _type_: _description_
    """
    density = gmm.find_new_density(x_prime)

    max_med_density = gmm.med_densities[change_class]
    return density - max_med_density


class MyProblem(ElementwiseProblem):
    """PyMoo problem formulation

    Args:
        ElementwiseProblem (_type_): Inherits from PyMoo ElementwiseProblem
    """

    def __init__(self, x_orig, x_orig_proba, pred_proba_func, change_class, xl, xu,
                 MAD, GMM, plausible):
        super().__init__(n_var=x_orig.reshape(1, -1).shape[1],
                         n_obj=2,
                         n_constr=2 if plausible else 1,
                         xl=xl,
                         xu=xu)

        self.x_orig = x_orig
        self.x_orig_proba = x_orig_proba
        self.pred_proba_func = pred_proba_func
        self.MAD = MAD
        self.GMM = GMM
        self.change_class = change_class
        self.plausible = plausible

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = L1_objective(x, self.x_orig, self.MAD)
        f2 = pred_objective(x, self.x_orig_proba, self.pred_proba_func, self.change_class)

        g1 = min_prob_constraint(x, self.pred_proba_func, self.change_class)

        out["F"] = [f1, f2]

        out["G"] = None
        if self.plausible:
            g2 = plaus_constraint(x, self.GMM, self.change_class)
            out["G"] = [g1, g2]

        else:
            out["G"] = [g1]


def get_problem(X, Y, pred_proba_func, x_orig, change_class,
                plausible, use_mad):
    '''Performs bound collection,
        MAD generation,
        and GMM from training dataset'''

    xl, xu = bound_collector(X)
    
    # Saves time having to recompute this every time in pred_objective()
    x_orig_proba = pred_proba_func(x_orig.reshape(1, -1)
                                 ).reshape(-1)[change_class]

    if use_mad:
        MAD = find_MAD(X)
    else:
        MAD = False

    if plausible:
        GMM = GMMDensity()
        GMM.fit_class_gmm(X, Y)
        GMM.estimate_class_density(X, Y)

    else:
        GMM = False

    return MyProblem(x_orig, x_orig_proba, pred_proba_func, change_class, xl, xu,
                     MAD, GMM, plausible)
