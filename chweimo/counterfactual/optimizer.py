"""Perfoms the optimization by utilizing the PyMoo algorithm component"""
from chweimo.counterfactual.problem import get_problem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize

import numpy as np
import pandas as pd

class Optimizer():
    def __init__(self, X, Y, pred_proba, seed=None):
        self.X_ = X
        self.Y_ = Y
        self.pred_proba_ = pred_proba
        self.seed_ = seed
        
        # Determined after calling generate_cf
        self.res_ = None
        self.sample_ = None
        self.change_class_ = None
        self.plausible_ = None

    def generate_cf(self,
                    sample,
                    change_class,
                    plausible=True,
                    use_MAD=True,
                    method="NSGA2",
                    **opt_param):
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
        # Including this here so it doesn't have to be passed again
        # when calling explain (wouldn't make sense to use cf for 
        # sample other than what they were generated for)
        self.sample_ = sample
        self.change_class_ = change_class
        self.plausible_ = plausible
        
        if method == "NSGA2":
            opt_param_ = {"termination": 100,
                      "pop_size": 40,
                      "n_offsprings": 20,
                      "sampling": "int_random",
                      "crossover": ["int_sbx", 0.9, 15],
                      "mutation": ["int_pm", 20],
                      "verbose": True}
            
            for key in opt_param.keys():
                try:
                    opt_param_[key] = opt_param[key]
                except KeyError as e:
                    print(e)
                    print(f"{key} is not a valid keyword argument.")

            termination = get_termination("n_gen", opt_param_["termination"])

            algorithm = NSGA2(pop_size=opt_param_["pop_size"],
                              n_offsprings=opt_param_["n_offsprings"],
                              sampling=get_sampling(opt_param_["sampling"]),
                              crossover=get_crossover(opt_param_["crossover"][0],
                                                      prob=opt_param_["crossover"][1],
                                                      eta=opt_param_["crossover"][2]),
                              mutation=get_mutation(opt_param_["mutation"][0],
                                                    eta=opt_param_["mutation"][1]),
                              eliminate_duplicates=True)

            problem = get_problem(self.X_, self.Y_, self.pred_proba_, sample,
                                  change_class, plausible, use_MAD)

            res = minimize(problem,
                           algorithm,
                           termination,
                           seed=self.seed_,
                           save_history=True,
                           verbose=opt_param_["verbose"])

            self.res_ = res

        else:
            raise NotImplementedError(
                "NSGA2 is currently the only optimizer combatible with ChWeiMO.")
            
    def show_cf(self, n_solutions, cols):
        xs = pd.DataFrame(self.res_.history[-1].pop.get("X")-self.sample_, columns=cols).head(5)
        fs = (pd.DataFrame(self.res_.history[-1].pop.get("F"), columns=["Distance", "Pred Δ"])*-1).head(5)


        non_dom_xs = pd.DataFrame(self.res_.X-self.sample_, columns=cols)
        non_dom_fs = pd.DataFrame(self.res_.F, columns=["Distance", "Pred Δ"])*-1
        non_dom = pd.concat((non_dom_xs, non_dom_fs), axis=1)

        fs["Distance"] = 1/fs["Distance"]
        non_dom["Distance"] = 1/non_dom["Distance"]

        combined = pd.concat((xs, fs), axis=1)

        best_obj1 = combined.sort_values(by="Distance").head(n_solutions)
        best_obj1.index = ["Obj1" for _ in range(best_obj1.shape[0])]

        best_obj2 = combined.sort_values(by="Pred Δ", ascending=False).head(n_solutions)
        best_obj2.index=["Obj2" for _ in range(best_obj2.shape[0])]


        combined = pd.concat((best_obj1, best_obj2)).reset_index()
        combined.rename({"index":"Best of"}, axis=1, inplace=True)

        non_dom_col = np.full((combined.shape[0], 1), "X")

        for i, v in combined.drop("Best of", axis=1).iterrows():
            for _, nv in non_dom.iterrows():
                if nv.equals(v):
                    non_dom_col[i] = "✓"
                    
        combined["Non-Dom?"] = non_dom_col

        final = combined.style.pipe(make_pretty)
        
        #display(pd.DataFrame(x_orig, ))
            
        return final      
        
def make_pretty(s):
    s.set_caption("Counterfactuals")
    
    s.apply(
        lambda s: np.where(s["Best of"] == "Obj1", ["background-color:#D55E00"]*len(s), ["background-color:#0072B2"]*len(s)), axis=1
        )
    
    s.apply(
        lambda s: np.where(s== "✓", "background-color:#009E73", "background-color:#CC79A7"), axis=0, subset=["Non-Dom?"]
        )
    
    s.set_table_styles([
        {"selector": "th.col_heading.level0", "props": "font-size:0.8em; background-color:black;"},
        {"selector": "th:not(.index_name)", "props": "font-size:0.8em; background-color:black;"}
    ])
    
    s.format(
        precision=1, formatter={
            ("Distance"): "{:.1f}",
            ("Pred Δ"): "{:.2f}"}
        )
    
    return s