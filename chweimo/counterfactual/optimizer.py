"""Perfoms the optimization by utilizing the PyMoo algorithm component"""
from chweimo.counterfactual.problem import get_problem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize

import numpy as np
import pandas as pd

class Optimizer():
    def __init__(self, X, Y, pred_proba, seed=None, **kwargs):
        self.X_ = X
        self.Y_ = Y
        self.pred_proba_ = pred_proba # Function supporting class-wise probability (Sklearn style)
        self.seed_ = seed
        
        # Determined after calling generate_cf
        self.res_ = None
        self.sample_ = None
        self.sample_proba_ = None
        self.change_class_ = None
        self.plausible_ = None
        
        if "col_names" in kwargs:
            self.col_names_ = kwargs["col_names"]
        elif isinstance(X, pd.DataFrame):
            self.col_names_ = X.columns
        else:
            self.col_names_ = [str(i) for i in range(X.shape[1])]


    def generate_cf(self,
                    sample,
                    change_class,
                    plausible=True,
                    use_mad=True,
                    method="NSGA2",
                    **kwargs):
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
                      "sampling": "real_random",
                      "crossover": ["real_sbx", 0.9, 15],
                      "mutation": ["real_pm", 20],
                      "verbose": False}
            
            for key in kwargs.keys():
                try:
                    opt_param_[key] = kwargs[key]
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
                                  change_class, plausible, use_mad)

            res = minimize(problem,
                           algorithm,
                           termination,
                           seed=self.seed_,
                           save_history=True,
                           verbose=opt_param_["verbose"])

            # Pulling probability out from the problem for display purposes later
            self.sample_proba_ = problem.x_orig_proba
            
            # Store the result set
            self.res_ = res

        else:
            raise NotImplementedError(
                "NSGA2 is currently the only optimizer combatible with ChWeiMO.")
            
    def show_cf(self, n_solutions, return_cf=False):
        
        # Grab the entire final population solutions and their objective scores
        xs = pd.DataFrame(self.res_.history[-1].pop.get("X")-self.sample_, columns=self.col_names_)
        fs = (pd.DataFrame(self.res_.history[-1].pop.get("F"), columns=["Distance", "Pred Δ"])*-1)
        final_pop = pd.concat((xs, fs), axis=1)
        
        # Grab the final non-dominated solutions and their objective scores
        non_dom_xs = pd.DataFrame(self.res_.X-self.sample_, columns=self.col_names)
        non_dom_fs = pd.DataFrame(self.res_.F, columns=["Distance", "Pred Δ"])*-1
        non_dom = pd.concat((non_dom_xs, non_dom_fs), axis=1)

        # Convert the distance back to its actual value
        final_pop["Distance"] = 1/fs["Distance"]
        non_dom["Distance"] = 1/non_dom["Distance"]

        # For all of the final population values, sort by the best objective values seperately
        best_obj1 = final_pop.sort_values(by="Distance").head(n_solutions)
        best_obj1.index = ["Obj1" for _ in range(best_obj1.shape[0])]

        best_obj2 = final_pop.sort_values(by="Pred Δ", ascending=False).head(n_solutions)
        best_obj2.index=["Obj2" for _ in range(best_obj2.shape[0])]

        # Then recombine, bring the objective index out as a seperate column and rename it
        final_pop = pd.concat((best_obj1, best_obj2)).reset_index()
        final_pop.rename({"index":"Best of"}, axis=1, inplace=True)

        # Initialize an array that represents which of the solutions are non-dominated
        non_dom_col = np.full((final_pop.shape[0], 1), "X")

        # Iterate through both the final population and the non-dominated solutions,
        # determine which are the non-dominated and mark them
        for i, v in final_pop.drop("Best of", axis=1).iterrows():
            for _, nv in non_dom.iterrows():
                if nv.equals(v):
                    non_dom_col[i] = "✓"
        
        final_pop["Non-Dom?"] = non_dom_col

        # Style the final output
        final_pop = final_pop.style.pipe(make_pretty)
        
        if return_cf:
            return final_pop
        else:
            display(
                pd.DataFrame(np.append(self.sample_, [self.sample_proba_]).reshape(1,-1),
                             columns=np.append(self.col_names_, ["Orig Class Prob"]))
                )
            display(final_pop)
            
    def get_results(self):
        return self.res_
        
        
def make_pretty(s):
    s.set_caption("Counterfactuals")
    
    s.apply(
        lambda s: np.where(s["Best of"] == "Obj1", ["background-color:#D55E00"]*len(s), ["background-color:#0072B2"]*len(s)), axis=1
        )
    
    s.apply(
        lambda s: np.where(s== "✓", "background-color:#009E73", "background-color:#CC79A7"), axis=0, subset=["Non-Dom?"]
        )
    
    s.set_table_styles([
        {"selector": "th.col_heading.level0", "props": "font-size:0.8em; background-color:grey;"},
        {"selector": "th:not(.index_name)", "props": "font-size:0.8em; background-color:grey;"},
        {"selector": "th", "props": [("border", "1px solid white !important")]}
    ])
    
    s.format(
        precision=1, formatter={
            ("Distance"): "{:.1f}",
            ("Pred Δ"): "{:.2f}"}
        )
    
    return s