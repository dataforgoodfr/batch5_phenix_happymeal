import cvxpy as cvx
import numpy as np
import pandas as pd
import sys
sys.path.insert(0,'../..')

# optimisation algorithm for balanced meal creation
# cvxpy + glpk implementation of Julie Seguela's gurobi implementation
# https://github.com/dataforgoodfr/batch5_phenix_happymeal/blob/master/algos-optimisation/milp_gurobi.py

__author__ = 'Aoife Fogarty'
__version__ = '0.1'
__maintainer__ = 'Aoife Fogarty'
__status__ = 'Development'


def optimize_baskets(listing_df, cat_distrib, delta_auth, meal_size):
    '''
    Args:
        listing_df (pandas df): contains weight, quantity and category of products to distribute into baskets
        cat_distrib (dict??): ideal distribution XXX
        delta_auth (float): max authorised difference between ideal and actual category distributions
        meal_size (float): ??max weight in grams of one basket
    Return:
        df with details of products in each basket, after optimized distribution
        df with details of products left undistributed after optimized distribution
        json with details of distributed and undistributed products, for plotting in UI
    '''

    # initial estimation of min and max number of meals we can make
    # TODO 
    n_meals_min = 0
    n_meals_max = 1

    for n_meals in range(n_meals_min, n_meals_max+1, 1):

        solution = optimize_baskets_for_mealsize(listing_df, cat_distrib, n_meals, delta_auth, meal_size):

        if solution is None:
            # we have reached the max possible number of meals
            break

        # TODO deal with case where even n_meals_min doesn't give a viable solution

    # postprocess solution to get jsons for plotting
    results_json = {}
    # results_json = postprocess_optimised_solution(solution)

    return df_distributed, df_undistributed, results_json


def load_meal_balancing_parameters(filename):
    '''
    Args:
        filename (str): containing preset parameters
    '''

    # TODO

    return cat_distrib, delta_auth, meal_size


def optimize_baskets_for_mealsize(listing_df, cat_distrib, n_meals, delta_auth, meal_size):
    '''
    For a given mealsize, get the distribution of products into
    baskets/meals that best corresponds to a given distribution of
    categories in each basket

    Args:
        listing_df (pandas df): contains weight, quantity and category of products to distribute into baskets
        cat_distrib (dict??): ideal distribution XXX
        delta_auth (float): max authorised difference between ideal and actual category distributions
        meal_size (float): ??max weight in grams of one basket
        n_meals (int): number of baskets to construct
    Returns:
        solution matrix if a solution could be found for this value of n_meals
        Otherwise returns None
    '''

    # use the delta parameter to set allowed upper and lower
    # limits to the category distribution of each basket
    cat_distrib_upper = {k: (v + delta_auth) * meal_size
                         for (k,v) in cat_distrib.items()}
    cat_distrib_lower = {k: (v - delta_auth) * meal_size
                         for (k,v) in cat_distrib.items()}

    timeout = 1000

    # construct an optimised n_meals * n_products matrix
    solution_matrix = solver_cvxpy_glpk(??, timeout)
    #solution_matrix = solver_gurobi()

    if solution:
        return solution_matrix
    else:
        return None


def postprocess_optimised_solution(solution):
    '''
    Process solution matrix (output of MILP solver)
    to get json for plotting in UI
    '''
    pass


def solver_cvxpy_glpk(n_meals, cat_distrib, listing_df):
    '''
    Returns:
        integer matrix of dimensions n_meals * n_products
        Each matrix element is the number of items of a product in a basket
    '''

    n_products = ??
    weights = ??

    meal_domain = np.arange(0, n_meals, step = 1)
    pdt_domain = np.arange(0, n_products, step = 1)
    cat_domain = list(cat_distrib.keys())

    #########################
    ## System specifiction ##
    #########################

    # matrix of integers to solve for
    X = cvx.Variable((n_meals, n_products), integer=True)

    # Objective function: maximize the weight in the baskets
    # (note: cvx.multiply is element-wise multiplication)
    weights = np.tile(weights, (n_meals, 1))  # shape n_meals * n_products
    objective = cvx.Maximize(cvx.sum(cvx.multiply(X, weights)))

    constraints = []

    # Constraint: each product can only be in one basket or no basket
    constraints.append(cvx.sum(X, axis = 1) <= 1)

    # Constraint: limit difference between actual and ideal category distributions
    category_constraints = [cvx.sum(list(X[meal, pdt] * pdt_info_dict['qty_gram'][pdt] \
                                    for pdt in pdt_domain \
                                    if pdt_info_dict['code_cat'][pdt] == cat)) \
                  <= cat_distrib_upper[cat]\
                  for cat in cat_domain \
                  for meal in meal_domain]
    constraints.extend(category_constraints)

    category_constraints = [cvx.sum(list(X[meal, pdt] * pdt_info_dict['qty_gram'][pdt] \
                                    for pdt in pdt_domain \
                                    if pdt_info_dict['code_cat'][pdt] == cat)) \
                  >= cat_distrib_lower[cat]\
                  for cat in cat_domain \
                  for meal in meal_domain]
    constraints.extend(category_constraints)

    prob = cvx.Problem(objective, constraints)

    print(prob)
    #print(prob.solve())
    print(prob.solve(solver='GLPK_MI',verbose = True))
    #prob.solve(solver=cvx.GLPK_MI)

    solution = 0

    return solution
