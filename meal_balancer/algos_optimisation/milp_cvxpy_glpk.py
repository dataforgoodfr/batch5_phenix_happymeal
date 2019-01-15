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


def optimize_baskets(listing_df, cat_distrib, delta_auth, meal_weight, solver):
    '''
    Args:
        listing_df (pandas df): contains weight, quantity and category of products to distribute into baskets
        cat_distrib (dict): ideal distribution, keys=categories
        delta_auth (float): max authorised difference between ideal and actual category distributions
        meal_weight (float): ideal weight in grams of one basket
        solver (str): GLPK or GUROBI
    Return:
        df with details of products in each basket after optimized distribution, df has column 'allocated_basket' with value of 0 if item was not distributed
        json with details of distributed and undistributed products, for plotting in UI
    '''

    # use the delta parameter to set allowed upper and lower
    # limits to the category distribution of each basket
    cat_distrib_upper = {k: (v + delta_auth) * meal_weight
                         for (k,v) in cat_distrib.items()}
    cat_distrib_lower = {k: (v - delta_auth) * meal_weight
                         for (k,v) in cat_distrib.items()}

    # initial estimation of min and max number of meals we can make
    # TODO
    n_meals_min, n_meals_max = estimate_nmeals(listing_df, cat_distrib)

    for n_meals in range(n_meals_min, n_meals_max+1, 1):

        solution = optimize_baskets_for_nmeals(listing_df, cat_distrib_upper,
                                               cat_distrib_lower, n_meals,
                                               solver)

        if solution is None:
            # solver could not find an optimal solution
            # we have reached the max possible number of meals
            break

        # TODO deal with case where even n_meals_min doesn't give a viable solution

    # postprocess solution to get jsons for plotting
    results_json = postprocess_optimised_solution_for_ui(solution.value, listing_df)

    # get complete list of items in each basket, for output as csv
    #df_solution = postprocess_optimised_solution(solution)
    df_solution = None  # TODO

    return df_solution, results_json


def estimate_nmeals(listing_df, cat_distrib_lower):
    '''
    Estimate the number of balanced meals that can be made
    from a given listing
    '''
    # # if a category is missing from the dataframe
    # # the number of possible meals is 0
    # cat_in_listing = listing_df[''].unique()
    # for cat in cat_distrib.keys():
    #     if cat not in cat_in_listing:
    #         print('Cannot fnd malanced meals because category {} is missing from listing'.format(cat))
    #         return 0, 0

    # Use two different methods to estimate the maximum number of possible meals
    df_g = listing_df.groupby('codeAlim_2', as_index=False).agg({'weight_grams': 'sum', 'quantity': 'sum'})
    categories = df_g['codeAlim_2'].values

    # Method 1: total weight in each category in the listing / minimum
    # weight that should be in each basket in that category
    sum_weights = df_g['weight_grams'].values
    cat_weights = np.asarray([cat_distrib_lower[c] for c in categories])
    n_meals_max_1 = np.min(np.floor(sum_weights / cat_weights))

    # Method 2: look at total number of items in each cat
    n_meals_max_2 = np.min(df_g['quantity'].values)

    n_meals_max = min(n_meals_max_1, n_meals_max_2)

    # TODO how to set n_meals_min ?
    n_meals_min = 1

    return n_meals_min, n_meals_max


def load_meal_balancing_parameters(distrib_filename):
    '''
    Args:
        distrib_filename (str): containing ideal distribution
    '''

    # TODO load all parameters from file
    delta_auth = 0.05
    meal_weight = 2000

    df = pd.read_csv(distrib_filename)

    # for level 1 categories, we have to remove duplicates first
    df = df.drop_duplicates(['codeAlim_1', 'idealDistrib_1'])

    cat_distrib = dict(zip(df['codeAlim_1'].values), df['idealDistrib_1'].values)
    print(cat_distrib, 'yyy')

    return cat_distrib, delta_auth, meal_weight


def optimize_baskets_for_nmeals(listing_df, cat_distrib_upper, cat_distrib_lower, n_meals, solver):
    '''
    For a given number of meals (n_meals), get the distribution of products
    into baskets/meals that best corresponds to a given distribution of
    categories in each basket

    Args:
        listing_df (pandas df): contains weight, quantity and category of products to distribute into baskets
        cat_distrib_upper, cat_distrib_lower (dict): ideal distribution
        n_meals (int): number of baskets to construct
        solver (str): GLPK or GUROBI
    Returns:
        solution matrix if a solution could be found for this value of n_meals
        Otherwise returns None
    '''

    # construct an optimised n_meals * n_products matrix
    if solver == 'GLPK':
        solution_matrix = solver_cvxpy_glpk(listing_df, cat_distrib_upper, cat_distrib_lower, n_meals)
    elif solver == 'GUROBI':
        # NOT YET IMPLEMENTED
        solution_matrix = solver_gurobi(listing_df, cat_distrib_upper, cat_distrib_lower, n_meals)

    print(solution_matrix)
    quit()

    if solution:
        return solution_matrix
    else:
        return None


def postprocess_optimised_solution(solution, listing_df):
    '''
    Process solution matrix (output of MILP solver)
    to get dataframe with allocation of items to baskets

    Note that the output contains one line per item, and not one
    line per product as in other dataframes earlier in the workflow

    Input:
        solution (matrix) n_meals * n_products
    Returns:
        dataframe with one line for each item, and column
        'allocated_basket' (0 if item was not allocated)
    '''
    pass


def postprocess_optimised_solution_for_ui(solution, listing_df):
    '''
    Process solution matrix (output of MILP solver)
    to get json for plotting in UI
    '''

    # how many items of each product have been allocated to baskets
    allocated = np.sum(solution, axis=1)  # sum down columns (check)

    # how many items of each product were there originally
    total = listing_df.quantities.values

    if len(total) != len(allocated):
        print('HERE')
        quit()

    remaining = total - allocated

    listing_df['allocated'] = allocated
    listing_df['remaining'] = remaining

    listing_df['allocated_weighted'] = allocated * listing_df['weight_grams']
    listing_df['remaining_weighted'] = remaining * listing_df['weight_grams']

    df_g = listing_df.groupby(['codeAlim_2', 'labelAlim_2'], as_index=False).agg({'allocated_weighted': 'sum', 'remaining_weighted': 'sum'})

    df_g['allocated_weighted_frac'] = df['allocated_weighted'] / np.sum(df['allocated_weighted'].values)
    df_g['remaining_weighted_frac'] = df['remaining_weighted'] / np.sum(df['remaining_weighted'].values)

    results = {
            'allocated_items': zip(df_g['labelAlim_2'].values,
                                   df_g['allocated_weighted_frac'].values),
            'remaining_items': zip(df_g['labelAlim_2'].values,
                                   df_g['remaining_weighted_frac'].values)}

    return results


def solver_cvxpy_glpk(listing_df, cat_distrib_upper, cat_distrib_lower, n_meals):
    '''
    Returns:
        integer matrix of dimensions n_meals * n_products
        Each matrix element is the number of items of a product in a basket
    '''

    print('Trying to find solution for {} meals'.format(n_meals))

    n_products = len(listing_df)
    weights = listing_df['weight_grams'].values
    categories = listing_df['codeAlim_2'].values
    quantities = listing_df['quantity'].values

    meal_domain = np.arange(0, n_meals, step = 1)
    pdt_domain = np.arange(0, n_products, step = 1)
    cat_domain = list(cat_distrib_upper.keys())

    #########################
    ## System specifiction ##
    #########################

    # matrix of integers to solve for
    X = cvx.Variable((n_meals, n_products), integer=True)

    # Objective function: maximize the weight in the baskets
    # (note: cvx.multiply is element-wise multiplication)
    weights_matrix = np.tile(weights, (n_meals, 1))  # shape n_meals * n_products
    print('weights_matrix', weights_matrix.shape)
    objective = cvx.Maximize(cvx.sum(cvx.multiply(X, weights_matrix)))

    constraints = []

    # Constraint: each product can only be in one basket or no basket
    #constraints.append(cvx.sum(X, axis = 1) <= quantities)
    constraints.append(X >= 0)

    # Constraint: limit difference between actual and ideal category distributions
    category_constraints = [cvx.sum(list(X[meal, pdt] * weights[pdt] \
                                    for pdt in pdt_domain \
                                    if categories[pdt] == cat)) \
                  <= cat_distrib_upper[cat]\
                  for cat in cat_domain \
                  for meal in meal_domain]
    constraints.extend(category_constraints)

    category_constraints = [cvx.sum(list(X[meal, pdt] * weights[pdt] \
                                    for pdt in pdt_domain \
                                    if categories[pdt] == cat)) \
                  >= cat_distrib_lower[cat]\
                  for cat in cat_domain \
                  for meal in meal_domain]
    constraints.extend(category_constraints)

    prob = cvx.Problem(objective, constraints)

    parameters = {'tm_lim' : 10}
    error = False
    try:
        solution_status = prob.solve(solver='GLPK_MI',verbose = True, solver_specific_opts=parameters)
    except DCPError:
        print('Problem is not Disciplined Convex Programming compliant')
        error = True
    except SolverError:
        print('No suitable solver exists among the installed solvers, or an unanticipated error has been encountered')
        error = True

    print(X.value)
    if error:
        return None

    return X

