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
    n_meals_min, n_meals_max = estimate_nmeals(listing_df, cat_distrib_lower)
    print('Trying to find between {} and {} meals'.format(n_meals_min, n_meals_max))

    # initialise solution matrix
    solution = None

    for n_meals in range(n_meals_min, n_meals_max+1, 1):

        # save best solution so far
        solution_previous = solution

        solution = optimize_baskets_for_nmeals(listing_df, cat_distrib_upper,
                                               cat_distrib_lower, n_meals,
                                               solver)

        if solution is None:
            # solver could not find an optimal solution
            # we have reached the max possible number of meals
            print('cannot find solution for ',n_meals)
            solution = solution_previous
            break

        # TODO deal with case where even n_meals_min doesn't give a viable solution

    # postprocess solution to get jsons for plotting
    results_json = postprocess_optimised_solution_for_ui(solution, listing_df)
    print(results_json)

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
    df_g = listing_df.groupby('codeAlim_1', as_index=False).agg({'weight_grams': 'sum', 'quantity': 'sum'})
    categories = df_g.codeAlim_1.values

    # Method 1: total weight in each category in the listing / minimum
    # weight that should be in each basket in that category
    sum_weights = df_g['weight_grams'].values

    weights_in_listing = dict(zip(categories, sum_weights))
    max_meals = []
    for c in cat_distrib_lower.keys():
        w_1 = weights_in_listing.get(c, 0.0)
        w_2 = max(cat_distrib_lower[c], 0)
        if w_2 == 0:
            continue
        max_meals.append(np.floor(w_1 / w_2))

    n_meals_max_1 = min(max_meals)

    # Method 2: look at total number of items in each cat
    n_meals_max_2 = np.min(df_g['quantity'].values)

    n_meals_max = int(min(n_meals_max_1, n_meals_max_2))

    # TODO how to set n_meals_min ?
    n_meals_min = 1

    return n_meals_min, n_meals_max


def load_meal_balancing_parameters(distrib_filename):
    '''
    Args:
        distrib_filename (str): containing ideal distribution and food categories
    '''

    # TODO load all parameters from file
    delta_auth = 0.05
    meal_weight = 1000

    df = pd.read_csv(distrib_filename, sep=';')

    # for level 1 categories, we have to remove duplicates first
    df = df.drop_duplicates(['codeAlim_1', 'idealDistrib_1'])

    cat_distrib = dict(zip(df['codeAlim_1'].values, df['idealDistrib_1'].values))

    assert np.isclose(1.0, sum(cat_distrib.values()))

    return cat_distrib, delta_auth, meal_weight


def load_category_mappings(distrib_filename):
    '''
    Args:
        distrib_filename (str): containing ideal distribution and food categories
    '''

    df = pd.read_csv(distrib_filename, sep=';')

    map_label2_code1 = dict(zip(df['labelAlim_2'].values,
                                df['codeAlim_1'].values))
    map_code1_label1 = dict(zip(df['codeAlim_1'].values,
                                df['labelAlim_1'].values))

    return map_label2_code1, map_code1_label1 


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

    # TODO this function isn't needed any more

    # construct an optimised n_meals * n_products matrix
    if solver == 'GLPK':
        solution_matrix = solver_cvxpy_glpk(listing_df, cat_distrib_upper, cat_distrib_lower, n_meals)
    elif solver == 'GUROBI':
        # NOT YET IMPLEMENTED
        solution_matrix = solver_gurobi(listing_df, cat_distrib_upper, cat_distrib_lower, n_meals)

    return solution_matrix


def map_cat2_cat1(code):
    '''
    Map level 2 food categories to level 1 categories
    '''
    mapping = {0:0, 11:10, 12:10, 20:20, 30:30, 41:40, 42:40, 51:50, 52:50, 61:60, 62:60, 70:70}

    return mapping[code]


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
    allocated = np.sum(solution, axis=0)  # sum down columns (check)
    n_balanced_meals = solution.shape[0]
    print(allocated)

    # how many items of each product were there originally
    total = listing_df.quantity.values

    assert len(total) == len(allocated)

    remaining = total - allocated
    n_remaining_items = np.sum(remaining)
    print(remaining)

    listing_df['allocated'] = allocated
    listing_df['remaining'] = remaining

    listing_df['allocated_weighted'] = allocated * listing_df['weight_grams']
    listing_df['remaining_weighted'] = remaining * listing_df['weight_grams']

    df_g = listing_df.groupby(['codeAlim_1', 'labelAlim_1'], 
                              as_index=False).agg({'allocated_weighted': 'sum', 
                                                   'remaining_weighted': 'sum'})

    total_weight_allocated_items = np.sum(df_g['allocated_weighted'].values)
    total_weight_remaining_items = np.sum(df_g['remaining_weighted'].values)
    total_weight = total_weight_allocated_items + total_weight_remaining_items

    if total_weight_allocated_items > 0:
        df_g['allocated_weighted_frac'] = df_g['allocated_weighted'] / total_weight_allocated_items
    if total_weight_remaining_items > 0:
        df_g['remaining_weighted_frac'] = df_g['remaining_weighted'] / total_weight_remaining_items

    results = {
            'allocated_items': list(zip(df_g['labelAlim_1'].values,
                                        df_g['allocated_weighted_frac'].values)),
            'remaining_items': list(zip(df_g['labelAlim_1'].values,
                                        df_g['remaining_weighted_frac'].values)),
            'nb_balanced_meals': n_balanced_meals,
            'nb_remaining_items': n_remaining_items,
            'pct_weight_allocated_items': total_weight_allocated_items / total_weight,
            'pct_weight_remaining_items': total_weight_remaining_items / total_weight}

    print(results)
    quit()
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
    categories = listing_df['codeAlim_1'].values
    quantities = listing_df['quantity'].values

    meal_domain = np.arange(0, n_meals, step = 1)
    pdt_domain = np.arange(0, n_products, step = 1)
    cat_domain = list(cat_distrib_upper.keys())

    #print(categories)
    #print(quantities)
    #print(weights)
    #print(cat_distrib_upper)
    #print(cat_distrib_lower)

    #########################
    ## System specifiction ##
    #########################

    # matrix of integers to solve for
    X = cvx.Variable((n_meals, n_products), integer=True)

    # Objective function: maximize the weight in the baskets
    # (note: cvx.multiply is element-wise multiplication)
    weights_matrix = np.tile(weights, (n_meals, 1))  # shape n_meals * n_products
    #print(weights_matrix)
    objective = cvx.Maximize(cvx.sum(cvx.multiply(X, weights_matrix)))

    constraints = []

    # Constraint: each product can only be in one basket or no basket
    constraints.append(cvx.sum(X, axis = 0) <= quantities)
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

    if error:
        return None

    print('Put {:.0f} g in baskets out of {:.0f} g total'.format(prob.value, np.sum(weights * quantities)))

    return X.value
