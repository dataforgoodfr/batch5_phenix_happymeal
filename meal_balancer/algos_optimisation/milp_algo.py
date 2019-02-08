from __future__ import print_function
from ortools.sat.python import cp_model

import cvxpy as cvx
import json
import numpy as np
import pandas as pd
import sys
#sys.path.insert(0,'../..')
from meal_balancer.grouping import create_batches
from gurobipy import *


# optimisation algorithm for balanced meal creation
# cvxpy + glpk implementation of Julie Seguela's gurobi implementation
# https://github.com/dataforgoodfr/batch5_phenix_happymeal/blob/master/algos-optimisation/milp_gurobi.py

__author__ = 'Aoife Fogarty'
__version__ = '0.1'
__maintainer__ = 'Aoife Fogarty'
__status__ = 'Development'


def optimize_baskets(listing_df, cat_distrib, cat_mandatory, delta_auth, meal_weight, 
                     results_filename, solver):
    '''
    Args:
        listing_df (pandas df): contains weight, quantity and
                   category of products to distribute into baskets
        cat_distrib (dict): ideal distribution, keys=categories
        delta_auth (float): max authorised difference between ideal
                            and actual category distributions
        meal_weight (float): ideal weight in grams of one basket
        solver (str): GLPK or CBC
    Return:
        df with details of products in each basket after optimized distribution, df has column 'allocated_basket' with value of 0 if item was not distributed
        json with details of distributed and undistributed products, for plotting in UI
    '''

    # use the delta parameter to set allowed upper and lower
    # limits to the category distribution of each basket
    cat_distrib_upper = {k: (v * (1 + delta_auth) * meal_weight)
                         for (k,v) in cat_distrib.items()}
    cat_distrib_lower = {k: (v * (1 - delta_auth) * meal_weight)
                         for (k,v) in cat_distrib.items() if k in cat_mandatory}
    cat_distrib       = {k: v for k, v in cat_distrib.items() if v > 0}
    
    # initial estimation of min and max number of meals we can make
    n_meals_max = estimate_nmeals_max(listing_df, cat_distrib_lower)
    n_meals_min = 1
    # n_meals_min = estimate_nmeals_min(listing_df, cat_distrib, meal_weight)
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
            print('cannot find solution for ', n_meals)
            solution = solution_previous
            break

    # postprocess solution to get jsons for plotting
    results_json = postprocess_optimised_solution_for_ui(solution, listing_df, results_filename)
    #print(results_json)

    # get complete list of items in each basket, for output as csv
    df_solution = postprocess_optimised_solution(solution, listing_df)

    return df_solution, results_json


def estimate_nmeals_min(listing_df, cat_distrib, meal_weight):
    '''
    Estimate the min number of balanced meals that can be made
    from a given listing, based on the 'naive distribution' algorithm

    Args:
        listing_df (pandas df): contains weight, quantity and
                   category of products to distribute into baskets
        cat_distrib (dict): ideal distribution, keys=categories
        meal_weight (float): ideal weight in grams of one basket
    '''
    # create new df with one line per item instead of one line per product
    df_new = pd.DataFrame([listing_df.ix[idx]
                           for idx in listing_df.index
                           for _ in range(int(listing_df.ix[idx]['quantity']))])

    # put items in format needed by create_batches
    df_new = df_new.reset_index(drop=True)\
                   .rename(columns={'codeAlim_1': 'category',
                                    'weight_grams': 'quantity'})
    items = df_new[['category', 'quantity']].to_dict(orient='records')

    assert np.isclose(len(items), listing_df.quantity.sum(), rtol=0.05)

    all_batches, storage, big_items, unidentified, tmp_items, loss = create_batches(items=items,
            balanced_split=cat_distrib,
            batch_qty=meal_weight,
            overflow_thresh=0.05,
            underflow_thresh=0.05)

    # TODO we could also use other info from this function
    # to initialise the MILP algo in the following step

    n_meals_min = len(all_batches)
    return n_meals_min


def estimate_nmeals_max(listing_df, cat_distrib_lower):
    '''
    Estimate the max number of balanced meals that can be made
    from a given listing, based on the number of items and
    total weight in each category
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
    sum_weights = df_g['weight_grams'].values * df_g['quantity'].values

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
    n_meals_max_2 = np.min(df_g.loc[df_g['codeAlim_1'].isin(cat_distrib_lower.keys()), 'quantity'].values)

    n_meals_max = int(min(n_meals_max_1, n_meals_max_2))

    return n_meals_max


def load_meal_balancing_parameters(distrib_filename, listing_df):
    '''
    Args:
        distrib_filename (str): containing ideal distribution and food categories
    '''
    
    cat_status = 1
    
    map_label2_code1, map_code1_label1 = load_category_mappings(distrib_filename)

    df = pd.read_csv(distrib_filename, sep=';')
    
    # eligible categories (found in listing)
    cat_ok = list(set(listing_df['codeAlim_1']))
    cat_mandatory = list(set(df.loc[df['mandatory'] == 1, 'codeAlim_1']))

    if pd.Series(cat_mandatory).isin(cat_ok).sum() < len(cat_mandatory):
        cat_missing = pd.Series(list(set(cat_mandatory)-set(cat_ok)))
        cat_missing_labels = cat_missing.apply(lambda x: map_code1_label1[x]).values
        cat_status = 0
        print(', '.join(cat_missing_labels) + ' are not in input listing')
        #sys.exit(', '.join(cat_missing_labels) + ' are not in input listing')

    # for level 1 categories, we have to remove duplicates first
    df = df.loc[df['codeAlim_1'].isin(cat_ok),].drop_duplicates(['codeAlim_1', 'idealDistrib_1'])
    #df = df.drop_duplicates(['codeAlim_1', 'idealDistrib_1'])
    
    # transform to 100% 
    if cat_status == 1:
        df['idealDistrib_1'] = df['idealDistrib_1']*(1/sum(df['idealDistrib_1'].values))

    cat_distrib = dict(zip(df['codeAlim_1'].values, df['idealDistrib_1'].values))
   
    # assert np.isclose(1.0, sum(cat_distrib.values()))

    return cat_distrib, cat_mandatory, cat_status


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
    elif solver == 'CBC':
        solution_matrix = solver_cbc(listing_df, cat_distrib_upper, cat_distrib_lower, n_meals)

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

    if solution is None:
        return None
    else:
        names      = listing_df['Produit_Nom'].values
        ean        = listing_df['EAN'].values
        categories = listing_df['labelAlim_1'].values
        weights    = listing_df['weight_grams'].values
    
        meal_domain = range(solution.shape[0])
        pdt_domain = range(solution.shape[1])
    
        # init for remaining items basket
        df_solution = listing_df[['Produit_Nom', 'EAN', 'labelAlim_1', 'quantity', 'weight_grams']]
        df_solution['allocated_basket'] = 0
        
        for pdt in pdt_domain:
            for meal in meal_domain:
                if solution[meal, pdt] > 0:
                    df_solution = df_solution.append({
                            'Produit_Nom': names[pdt],
                            'EAN': ean[pdt],
                            'labelAlim_1': categories[pdt],
                            'quantity': solution[meal, pdt],
                            'weight_grams': weights[pdt],
                            'allocated_basket': meal+1                        
                            }, ignore_index = True)
                    # update remaining quantity of product pdt
                    df_solution.iloc[pdt, 3] = df_solution.iloc[pdt, 3]-solution[meal, pdt]
        # remove rows if no item
        df_solution = df_solution.loc[df_solution['quantity'] > 0, ]
        
        return(df_solution)


def postprocess_optimised_solution_for_ui(solution, listing_df, results_filename):
    '''
    Process solution matrix (output of MILP solver)
    to get json for plotting in UI
    '''

    # how many items of each product were there originally
    total = listing_df.quantity.values

    # how many items of each product have been allocated to baskets
    if solution is None:
        n_balanced_meals = 0
        allocated = np.zeros(len(total), dtype=np.float64)
    else:
        allocated = np.sum(solution, axis=0)  # sum down columns (check)
        n_balanced_meals = solution.shape[0]

    assert len(total) == len(allocated)

    remaining = total - allocated
    n_allocated_items = np.sum(allocated)
    n_remaining_items = np.sum(remaining)

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

    df_g['input_weighted_frac'] = (df_g['allocated_weighted'] + df_g['remaining_weighted'])/ total_weight

    if total_weight_allocated_items > 0:
        # df_g['allocated_weighted_frac'] = df_g['allocated_weighted'] / total_weight_allocated_items
        df_g['allocated_weighted_frac'] = df_g['allocated_weighted'] / total_weight
    else: 
        df_g['allocated_weighted_frac'] = 0.0
        
    if total_weight_remaining_items > 0:
        # df_g['remaining_weighted_frac'] = df_g['remaining_weighted'] / total_weight_remaining_items
        df_g['remaining_weighted_frac'] = df_g['remaining_weighted'] / total_weight
    else:
        df_g['remaining_weighted_frac'] = 0.0

    results = {
            'pct_input_items'    : list(zip(df_g['labelAlim_1'].values,
                                        df_g['input_weighted_frac'].values)),
            'pct_allocated_items': list(zip(df_g['labelAlim_1'].values,
                                        df_g['allocated_weighted_frac'].values)),
            'pct_remaining_items': list(zip(df_g['labelAlim_1'].values,
                                        df_g['remaining_weighted_frac'].values)),
            'nb_balanced_meals': int(n_balanced_meals),
            'nb_allocated_items': int(n_allocated_items),
            'nb_remaining_items': int(n_remaining_items),
            'total_input_weight': int(total_weight),
            'total_allocated_weight': int(total_weight_allocated_items),
            'total_weight_remaining_items': int(total_weight_remaining_items),
            'pct_weight_allocated_items': total_weight_allocated_items / total_weight,
            'pct_weight_remaining_items': total_weight_remaining_items / total_weight}

    with open(results_filename, 'w') as f:
        json.dump(results, f)

    return json.dumps(results)


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
    cat_domain_lower = list(cat_distrib_lower.keys())
    cat_domain_upper = list(cat_distrib_upper.keys())

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
                  for cat in cat_domain_upper \
                  for meal in meal_domain]
    constraints.extend(category_constraints)

    category_constraints = [cvx.sum(list(X[meal, pdt] * weights[pdt] \
                                    for pdt in pdt_domain \
                                    if categories[pdt] == cat)) \
                  >= cat_distrib_lower[cat]\
                  for cat in cat_domain_lower \
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


class ObjectivePrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self):
        print('Solution %i, time = %f s, objective = %i' %
              (self.__solution_count, self.WallTime(), self.ObjectiveValue()))
        self.__solution_count += 1


def solver_cbc(listing_df, cat_distrib_upper, cat_distrib_lower, n_meals):
   
    """Solve the assignment problem."""
    model = cp_model.CpModel()
    
    # CP-SAT solver is integer only.
   
    weights = np.floor(listing_df['weight_grams'].values).astype(int)
    categories = listing_df['codeAlim_1'].values
    quantities = np.floor(listing_df['quantity'].values).astype(int)
    
    n_products = len(listing_df)
    meal_domain = range(n_meals)
    pdt_domain = range(n_products)
    
    cat_distrib_lower = {k: math.ceil(v) for (k,v) in cat_distrib_lower.items()}
    cat_distrib_upper = {k: math.floor(v) for (k,v) in cat_distrib_upper.items()}
    
    cat_domain_lower = list(cat_distrib_lower.keys())
    cat_domain_upper = list(cat_distrib_upper.keys())

    horizon = math.floor(quantities.max())

    # Variables

    ## x_ij = 1 if product i is assigned to meal j
    x = {}
    for meal in meal_domain:
        for pdt in pdt_domain:
            x[meal, pdt] = model.NewIntVar(0, horizon, 'x[%i,%i]' % (meal, pdt))
            #x[meal, pdt] = model.NewBoolVar('x[%i,%i]' % (meal, pdt))

    ### Constraints

    # Each product i is assigned to a meal and only one.
    for pdt in pdt_domain:
        model.Add(sum(x[meal, pdt] for meal in meal_domain) <= quantities[pdt])

    # Can't oversize a category too much    
    for meal in meal_domain:
        for cat in cat_domain_upper:
            model.Add(sum(x[meal, pdt]*weights[pdt] \
                          for pdt in pdt_domain if categories[pdt] == cat) 
                      <= cat_distrib_upper[cat])
            
    # Can't undersize a category too much      
    for meal in meal_domain:
        for cat in cat_domain_lower:       
            model.Add(sum(x[meal, pdt]*weights[pdt] \
                          for pdt in pdt_domain if categories[pdt] == cat) 
                      >= cat_distrib_lower[cat])

    # Objective
    model.Maximize(sum(x[meal, pdt]*weights[pdt] \
                       for pdt in pdt_domain \
                       for meal in meal_domain))

    # Solve and print out the solution
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2
    objective_printer = ObjectivePrinter()
    status = solver.SolveWithSolutionCallback(model, objective_printer)
    #print(solver.ResponseStats())
    
    x_matrix = np.zeros([n_meals, n_products], dtype = int)
    
    if (status == cp_model.OPTIMAL) | (status == cp_model.FEASIBLE):
        for i in meal_domain:
            print('Meal %i' % i)
            for j in pdt_domain:
                if solver.Value(x[i, j]):
                    #print('  - Product %i' % j + ' : ' + str(solver.Value(x[i, j])))
                    x_matrix[i, j] = solver.Value(x[i, j])
        print(str(solver.ObjectiveValue()) + 'g allocated')
        return x_matrix
    else:
        return None


def solver_gurobi(listing_df, cat_distrib_upper, cat_distrib_lower, n_meals):
    '''
    Returns:
        integer matrix of dimensions n_meals * n_products
        Each matrix element is the number of items of a product in a basket
    '''
    
    # Fixed args
    time_limit = 600
    display_interval = 30
        
    print('Trying to find solution for {} meals'.format(n_meals))

    # Model creation, decision variable Xi,j, meal i associated with product j?
    hm = Model("HM")
   
    pdt_info_dict = listing_df.to_dict()
    
    n_products = len(listing_df)
    weights = listing_df['weight_grams'].values
    categories = listing_df['codeAlim_1'].values
    quantities = listing_df['quantity'].values

    meal_domain = np.arange(0, n_meals, step = 1)
    pdt_domain = listing_df.index.tolist()
    cat_domain_lower = list(cat_distrib_lower.keys())
    cat_domain_upper = list(cat_distrib_upper.keys())


    #print(categories)
    #print(quantities)
    #print(weights)
    #print(cat_distrib_upper)
    #print(cat_distrib_lower)

    #########################
    ## System specifiction ##
    #########################

    # matrix of integers to solve for
    X = hm.addVars(meal_domain, pdt_domain, vtype=GRB.INTEGER, name='x')

    # Objective function: maximize the weight in the baskets
    #hm.setObjective(quicksum(X.sum('*', pdt) \
    #                         for pdt in pdt_domain), \
    #                sense = GRB.MAXIMIZE)
        
    hm.setObjective(quicksum(X[meal, pdt] * pdt_info_dict['weight_grams'][pdt] \
                             for pdt in pdt_domain  \
                             for meal in meal_domain),
                    sense = GRB.MAXIMIZE)

    # Constraint: can't use more than available for each product
    hm.addConstrs((X.sum('*', pdt) <= pdt_info_dict['quantity'][pdt]\
                  for pdt in pdt_domain))

    # Constraint: can't oversize or undersize a category too much
    hm.addConstrs(quicksum((X[meal, pdt] * pdt_info_dict['weight_grams'][pdt]) \
                           for pdt in pdt_domain if pdt_info_dict['codeAlim_1'][pdt] == cat) \
                  <= cat_distrib_upper[cat]\
                  for cat in cat_domain_upper \
                  for meal in meal_domain)
    
    hm.addConstrs(quicksum((X[meal, pdt] * pdt_info_dict['weight_grams'][pdt]) \
                           for pdt in pdt_domain if pdt_info_dict['codeAlim_1'][pdt] == cat) \
                  >= cat_distrib_lower[cat]
                  for cat in cat_domain_lower \
                  for meal in meal_domain)
    
    hm.Params.TimeLimit       = time_limit
    hm.Params.Threads         = 0
    hm.Params.DisplayInterval = display_interval
    
    hm.optimize()
    
    status = hm.status

    res_df = pd.DataFrame(columns = ('meal', 'pdt', 'qty'))
    
    for meal in meal_domain:
        for pdt in pdt_domain:
            if X[meal, pdt].getAttr('x') >= 1:
                res_df = res_df.append({
                        'meal': meal,
                        'pdt': pdt,
                        'qty': X[meal, pdt].getAttr('x')
                        }, ignore_index = True)
                
    # Adding product name and category
    res_df = res_df.merge(listing_df.reset_index(), left_on = 'pdt', right_on = 'index')
    print(sum(res_df.loc[res_df['qty'] > 0, 'weight_grams']*res_df.loc[res_df['qty'] > 0, 'qty']))
    print(sum(listing_df['weight_grams']*listing_df['quantity']))
    
    return res_df
