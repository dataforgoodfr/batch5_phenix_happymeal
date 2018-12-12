import cvxpy as cvx
import numpy as np
import pandas as pd
import sys
sys.path.insert(0,'../..')
from utils.off_mgt import convert_quantity
from sklearn.preprocessing import LabelEncoder

# ** work in progress **

# optimisation algorithm for balanced meal creation
# cvxpy + glpk implementation of Julie Seguela's gurobi implementation
# https://github.com/dataforgoodfr/batch5_phenix_happymeal/blob/master/algos-optimisation/milp_gurobi.py

__author__ = 'Aoife Fogarty'
__version__ = '0.1'
__maintainer__ = 'Aoife Fogarty'
__status__ = 'Development'


def main():

    # TODO replace with more realistic distribution for tests
    cat_distrib = {1: 0.11, 2: 0.02, 3: 0.02, 4: 0.2, 5: 0.25, 6: 0.32, 7: 0.01,
                   8: 0.005, 9: 0.005, 10: 0.05, 11: 0.01}
    n_categories = max(cat_distrib.keys())
    delta_auth = 0.05
    meal_size = 1400
    n_meals = 10

    # use the delta parameter to set allowed upper and lower
    # limits to the category distribution of each basket
    cat_distrib_upper = {k: (v + delta_auth) * meal_size
                         for (k,v) in cat_distrib.items()}
    cat_distrib_lower = {k: (v - delta_auth) * meal_size
                         for (k,v) in cat_distrib.items()}

    use_real_test_data = True

    if use_real_test_data:
        # load real test data
        filename = 'sample_offres.csv'
        pdt_info_dict = load_real_test_data(filename)
        n_products = len(pdt_info_dict)
    else:
        # create random fake test data
        n_products = 100
        pdt_info_dict = create_fake_test_data(n_products, n_meals, n_categories)

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


def create_fake_test_data(n_products, n_meals, n_categories):
    '''
    create n_products products with random weights and categories and
    number of items
    '''

    # weight of each item
    max_item_weight = 200
    weights = np.random.random(size=n_products) * max_item_weight + 1.0

    # category of each item
    categories = np.random.random_integers(low=1, high=n_categories, size=n_products)

    # number of items of each product 
    # (based on observed data it is something like an exponential distribution)
    n_items = np.random.exponential(scale=2.0, size=n_products)

    pdt_info_dict = {'qty_gram': weights, 'code_cat': categories, 'n_items': n_items}

    return pdt_info_dict


def load_real_test_data(filename):
    '''
    Parse csv file extracted from Offres table in Phenix database
    Warning: not all weights are correctly parsed yet
    Temporary function for testing and benchmarking MILP algos
    '''

    # category to use for tests
    category_to_use = 'pnns_groups_1'

    df = pd.read_csv(filename, sep=',', encoding='iso-8859-1')
    df = df[~df[category_to_use].isnull()]

    # get weight in grams of each product
    df['weight_per_entry'] = df.quantity_off.apply(lambda s: convert_quantity(s))
    df = pd.concat([df.drop(['weight_per_entry'], axis=1), df['weight_per_entry'].apply(pd.Series)], axis=1)
    df = df[~df['val'].isnull()]
    df = df.rename(columns={'val': 'qty_gram'})

    # encode category of each product as an integer
    le = LabelEncoder()
    df['code_cat'] = le.fit_transform(df[category_to_use].values)
    # TODO return le.classes_ (label of each integer class) for later use

    # regroup duplicate products
    df['QuantiteValeur'] = df.groupby(['EAN', 'code_cat', 'qty_gram'])['QuantiteValeur'] \
                             .transform('sum') \
                             .astype(int)
    # Note: in some very rare cases QuantiteValeur is not an integer, 
    # but we neglect this
    df = df[['EAN', 'Nom', 'code_cat', category_to_use, 'qty_gram', 'QuantiteValeur']] \
                         .drop_duplicates(['EAN', 'code_cat', 'qty_gram']) \
                         .reset_index(drop = True)

    df = df[df['QuantiteValeur'] > 0]

    return df.to_dict(orient='list')


if __name__ == '__main__':
    main()
