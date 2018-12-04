import cvxpy as cvx
import numpy as np

# ** work in progress **

# optimisation algorithm for balanced meal creation
# cvxpy + glpk implementation of Julie Seguela's gurobi implementation
# https://github.com/dataforgoodfr/batch5_phenix_happymeal/blob/master/algos-optimisation/milp_gurobi.py

__author__ = 'Aoife Fogarty'
__version__ = '0.1'
__maintainer__ = 'Aoife Fogarty'
__status__ = 'Development'


cat_distrib = {1: 0.12, 2: 0.025, 3: 0.025, 4: 0.25, 5: 0.25, 6: 0.33}
n_categories = max(cat_distrib.keys())
delta_auth = 0.05
meal_size = 1400
n_items = 100
n_meals = 10

# use the delta parameter to set allowed upper and lower
# limits to the category distribution of each basket
cat_distrib_upper = {k: (v + delta_auth) * meal_size 
                     for (k,v) in cat_distrib.items()}
cat_distrib_lower = {k: (v - delta_auth) * meal_size 
                     for (k,v) in cat_distrib.items()}

#################################
## create fake test input data ##
#################################

# weight of each item
max_item_weight = 200
weights = np.random.random(size=n_items) * max_item_weight + 1.0

# category of each item
categories = np.random.random_integers(low=1, high=n_categories, size=n_items)
pdt_info_dict = {'qty_gram': weights, 'code_cat': categories}

weights = np.tile(weights, (n_meals, 1))  # shape n_meals * n_items

meal_domain = np.arange(0, n_meals, step = 1)
pdt_domain = np.arange(0, n_items, step = 1)
cat_domain = list(cat_distrib.keys())

#########################
## System specifiction ##
#########################

# matrix of integers to solve for
X = cvx.Variable((n_meals, n_items), boolean=True)

# Objective function: maximize the weight in the baskets
# (note: cvx.multiply is element-wise multiplication)
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

