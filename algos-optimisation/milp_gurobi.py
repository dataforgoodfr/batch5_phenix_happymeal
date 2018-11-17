#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
MILP for balanced meal creation
Requires for a gurobi license
Example of use : 
python "algos-optimisation/milp_gurobi.py" "data/intermediate/sample_offres_qty.csv" "data/intermediate/happy_meals.csv" 10 0.1 1000
'''

__author__ = 'Julie Seguela'
__version__ = '0.1'
__maintainer__ = 'Julie Seguela'
__status__ = 'Development'


import sys
import pandas as pd
import numpy as np
from gurobipy import *

# Fixed args
time_limit = 600
display_interval = 30

cat_distrib = {1: 0.12, 2: 0.025, 3: 0.025, 4: 0.25, 5: 0.25, 6: 0.33}
cat_mapping_file = "data/mapping_pnnsgroups2.csv"

#listing_in = "data/intermediate/sample_offres_qty.csv"
#listing_out = "data/intermediate/happy_meals.csv"
#init_qty = 10
#delta_auth = 0.1
#meal_size = 1400

# Read script args
listing_in          = sys.argv[1] # products listing input
listing_out         = sys.argv[2] # output meals
init_qty            = float(sys.argv[3]) # number of meals for init purposes
delta_auth          = float(sys.argv[4]) # authorized delta between optimal and output
meal_size           = float(sys.argv[5]) # size of meal for one person (in grams)


# Read input listing
df_in = pd.read_csv(listing_in, sep = ",")

# Read categories mapping
mapping = pd.read_csv(cat_mapping_file, sep = ";")

# Add product category
df_in = df_in.merge(mapping, how = "inner", on = "pnns_groups_2")

# Remove non eligible products
df_in = df_in.loc[df_in['code_cat'] != 0,]

# Extract product quantity in grams
# to be done

# Build product dictionary: agregate by product with keeping names
pdt_info_df = df_in
pdt_info_df['QuantiteValeur'] = pdt_info_df.groupby(['EAN', 'code_cat', 'qty_gram'])['QuantiteValeur'] \
                                           .transform('sum')
pdt_info_df = pdt_info_df[['EAN', 'Nom', 'code_cat', 'pnns_groups_2', 'qty_gram', 'QuantiteValeur']] \
                         .drop_duplicates(['EAN', 'code_cat', 'qty_gram']) \
                         .reset_index(drop = True)

pdt_info_dict = pdt_info_df.to_dict()

# Model creation, decision variable Xi,j, meal i associated with product j?
hm = Model("HM")
meal_domain = np.arange(0, init_qty, step = 1) + 1
pdt_domain = pdt_info_df.index.tolist()
cat_domain = list(cat_distrib.keys())

# Xi,j are integers
x = hm.addVars(meal_domain, pdt_domain, vtype=GRB.INTEGER, name='x')


# Objective function: minimize surplus but could be maximizing nutriscore or something else
hm.setObjective(quicksum((pdt_info_dict['QuantiteValeur'][pdt] - x.sum('*', pdt)) \
                         for pdt in pdt_domain), \
                sense = GRB.MINIMIZE)


# Constraint: can't use more than available for each product
hm.addConstrs((x.sum('*', pdt) <= pdt_info_dict['QuantiteValeur'][pdt]\
               for pdt in pdt_domain))

# Constraint: can't oversize or undersize a category too much
hm.addConstrs(quicksum((x[meal, pdt] * pdt_info_dict['qty_gram'][pdt]) \
                       for pdt in pdt_domain if pdt_info_dict['code_cat'][pdt] == cat) \
              <= (cat_distrib[cat] + delta_auth) * meal_size\
              for cat in cat_domain \
              for meal in meal_domain)

hm.addConstrs(quicksum((x[meal, pdt] * pdt_info_dict['qty_gram'][pdt]) \
                       for pdt in pdt_domain if pdt_info_dict['code_cat'][pdt] == cat) \
              >= (cat_distrib[cat] - delta_auth) * meal_size\
              for cat in cat_domain \
              for meal in meal_domain)

hm.Params.TimeLimit       = time_limit
hm.Params.Threads         = 0
hm.Params.DisplayInterval = display_interval

hm.optimize()

status = hm.status
r = hm.status

res_dict = dict.fromkeys(meal_domain, [])
res_df = pd.DataFrame(columns = ('meal', 'pdt', 'qty'))

for meal in meal_domain:
    for pdt in pdt_domain:
        if x[meal, pdt].getAttr('x') >= 1:
            res_dict[meal] = res_dict[meal] + [[pdt, x[meal, pdt].getAttr('X')]]
            res_df = res_df.append({
                    'meal': meal,
                    'pdt': pdt,
                    'qty': x[meal, pdt].getAttr('x')
                    }, ignore_index = True)
            
# Adding product name and category
res_df = res_df.merge(pdt_info_df.reset_index(), left_on = 'pdt', right_on = 'index') \
               .sort_values('meal')
    
# Write results
res_df.to_csv(listing_out, sep = ";", index = False)
