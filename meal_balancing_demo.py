import numpy as np
import pandas as pd
from utils.off_mgt import convert_quantity
from sklearn.preprocessing import LabelEncoder
#from meal_balancer.algos_optimisation.milp_cvxpy_glpk import optimize_baskets, load_meal_balancing_parameters, load_category_mappings
from meal_balancer.algos_optimisation.milp_algo import *

__author__ = 'Aoife Fogarty'
__version__ = '0.1'
__maintainer__ = 'Aoife Fogarty'
__status__ = 'Development'


def main():

    # output of listing_categorization_demo.py
    filename = 'data/output/commande_94349.csv'
    # filename = 'data/output/commande_100P.csv'

    # # ideal distribution of products in each category 10, 20, etc.
    # # categories are listed in data/food_categories.csv
    # cat_distrib = {10: 0.5, 20: 0.5}

    # # authorised difference between real and ideal distribution
    # # TODO calibrate value
    # delta_auth = 0.05

    # # ideal total basket weight in grams
    # # TODO replace with real value
    # meal_size = 1000
    
    results_filename = 'data/output/results_sample.json'
    delta_auth = 0.5
    meal_weight = 5000
    print('delta_auth', delta_auth)
    print('meal_weight', meal_weight)

    distrib_filename = 'data/food_categories.csv'
    
    map_label2_code1, map_code1_label1 = load_category_mappings(distrib_filename)
    
    df_listing = pd.read_csv(filename, sep=';', decimal=',')

    # TODO chose between phenix and off grams
    df_listing['weight_grams'] = df_listing['phenix_grams'].astype(np.float64)
    
    # output of classifier is level 2 labels but constraints are for level 1 codes
    df_listing['codeAlim_1'] = df_listing['labelAlim_2'].apply(lambda x: map_label2_code1[x])
    df_listing['labelAlim_1'] = df_listing['codeAlim_1'].apply(lambda x: map_code1_label1[x])

    # filter out forbidden categories
    # TODO treat these better
    df_listing = df_listing[~df_listing.codeAlim_1.isin([0,70])]
    df_listing = df_listing[~pd.isnull(df_listing['weight_grams'])]
    
    cat_distrib, cat_mandatory, cat_status = load_meal_balancing_parameters(distrib_filename, df_listing)

    if cat_status == 1:
        result = optimize_baskets(df_listing, cat_distrib, cat_mandatory, delta_auth, 
                                  meal_weight, results_filename, solver='CBC')
    else:
        result = None

    print(result)


if __name__ == '__main__':
    main()
