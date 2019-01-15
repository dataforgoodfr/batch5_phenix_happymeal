import numpy as np
import pandas as pd
from utils.off_mgt import convert_quantity
from sklearn.preprocessing import LabelEncoder
from meal_balancer.algos_optimisation.milp_cvxpy_glpk import optimize_baskets, load_meal_balancing_parameters

__author__ = 'Aoife Fogarty'
__version__ = '0.1'
__maintainer__ = 'Aoife Fogarty'
__status__ = 'Development'


def main():

    # output of listing_categorization_demo.py
    filename = 'data/output/commande_94349.csv'

    # ideal distribution of products in each category 10, 20, etc.
    # categories are listed in data/food_categories.csv
    # TODO replace with real distribution
    cat_distrib = {10: 0.11, 20: 0.11, 30: 0.11, 40: 0.11, 50: 0.11, 60: 0.11, 70: 0.34}
    cat_distrib = {10: 0.5, 20: 0.5}

    n_categories = len(cat_distrib)

    # authorised difference between real and ideal distribution
    # TODO calibrate value
    delta_auth = 0.25

    # ideal total basket weight in grams
    # TODO replace with real value
    meal_size = 1400

    # filename = 'meal_balancing_parameters.json'
    distrib_filename = 'data/food_categories.csv'
    #cat_distrib, delta_auth, meal_size = load_meal_balancing_parameters(distrib_filename)

    df_listing = pd.read_csv(filename, sep=';', decimal=',')
    df_listing = df_listing.head(15)

    # TODO (temporary fix until we have the weights function) add random weights
    n_products = len(df_listing)
    max_item_weight = 200
    df_listing['weight_grams'] = np.random.random(size=n_products) * max_item_weight + 1.0
    df_listing['codeAlim_2'] = np.random.random_integers(low=1, high=n_categories, size=n_products) * 10

    result = optimize_baskets(df_listing, cat_distrib, delta_auth, meal_size, solver='GLPK')

    print(result)


if __name__ == '__main__':
    main()
