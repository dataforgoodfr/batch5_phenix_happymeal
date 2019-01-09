from utils.off_mgt import convert_quantity
from sklearn.preprocessing import LabelEncoder
from milp_cvxpy_glpk import optimize_baskets

__author__ = 'Aoife Fogarty'
__version__ = '0.1'
__maintainer__ = 'Aoife Fogarty'
__status__ = 'Development'


def main():

    # ideal distribution of products in each category 10, 20, etc.
    # categories are listed in data/food_categories.csv
    # TODO replace with real distribution
    cat_distrib = {10: 0.11, 20: 0.11, 30: 0.11, 40: 0.11, 50: 0.11, 60: 0.11, 70: 0.34}

    n_categories = len(cat_distrib)

    # authorised difference between real and ideal distribution
    # TODO calibrate value
    delta_auth = 0.05

    # ideal total basket weight in grams
    # TODO replace with real value
    meal_size = 1400

    # filename = 'meal_balancing_parameters.json'
    # cat_distrib, delta_auth, meal_size = load_meal_balancing_parameters(filename)

    df_listing = pd.read_csv('test_input.csv')

    result = optimize_baskets(df_listing, cat_distrib, delta_auth, meal_size)

    print(result)


if __name__ == '__main__':
    main()
