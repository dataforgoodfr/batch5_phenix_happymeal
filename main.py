# coding: utf-8

import json
import os
import argparse
import numpy as np
import pandas as pd

from io_mgt import listing_import as li
from utils.off_mgt import get_product_information, convert_to_gram, parse_qty_info
from preprocessing import categorization as ct
from meal_balancer.grouping import create_batches
from meal_balancer.balance_adjuster import missing_balance, tier_completion
from meal_balancer.algos_optimisation.milp_algo import *

mapping_file = 'data/mapping_off_ideal.csv'
model_classifier_file = 'data/clf_nutrients_rf_groupeAlim_2_light.sav'
model_matching_file = 'data/clf_names_nb_light.sav'
distrib_filename = 'data/food_categories.csv'


def parse_phenix_grams(product_name):
    phenix_qty = None
    # If several results, take the last one as weight information is usually at the end
    parsing_results = parse_qty_info(product_name)
    if len(parsing_results) > 0:
        selected_res = parsing_results[-1]
        converted_res = convert_to_gram(selected_res['value'], selected_res['unit'])
        # Successful conversion
        if converted_res['std']:
            phenix_qty = converted_res['val']

    return phenix_qty


def get_off_grams(ean):
    off_qty = None
    if pd.notnull(ean):
        # If several results, take the last one as weight information is usually at the end
        off_result = get_product_information(str(int(ean)))
        if isinstance(off_result, dict):
            off_dict = off_result.get('quantity')
            if isinstance(off_dict, dict) and off_dict.get('unit') == 'g':
                off_qty = convert_to_gram(off_dict.get('val'), off_dict.get('unit')).get('val')

    return off_qty

# python main.py --meal-weight 5000 --delta-auth 0.5 --csv-file-path commande_94349.csv --request-id 123
# python main.py --meal-weight 5000 --delta-auth 0.5 --csv-file-path commande_94349_price.csv --request-id 124

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meal-weight", help="meal weight in grams: 2000", type=float, required=True)
    parser.add_argument("--delta-auth", help="authorised difference between real and ideal distribution: 0.05", type=float, required=True)
    parser.add_argument("--csv-file-path", help="path to csv file: '/home/ubuntu/input/listing.csv'", type=str, required=True)
    parser.add_argument("--request-id", help="unique request id: 123456789", type=int, required=True)
    args = parser.parse_args()

    # access variables via args.meal_weight, args.csv_file_path etc.
    request_id = args.request_id
    
    results_filename = 'output/result_{}.json'.format(request_id)
    
    group_name_level = 'labelAlim_1'
    
     # Read food listing file
    input_listing = li.importListing(args.csv_file_path)

    # Extract quantity in grams
    input_listing['phenix_grams'] = (input_listing
                                     .product_name
                                     .apply(lambda s, f=parse_phenix_grams: f(s))
                                     )
    input_listing['off_grams'] = (input_listing
                                 .ean
                                 .apply(lambda s, f=get_off_grams: f(s))
                                 )
    
    # if phenix_grams not filled we try to fill with off_grams
    input_listing['weight_grams'] = input_listing.phenix_grams
    input_listing.loc[input_listing.weight_grams.isnull(), 'weight_grams'] = \
        input_listing.loc[input_listing.weight_grams.isnull(), 'off_grams']


    # Add food category to the data frame
    input_listing = ct.get_foodGroupFromToDF(input_listing,
                                             EAN_col='ean',
                                             product_name_col='product_name',
                                             mapping_file=mapping_file,
                                             model_classifier_file=model_classifier_file,
                                             model_matching_file=model_matching_file,
                                             group_name=group_name_level)

    input_listing.rename(columns = {group_name_level: 'group_name'}, inplace = True)
            
   # input_listing.quantity = np.floor(input_listing.quantity)
   # input_listing['quantity'] = input_listing['quantity'].astype(int)
    
    map_label2_code1, map_code1_label1, map_label1_code1 = load_category_mappings(distrib_filename)
    
    # bring back the level 1 codes
    input_listing['group_code'] = input_listing['group_name'].apply(lambda x: map_label1_code1[x])

    # filter out forbidden categories
    # TODO treat these better
    input_listing = input_listing[~input_listing.group_code.isin([0,70])]
    input_listing = input_listing[~pd.isnull(input_listing['weight_grams'])]
    
    cat_distrib, cat_mandatory, cat_status = load_meal_balancing_parameters(distrib_filename, 
                                                                            input_listing,
                                                                            'group_code')

    if cat_status == 1:
        result = optimize_baskets(input_listing, 
                                  cat_distrib, 
                                  cat_mandatory, 
                                  args.delta_auth, 
                                  args.meal_weight, 
                                  results_filename, 
                                  solver='CBC')
        print('Optimization done')
    else:
        result = None
        print('No meal can be found')


if __name__ == '__main__':
    main()
