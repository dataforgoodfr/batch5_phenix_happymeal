# -*- coding: utf-8 -*-
"""
Input: Food listing in csv format (e.g. output of Phenix platform)
Output: Previous listing + food category added in dataframe format
"""

__author__ = 'Julie Seguela'
__license__ = 'MIT License'
__version__ = '0.1'
__maintainer__ = 'Julie Seguela'
__status__ = 'Development'

import os
import argparse
import numpy as np
import pandas as pd

from io_mgt import listing_import as li
from preprocessing import categorization as ct
from utils.off_mgt import get_product_information, convert_to_gram, parse_qty_info
from meal_balancer.grouping import create_batches
from meal_balancer.balance_adjuster import missing_balance, tier_completion

parser = argparse.ArgumentParser()
parser.add_argument("order_id", type=int, help="Enter the order_id you want to process")  # 94349, 81708
parser.add_argument("--batch_qty", type=int, default=500, help="Enter the desired base basket size in grams")
args = parser.parse_args()


mapping_file = 'data/mapping_off_ideal.csv'
model_classifier_file = 'data/clf_nutrients_rf_groupeAlim_2_light.sav'
model_matching_file = 'data/clf_names_nb_light.sav'


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


def main():
    # Read food listing file
    input_listing = li.importListing(f"commande_{args.order_id}.csv")

    # Extract quantity in grams
    input_listing.drop(columns='#', inplace=True)
    input_listing['phenix_grams'] = input_listing.Produit_Nom.apply(lambda s, f=parse_phenix_grams: f(s))
    input_listing['off_grams'] = input_listing.EAN.apply(lambda s, f=get_off_grams: f(s))
    input_listing['weight_grams'] = input_listing.phenix_grams


    # Add food category to the data frame
    input_listing = ct.get_foodGroupFromToDF(input_listing,
                                             EAN_col='EAN',
                                             product_name_col='Produit_Nom',
                                             mapping_file=mapping_file,
                                             model_classifier_file=model_classifier_file,
                                             model_matching_file=model_matching_file,
                                             group_name='labelAlim_1')

    input_listing.columns = ['product_name', 'ean', 'unit_price', 'quantity', 'total_price', 'phenix_grams',
                             'off_grams', 'weight_grams', 'group_name', 'group_method']
    input_listing = input_listing[pd.notnull(input_listing.product_name)]
    float_cols = ['unit_price', 'total_price', 'quantity']
    input_listing[float_cols] = input_listing[float_cols].astype(float)
    input_listing.quantity = np.floor(input_listing.quantity)
    int_cols = ['ean', 'group_method', 'quantity']
    input_listing[int_cols] = input_listing[int_cols].astype(int)

    # Get ideal split
    ideal_split = pd.read_csv(os.path.join('data', 'food_categories.csv'), sep=';')
    ideal_split = ideal_split.drop_duplicates(['labelAlim_1', 'idealDistrib_1'])
    # cat_distrib = dict(zip(ideal_split['codeAlim_1'].values, ideal_split['idealDistrib_1'].values))
    ideal_split = pd.Series(ideal_split.idealDistrib_1.values, index=ideal_split.labelAlim_1).to_dict()

    # Get algo result
    selection = input_listing[pd.notnull(input_listing.weight_grams)].filter(
        items=['quantity', 'weight_grams', 'group_name'])
    items = []
    for idx, row in selection.iterrows():
        # Quantity
        items.extend([dict(category=row['group_name'], quantity=row['weight_grams'])
                      for ex in range(row['quantity'])])

    all_batches, storage, big_items, unidentified, tmp_items, loss = create_batches(items, ideal_split,
                                                                                    batch_qty=args.batch_qty,
                                                                                    overflow_thresh=0.5,
                                                                                    underflow_thresh=0.2)
    # Ready now to apply meal balancer
    req_portions = missing_balance(storage, ideal_split)
    # Prices for groupeAlim_2
    bare_prices = pd.read_csv(os.path.join('data', 'categ_prices.csv'))
    # Mapping
    mapping_groups = pd.read_csv(mapping_file, sep=';', encoding='UTF-8')
    categ_prices = pd.Series(bare_prices.loc[bare_prices.aggregation_level == 1, 'average_price'].values,
                             index=bare_prices.loc[bare_prices.aggregation_level == 1, 'category']).to_dict()

    # categ_prices = {cat: 1e-3 for cat in req_portions.keys()}
    res = tier_completion(req_portions, categ_prices, ideal_split, batch_qty=args.batch_qty)


    # Result printing
    print(f"Listing split: {input_listing.groupby('group_name').quantity.sum()}")
    print(f"Missing categories: {[elt for elt, val in ideal_split.items() if val > 0 and elt not in input_listing.group_name]}")
    print(f'Big items: {len(big_items)}')
    print(f'Unidentified: {len(unidentified)}')
    for tier in res:
        print(f"{tier.get('nb_meals')} meals at {tier.get('meal_price')} unit price (buying {tier.get('categ_bought')})")

    # write output file
    input_listing.to_csv(os.path.join('data', 'output', f"commande_{args.order_id}.csv"),
                         sep=';', encoding='UTF-8', index=False)


if __name__ == '__main__':
    main()
