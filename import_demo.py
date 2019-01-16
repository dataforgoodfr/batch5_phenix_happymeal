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

# import argparse
import pandas as pd
from io_mgt import listing_import as li
from preprocessing import categorization as ct
from utils.off_mgt import get_product_information, convert_to_gram, parse_qty_info

#parser = argparse.ArgumentParser()
#parser.add_argument("commande_id", type=str, help="Enter the commande_id you want to process")
#args = parser.parse_args()


listing_file          = 'commande_94349.csv'
mapping_file          = 'data/mapping_off_ideal.csv'
model_classifier_file = 'data/clf_nutrients_rf_groupeAlim_2_light.sav'
model_matching_file   = 'data/clf_names_nb_light.sav'


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
    input_listing = li.importListing(listing_file)

    # Add food category to the data frame
    input_listing = ct.get_foodGroupFromToDF(input_listing,
                                             EAN_col='EAN',
                                             product_name_col='Produit_Nom',
                                             mapping_file=mapping_file,
                                             model_classifier_file=model_classifier_file,
                                             model_matching_file=model_matching_file)
    print("category added")
    print(input_listing)
    print(input_listing.columns)
    input_listing['phenix_grams'] = input_listing.Produit_Nom.apply(lambda s, f=parse_phenix_grams: f(s))
    input_listing['off_grams'] = input_listing.EAN.apply(lambda s, f=get_off_grams: f(s))

    # write output file
    input_listing.to_csv("data/output/" + listing_file,
                         sep=';', encoding='UTF-8', index=False)

    # Ready now to extract quantity in grams


if __name__ == '__main__':
    main()
