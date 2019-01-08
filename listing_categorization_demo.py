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

import argparse
from pprint import pprint
import pandas as pd
from io_mgt import listing_import as li
from preprocessing import categorization as ct



#parser = argparse.ArgumentParser()
#parser.add_argument("commande_id", type=str, help="Enter the commande_id you want to process")
#args = parser.parse_args()


listing_file          = 'commande_94349.csv'
mapping_file          = 'data/mapping_off_ideal.csv'
model_classifier_file = 'data/clf_nutrients_rf_groupeAlim_2_light.sav'
model_matching_file   = 'data/clf_names_nb_light.sav'


def main():
    # Read food listing file
    input_listing = li.importListing(listing_file)

    # Add food category to the data frame
    input_listing = ct.get_foodGroupFromToDF(input_listing, 
                                             EAN_col = 'EAN',
                                             product_name_col= 'Produit_Nom',
                                             mapping_file = mapping_file,
                                             model_classifier_file = model_classifier_file,
                                             model_matching_file= model_matching_file)
    print(input_listing)
    
    # write output file
    input_listing.to_csv("data/output/"+ listing_file, 
                         sep = ';', encoding = 'UTF-8', index = False)
    
    # Ready now to extract quantity in grams
    
if __name__ == '__main__':
    main()