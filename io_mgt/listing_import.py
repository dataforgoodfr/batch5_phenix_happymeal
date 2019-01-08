#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Import and validation of listing file
'''

__author__ = 'Julie Seguela'
__license__ = 'MIT License'
__version__ = '0.1'
__maintainer__ = 'Julie Seguela'
__status__ = 'Development'

import sys
import pandas as pd

listing_file = 'commande_94349.csv'


def importListing(csv_file, 
                  EAN_col = 'EAN',  
                  product_name_col = 'PRODUIT / LIBELLE',  
                  quantity_col = 'QUANTITE',
                  weight_col = 'POIDS UNITAIRE'):
    
    # Read food listing file
    input_listing = pd.read_csv('data/'+ csv_file, sep=';', encoding='UTF-8')
    
    # Checks required columns are present
    for col in [EAN_col, product_name_col, quantity_col]:
        if pd.Series(col).isin(input_listing.columns).sum() < 1:
            sys.exit(col + ' is not in dataframe')
    
    # Rename columns
    input_listing.rename(columns={'PRODUIT / LIBELLE': 'Produit_Nom', 
                                  'QUANTITE': 'quantity'}, 
                         inplace=True)
    
    # If weight filled, we can use it
    if pd.Series(weight_col).isin(input_listing.columns).sum() >= 1:
        input_listing.rename(columns={'POIDS UNITAIRE': 'weight_grams'}, 
                             inplace=True)

    return input_listing
