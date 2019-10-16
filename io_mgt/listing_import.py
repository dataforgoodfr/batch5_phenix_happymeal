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
import numpy as np


def importListing(csv_file, 
                  ean_col=          'EAN',  
                  product_name_col= 'PRODUIT / LIBELLE',  
                  quantity_col=     'QUANTITE',
                  weight_col=       'POIDS UNITAIRE',
                  unit_price_col=   'VALEUR EN STOCK UNITAIRE (HT) ',
                  total_price_col=  'TOTAL (HT)'):
    
    # Read food listing file
    input_listing = pd.read_csv(csv_file, sep=';', encoding='UTF-8')
    
    # Checks required columns are present
    for col in [ean_col, product_name_col, quantity_col]:
        if pd.Series(col).isin(input_listing.columns).sum() < 1:
            sys.exit(col + ' is not in dataframe')
    
    # Rename columns
    input_listing.rename(columns={'EAN': 'ean',
                                  'PRODUIT / LIBELLE': 'product_name', 
                                  'QUANTITE': 'quantity',
                                  'VALEUR EN STOCK UNITAIRE (HT) ': 'unit_price',
                                  'TOTAL (HT)': 'total_price'}, 
                         inplace=True)
    
    # In the price version of the listing, there is a total row to remove
    input_listing = input_listing[pd.notnull(input_listing.product_name)]
    
    # drop useless columns
    useless_cols = '#'
    for col in useless_cols:
        if pd.Series(col).isin(input_listing.columns).sum() >= 1:
            input_listing.drop(columns=col, inplace=True)
    
    # If weight filled, we can use it
    if pd.Series(weight_col).isin(input_listing.columns).sum() >= 1:
        input_listing.rename(columns={'POIDS UNITAIRE': 'weight_grams'}, 
                             inplace=True)
        
    # convert numbers to floats
    float_cols = ['unit_price', 'total_price', 'quantity']
    for col in float_cols:
        if pd.Series(col).isin(input_listing.columns).sum() > 0:
            input_listing[col] = (input_listing[col]
                                  .str.replace(',', '.')
                                  .astype(float)
                                  )
    # convert ean_col to int type
    int_cols = 'ean'
    input_listing[int_cols] = input_listing[int_cols].apply(np.int64)

    return input_listing
