#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Product grouping for balanced meal creation
'''

__author__ = 'François-Guillaume Fernandez'
__license__ = 'MIT License'
__version__ = '0.1'
__maintainer__ = 'François-Guillaume Fernandez'
__status__ = 'Development'

import pandas as pd


def extract_qty_info(s, sep=','):

    split = [elt.strip() for elt in s.split(sep) if elt.strip()[0].isdigit()]
    qty = None
    if len(split) == 1:
        qty = split[0]

    return qty


def parse_listing(csv_path):

    raw_df = pd.read_csv(csv_path, sep=';', encoding='latin', usecols=['PRODUIT / LIBELLE', 'QUANTITE'])
    raw_df.rename(columns={'PRODUIT / LIBELLE': 'product_info', 'QUANTITE': 'quantity'}, inplace=True)
    df = raw_df.iloc[::2].reset_index(drop=True)
    df.product_info = df.product_info.apply(lambda s: s.split('/')[0].strip())
    df['EAN'] = raw_df.iloc[1::2].reset_index(drop=True).product_info.apply(lambda s: s.split(' : ')[-1]).astype(int)
    df.quantity = df.quantity.apply(lambda s: s.split()[0].replace(',', '.')).astype(float)
    df['product_qty'] = df.product_info.apply(lambda s: extract_qty_info(s))

    return df
