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

import openfoodfacts
import math
import re
import pandas as pd


def get_product_information(barcode):
    """
    Retrieve product information from OpenFoodFacts database
    Args:
        barcode (str): barcode of the product
    Returns:
        product_information (dict): nested dictionary with product information
    """

    product_information = None
    res = openfoodfacts.products.get_product(str(barcode))
    if res['status']:
        p = res.get('product')
        product_information = dict(product_name=p.get('product_name'),
                                   categories=p.get('categories'),
                                   group=p.get('pnns_groups_1'),
                                   subgroup=p.get('pnns_groups_2'),
                                   food_group=rename_group(p.get('pnns_groups_2')),
                                   brands=p.get('brands'),
                                   stores=p.get('stores'),
                                   image_url=p.get('image_url'),
                                   packaging=p.get('packaging'),
                                   allergens=p.get('allergens'),
                                   traces=p.get('traces'),
                                   ingredients=p.get('ingredients'),
                                   quantity=convert_quantity(p.get('quantity')),
                                   nutriments=p.get('nutriments'),
                                   nutri_score=p.get('nutrition_grade_fr'))
    return product_information


def missing_balance(unbalanced_items, actual_split):
    """
    Get the required portions to complete balanced meals
    Args:
        unbalanced_items (dict): dictionary item_categ --> list of balanced portions
        actual_split (dict): dictionary item_categ --> ideal individual meal split 
    Returns:
        required_items (dict): dictionary item categ --> missing quantity
    """

    # Determine the category availability
    available_portions = {item_categ: len(unbalanced_items.get(item_categ, [])) for item_categ in actual_split}
    max_portions = max(available_portions.values())

    # Determine what we still need
    required_portions = {item_categ: max_portions - categ_count for item_categ, categ_count in available_portions.items()}

    return required_portions

def order_completion(required_portions)

