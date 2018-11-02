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
                                   brands=p.get('brands'),
                                   stores=p.get('stores'),
                                   image_url=p.get('image_url'),
                                   packaging=p.get('packaging'),
                                   allergens=p.get('allergens'),
                                   traces=p.get('traces'),
                                   ingredients=p.get('ingredients'),
                                   quantity=p.get('quantity'),
                                   nutriments=p.get('nutriments'))
    return product_information
