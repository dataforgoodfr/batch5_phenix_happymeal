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

def convert_quantity(str_qty):
    """
    Convert OFF quantity to [value, unit]
    Args:
        str_qty (str): OFF quantity extract (first number found)
    Returns:
        dict_qty (dict): nested dictionary with product quantity
    """
    import re
    value = None
    unit = None
    dict_qty = {'val': value, 'unit': unit}
    
    try:
        lst_qty = str_qty.split(',')
    except:
        lst_qty = ['','']
        
    lst_dict = []
    for s in lst_qty:
        string = s.strip()
        value = float(re.search('\d+[,.]?\d*',string).group(0)) if re.search('\d+[,.]?\d*',string)!=None else ''
        unit = re.search('([a-zA-Z]+)',string, re.UNICODE).group(0) if re.search('([a-zA-Z]+)',string, re.UNICODE)!=None else ''
        unit = rename_qty2stdunit(unit)
        dict_qty = convert_qty2gram({'val': value, 'unit': unit})
        return dict_qty
    
    return dict_qty

def rename_qty2stdunit(unit):
    """
    Clean OFF quantity unit to a standard unit name (g, kg, mg, gal, egg, portion, l...)
    Args:
        unit (str): OFF quantity unit
    Returns:
        std_unit (str): standard unit name
    """
    clean_matrix = {None:[None,''],
                    'g':['g','gr','grammes','G','grams','gramme','grs','Grammes',
                         'gram','gramm','grames','GR','gms','gm','grammi',
                         'grm','gramos','gammes','Grs','gramas','Gramos',
                         'grme','Gramm','gra','Gr','grms','ghr','gfd'],
                    'kg':['kg','Kg','KG','kgs','kilogrammae','klg','Kilogramm','kgi',
                         'kgr','kilos','kilo'],
                    'mg':['mg','mcg','mG','Mg','MG'],
                    'gal':['gal','gallon','GAL','Gal','GALLON','Gallon'],
                    'egg':['egg','eggs','Eggs','huevos','oeufs','Oeufs','ufs'],
                    'portion':['portion','servings','Servings','Serving',
                         'Unidad', 'triangles','beignets','galettes','baguettines',
                         'oranges','magret','baguette','Galettes','courge',
                         'galetttes','meringues','galetted','baguettes',
                         'Burger','gaufrettes','mangue','yogourts','gaufres',
                         'Gaufres','burgers','galletas','hamburguesas','vegano',
                         'fromage','mignonnettes','Portionsfilets','avocats',
                         'Fruit','fruits','fruit','portions','filets'],
                    'l':['l','L','litre','litres','Litre','Litres','Liters','liter',
                         'litro','Liter'],
                    'ml':['ml','mL','ML','Ml'],
                    'cl':['cl','cL','CL','Cl'],
                    'dl':['dl','dL','DL','Dl'],
                    'oz':['oz','OZ','Oz','oZ'],
                    'lb':['lb','LB','Lb','lB','lbs']}
    std_unit = [key  for (key, value) in clean_matrix.items() if (unit in value)]
    std_unit = [unit] if not std_unit else std_unit
    return std_unit[0]

def convert_qty2gram(qty = {'val': '', 'unit': ''}):
    """
    Convert OFF quantity to a standard quantity (in grams)
    Args:
        unit (dict): OFF quantity value
    Returns:
        std_unit (str): standard unit name
    """
    init_val = qty['val']
    init_unit = qty['unit']
    
    convert_matrix = {'g':1.0,
                      'kg':1000,
                      'mg':0.001,
                      'gal':3785.41,
                      'egg':50.0,              # TO BE VALIDATED
                      'portion':100.0,         # TO BE VALIDATED
                      'l':1000.0,
                      'ml':1.0,
                      'cl':10.0,
                      'dl':100.0,
                      'oz':28.3495,
                      'lb':453.592}
    if (init_val!='') & (init_unit in convert_matrix.keys()):
        conv_val = convert_matrix[init_unit]*init_val
        conv_unit = 'g'
        conv_std = True
    else:
        conv_val = init_val
        conv_unit = init_unit
        conv_std = False
    return {'val': conv_val, 'unit': conv_unit, 'std':conv_std}

def rename_group(str_group2=None):
    """
    Rename OFF food group (pnns_group_2) to a standard name
    Args:
        str_group2 (str): OFF food group name
    Returns:
        conv_group (str): standard food group name 
    """
    #convert_group1 = {'Beverage':['Beverages'],
    #                  'Cereals':['Cereals and potatoes'],
    #                  'Meal':['Composite foods'],
    #                  'Fat':['Fat and sauces'],
    #                  'Meat':['Fish Meat Eggs'],
    #                  'Fruits and vegetables':['Fruits and vegetables','fruits-and-vegetables'],
    #                  'Dairy':['Milk and dairy products'],
    #                  'Snack':['Salty snacks','Sugary snacks','sugary-snacks'],
    #                  None:[None,'unknown','']}
    convert_group2 = {'Beverage':['Alcoholic beverages','Artificially sweetened beverages',
                                  'Fruit juices','Fruit nectars','Non-sugared beverages',
                                  'Sweetened beverages'],
                      'Cereals':['Bread','Breakfast cereals','Cereals','Legumes','Patatoes'],
                      'Meal':['One-dish meals','Pizza pies and quiche','Sandwich'],
                      'Fat':['Dressings and sauces','Fats'],
                      'Meat':['Tripe dishes','Eggs','Fish and seafood','Meat','Processed meat','Nuts'],
                      'Fruit':['Fruits','fruits','Dried fruits'],
                      'Vegetable':['Soups','Vegetables','vegetables'],
                      'Dairy':['Cheese','Dairy desserts','Ice cream','Milk and yogurt'],
                      'Snack':['Appetizers','Salty and fatty products','Biscuits and cakes',
                               'Chocolate products','Sweets','pastries'],
                      None:[None,'unknown','']}
    conv_group = [key for (key, value) in convert_group2.items() if (str_group2 in value)]
    conv_group = [None] if not conv_group else conv_group
    return conv_group[0]