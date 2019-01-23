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

GRAM_CONVERSION_TABLE = dict(g=1.0, gr=1.0, kg=1000., mg=1e-3, gal=3785.41, egg=50.0, portion=100.,
                             l=1000., ml=1., cl=10., dl=100., oz=28.3495, lb=453.592)

solid_units = ['g', 'gr', 'mg', 'kg', 'lb', 'pound', 'oz']
liquid_units = ['l', 'litre', 'ml', 'cl', 'gal', 'gallon']
ALL_UNITS = solid_units + liquid_units


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
    Convert OFF quantity to dictionary of format
    {'val': float, 'unit': float, 'std': bool, 'approx': bool }

    'std' is True if value could be converted to g using a standard conversion factor
    'approx' is True if the original units were not in 'g', 'kg', 'mg'

    For strings that contain several quantities, 
    returns the most precise quantity (g, kg, mg > everything else)
    and then the highest quantity 
    e.g. "130 g (4 tranches de 32,5 g)" => 130 g
    "230 g, dont 30 g d'accompagnement, en 2 portions" => 230 g

    Args:
        str_qty (str): OFF quantity extract 
    Returns:
        dict_qty (dict)
    """

    if str_qty is None:
        return None
    assert isinstance(str_qty, str)
    dict_qty = {'val': None, 'unit': None, 'std': False, 'approx': True}

    if type(str_qty) == float and math.isnan(str_qty):
        return dict_qty

    # remove all stray 'e' in OFF quantity strings
    # regex matches whitespace followed by 'e' followed by any of whitespace, EOS, non-word character
    # only remove the 'e' and keep all other characters
    str_qty = re.sub('(\s+)(?:e)(\s|$|\W)', r'\1\2', str_qty)

    # split string into individual pieces of information
    # e.g. "400 g, 2 sachets" => ["400 g", "2 sachets"]
    pattern = ',\s+|\s+,|\(|\)|soit'
    try:
        lst_qty = re.split(pattern, str_qty)
    except:
        lst_qty = ['']

    lst_dict = []
    for s in lst_qty:
        if (s is None) or (s == ''):
            continue
        string = s.strip()

        # extract quantities with multiplier (e.g. 4x25g)
        # Regex matches integer followed by x or *, followed by integer
        # or decimal (value), and the following letters (unit)
        # up until whitespace or non-word character
        pattern = '(\d+)\s*[\*x]\s*(\d+[,.]?\d*)\s*([a-zA-Z]+)(?:\s|]|\W)*'
        m = re.search(pattern, string, re.UNICODE)
        if m is not None:
            multiplier = int(m.group(1))
            value = float(m.group(2).replace(',','.')) * multiplier
            unit = rename_qty2stdunit(m.group(3))
            # dict_qty = convert_qty2gram({'val': value, 'unit': unit})
            dict_qty = convert_to_gram(value, unit)
            lst_dict.append(dict_qty)
            continue
       
        # If no quantity with multiplier was found, try to find one without multiplier
        # Regex matches integer or decimal (value), and the following letters (unit)
        # up until whitespace or non-word character
        pattern = '(\d+[,.]?\d*)\s*([a-zA-Z]+)(?:\s|]|\W)*'
        m = re.search(pattern, string, re.UNICODE)
        if m is not None:
            value = float(m.group(1).replace(',','.'))
            unit = rename_qty2stdunit(m.group(2))
            # dict_qty = convert_qty2gram({'val': value, 'unit': unit})
            dict_qty = convert_to_gram(value, unit)
            lst_dict.append(dict_qty)
            continue

        print(s, 'not matched by regex')

    if len(lst_dict) == 0:
        return dict_qty

    elif len(lst_dict) == 1:
        return lst_dict[0]

    else:
        df = pd.DataFrame(lst_dict) 
        # Return the most precise quantity (g, kg, mg > everything else),
        # then the heaviest quantity.
        # Precision is more important than weight.
        # Only return non-converted quantity if no converted quantity
        # is available
        df = df.sort_values(by=['std', 'approx', 'val'], ascending=[False, True, False])

        return df.iloc[0].to_dict()


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


def convert_to_gram(value, unit):
    """
    Convert OFF quantity to a standard quantity (in grams)
    Args:
        value (float): quantity value
        unit (str): quantity unit
    Returns:
        qty_dict (dict): a dictionary including the value, unit, success, and approximation information
    """

    res = dict(val=value, unit=unit, std=False, approx=True)
    if isinstance(value, float) and isinstance(unit, str):
        gr_multiplier = GRAM_CONVERSION_TABLE.get(unit)
        if isinstance(gr_multiplier, float):
            res = dict(val=value * gr_multiplier, unit='g', std=True, approx=(unit in ['g', 'kg', 'mg']))

    return res


def parse_qty_info(product_name):
    """
    Parse quantity information from the product name
    Args:
        product_name (str): name of the product to parse
    Returns:
        results (list): list of dictionaries containing both value and unit information
    """

    results = []
    if isinstance(product_name, str):

        str_groups = re.split(' |,|, ', product_name)
        # Locate groups with digits
        skip_next_group = False
        for group_pos, str_group in enumerate(str_groups):

            if skip_next_group:
                skip_next_group = False
                continue

            if any(char.isdigit() for char in str_group):
                # Check if unit available in the same group or the next
                tmp_val, candidate_units = '', []
                multiplier = None
                figure_stored, unit_found = False, False
                for pos, char in enumerate(str_group):

                    # Acquire all digits
                    if char.isdigit() or (char == '.' and figure_stored):
                        tmp_val += char
                        figure_stored = True
                    elif figure_stored:
                        # Unit right next to it in the group
                        if any(unit == str_group[pos: pos + len(unit)].lower() for unit in ALL_UNITS):
                            unit_found = True
                            candidate_units.extend([tmp_unit for tmp_unit in ALL_UNITS if tmp_unit == str_group[pos: pos + len(tmp_unit)].lower()])
                            break
                        else:
                            # Portion multiplier
                            if char.lower() == 'x':
                                multiplier = int(tmp_val)
                            figure_stored = False
                            tmp_val = ''

                else:
                    # Unit in the beginning of next group
                    if figure_stored and not unit_found and group_pos + 1 < len(str_groups):
                        tmp_list = [tmp_unit for tmp_unit in ALL_UNITS if tmp_unit == str_groups[group_pos + 1][: len(tmp_unit)].lower()]
                        if len(tmp_list) > 0:
                            unit_found, skip_next_group = True, True
                            candidate_units.extend(tmp_list)

                # Check what we got
                if unit_found:
                    # Take longest matching unit
                    unit = ''
                    for tmp_unit in candidate_units:
                        if len(tmp_unit) > len(unit):
                            unit = tmp_unit
                    if multiplier:
                        tmp_val = multiplier * float(tmp_val)
                    if tmp_val == '':
                        tmp_val = 'NaN'
                    results.append(dict(value=float(tmp_val), unit=unit))
                    break

    return results


def convert_qty2gram(qty = {'val': '', 'unit': ''}):
    """
    Convert OFF quantity to a standard quantity (in grams)
    Args:
        qty (dict): OFF quantity value and unit
    Returns:
        dict with value converted to grams (if possible) and
        two new keys:
           std: True if value could be converted using a standard 
                conversion factor
           approx: True if the original units were not in 'g', 'kg', 'mg'
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

    # all conversions not from g, kg or mg are approximate conversions
    approx = True
    if init_unit in ['g', 'kg', 'mg']:
        approx = False

    return {'val': conv_val, 'unit': conv_unit, 'std': conv_std, 'approx': approx}


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
