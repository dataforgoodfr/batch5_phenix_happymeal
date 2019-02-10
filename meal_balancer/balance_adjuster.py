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

from meal_balancer.grouping import Pile


def missing_balance(unbalanced_items, split_categ):
    """
    Get the required portions to complete balanced meals
    Args:
        unbalanced_items (dict): dictionary item_categ --> list of balanced portions
        split_categ (list): dictionary item_categ --> ideal individual meal split 
    Returns:
        required_items (dict): dictionary item categ --> missing quantity
    """

    # Determine the category availability
    available_portions = {item_categ: unbalanced_items.get(item_categ, Pile()).size()
                          for item_categ, val in split_categ.items() if val > 0}
    max_portions = max(available_portions.values())

    # Determine what we still need
    required_portions = {item_categ: max_portions - categ_count
                         for item_categ, categ_count in available_portions.items()}

    return required_portions


def tier_completion(required_portions, categ_prices, ideal_split, batch_qty=500):
    """
    Complete meals by spending-tier
    Args:
        required_portions (dict): dictionary item_categ --> list of balanced portions
        categ_prices (dict): dictionary item_categ --> average price per kilogram
        batch_qty (float): size of the batch (grams)
    Returns:
        required_items (dict): dictionary item categ --> missing quantity
    """

    meal_prices, nb_meals = [], []

    # Categories to buy
    tmp_dict = {key: val for key, val in required_portions.items()}

    # Sort by the number of portions required
    tmp_vals = sorted(list(set(tmp_dict.values())), reverse=True)
    # Get a list of each categ missing by tier
    buying_categs = [[]] * (len(tmp_vals) - 1)
    for categ, categ_need in tmp_dict.items():
        if categ_need > 0:
            buying_categs[tmp_vals.index(categ_need)].append(categ)
    # Loop on increasingly needed categ
    running_categs = []
    for idx, categs in enumerate(buying_categs):
        running_categs.extend(categs)
        # get the price per extra meal
        meal_prices.append(batch_qty / 1000 * sum([ideal_split[cat] * categ_prices[cat] for cat in running_categs]))
        nb_meals.append(tmp_vals[idx] - tmp_vals[idx + 1])

    res = [dict(nb_meals=tier_meals, meal_price=meal_prices[idx], categ_bought=buying_categs[idx])
           for idx, tier_meals in enumerate(nb_meals)]

    return res
