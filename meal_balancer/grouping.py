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

import numpy as np


class Pile:
    def __init__(self):
        self.items = []

    def size(self):
        return len(self.items)

    def is_empty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]


def evaluate_split(ideal_split, actual_split, multiplier=1):
    """
    Group items accordingly to an expected distribution
    Args:
        ideal_split (dict): list of item IDs
        actual_split (dict): dictionary mapping item IDs to a category
        multiplier (int): multiplier for ideal split
    Returns:
        norm (float): L2 distance between the two distributions
    """

    delta = [actual_split[cat] - (val * multiplier) for cat, val in ideal_split.items()]

    return np.linalg.norm(delta)


def evaluate_batch(ideal_split, actual_batch, multiplier=1):
    """
    Group items accordingly to an expected distribution
    Args:
        ideal_split (dict): ideal split of quantities by category
        actual_batch (list(dict)): list of items
        multiplier (int): multiplier for ideal split
    Returns:
        norm (float): L2 distance between the two distributions
    """

    actual_split = {cat: 0. for cat in ideal_split.keys()}
    for item in actual_batch:
        actual_split[item['category']] += item['quantity']

    return evaluate_split(ideal_split, actual_split, multiplier=multiplier)


def create_batches(items, balanced_split, batch_qty, overflow_thresh=0., underflow_thresh=0.):
    """
    Group items accordingly to an expected distribution
    Args:
        items (list(dict)): list of items {'category': cat, 'quantity': q}
        balanced_split (dict): expected split of quantity by category for a balanced batch
        batch_qty (float): desired quantity in each batch
        overflow_thresh (float): maximum overflowing quantity accepted
        underflow_thresh (float): minimum undeflowing quantity accepted
    Returns:
        all_batches (list): list of batches containing the indexes of items
        storage (dict): remaining categorized batches
        big_items (dict): items that were too big to be matched
        unidentified (dict): items whose categories couldn't be matched against the ideal distribution
        tmp_items (dict): items that could not form a batch
        loss (float): L2 loss of meal against ideal split
    """

    # theoretical_limit = int(sum([item['quantity'] for item in items]) / batch_qty)
    # print('Maximum number of balanced batches: %s' % theoretical_limit)

    # Check balanced split
    if round(sum(balanced_split.values()), 2) != 1.00:
        raise ValueError('A balanced split must have category proportions summing up to 1')

    # Expected quantities
    balanced_batch = {cat: q * batch_qty for cat, q in balanced_split.items()}
    # Overflowing items and temporary storage
    large_items, tmp_items, storage, unidentified = dict(), dict(), dict(), list()
    for cat in balanced_split.keys():
        large_items[cat] = Pile()
        tmp_items[cat] = []
        storage[cat] = Pile()

    # Looping
    for item_idx, item_info in enumerate(items):

        # Unidentified items
        if item_info['category'] not in balanced_batch:
            unidentified.append(item_idx)
            continue

        # Large items
        if item_info['quantity'] > balanced_batch.get(item_info['category']):
            if item_info['quantity'] > balanced_batch.get(item_info['category']) * (1.0 + overflow_thresh):
                # Append to large items
                large_items[item_info['category']].push(item_info)
            else:
                # Append to correct storage
                storage[item_info['category']].push([item_info])
        else:
            # Put in temporary storage
            tmp_items[item_info['category']].append(item_info)
            tmp_qty = sum([item['quantity'] for item in tmp_items[item_info['category']]])
            # Check if tmp_items has enough quantity of this category
            if tmp_qty > balanced_batch.get(item_info['category']) * (1.0 - underflow_thresh):
                storage[item_info['category']].push(tmp_items[item_info['category']])
                tmp_items[item_info['category']] = []

    # Balanced batches (storage, large_items, unidentified, tmp_items)
    nb_batches = min([p.size() for p in storage.values()])
    batches = [[item for cat in balanced_split.keys() for item in storage[cat].pop()] for idx in range(nb_batches)]

    # Multiple persons
    all_batches = {1: batches}
    big_items = []
    for cat in balanced_split.keys():
        for idx in range(large_items[cat].size()):
            batch_multiplier = int(round(large_items[cat].peek()['quantity'] / balanced_batch[cat]))
            # Check if we have enough to make bigger batch
            if all(p.size() >= batch_multiplier for other_cat, p in storage.items() if other_cat != cat):
                # Create the batch
                batch = [large_items[cat].pop()]
                for other_cat, p in storage.items():
                    if other_cat != cat:
                        for n in range(batch_multiplier):
                            batch.extend(p.pop())

                # Put it in all_batches
                if batch_multiplier not in all_batches:
                    all_batches[batch_multiplier] = []
                all_batches[batch_multiplier].append(batch)
            else:
                big_items.append(large_items[cat].pop())

    loss = sum([evaluate_batch(balanced_batch, b, multiplier=multiplier)
                for multiplier, batches_m in all_batches.items() for b in batches_m])

    return all_batches, storage, big_items, unidentified, tmp_items, loss
