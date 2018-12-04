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

import argparse
import numpy as np
from meal_balancer.grouping import create_batches

cat_distrib = dict(A=0.05, B=0.3, C=0.5, D=0.15)

parser = argparse.ArgumentParser()
# Simulation-related
parser.add_argument('nb_samples', type=int, help='Number of random samples')
parser.add_argument('--item_max_qty', type=float, default=100., help='Maximum quantity generated for a single item')
# Processing-related
parser.add_argument('--portion_size', type=float, default=500., help='Quantity for a single individual')
parser.add_argument('--overflow_thresh', type=float, default=0.2, help='Maximum overflowing quantity accepted')
parser.add_argument('--underflow_thresh', type=float, default=0.1, help='Minimum undeflowing quantity accepted')
args = parser.parse_args()


def main():

    # Sample generation
    np.random.seed(42)
    prob = np.random.rand(args.nb_samples) * len(cat_distrib)
    qty = np.random.rand(args.nb_samples) * args.item_max_qty
    items = [{'category': ['A', 'B', 'C', 'D'][int(prob[idx])], 'quantity': qty[idx]}
             for idx in range(args.nb_samples)]

    all_batches, storage, big_items, unidentified, tmp_items, loss = create_batches(items, cat_distrib,
                                                                                    batch_qty=args.portion_size,
                                                                                    overflow_thresh=args.overflow_thresh,
                                                                                    underflow_thresh=args.underflow_thresh)

    print('------------\nRESULT\n------------')
    print('\n'.join(['%s batches for %s persons (portion of %s): %s items' % (len(batches), nb_persons, 500., sum([len(b) for b in batches]))
                     for nb_persons, batches in all_batches.items()]))
    print('Average batch loss: %s' % (loss / float(sum([len(b) for b in all_batches.values()]))))
    print('Number of remaining items: %s portioned, %s unportioned' % (sum([len(b) for p in storage.values() for b in p.items]), sum([len(l) for l in tmp_items.values()])))
    print('Number of large items: %s' % len(big_items))
    print('Number of unindentified items: %s' % len(unidentified))


if __name__ == '__main__':
    main()
