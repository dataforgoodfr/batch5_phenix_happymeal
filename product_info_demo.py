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
from off_mgt import get_product_information
from pprint import pprint


parser = argparse.ArgumentParser()
parser.add_argument("barcode", type=str, help="Enter the barcode you want to analyze")
args = parser.parse_args()



def main():
    # Retrieve result
    product_information = get_product_information(args.barcode)
    pprint(product_information)


if __name__ == '__main__':
    main()
