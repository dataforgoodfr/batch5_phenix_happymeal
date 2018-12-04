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
from io_mgt.listing_parser import parse_listing


parser = argparse.ArgumentParser()
# Simulation-related
parser.add_argument('csv_file', type=str, help='The location of the CSV file to parse')
args = parser.parse_args()


def main():

    df = parse_listing(args.csv_file)

    print(df)


if __name__ == '__main__':
    main()
