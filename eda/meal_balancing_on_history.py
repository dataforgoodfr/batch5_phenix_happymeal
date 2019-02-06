# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 18:21:05 2019

@author: Julie Seguela
"""

import numpy as np
import pandas as pd
from meal_balancer.algos_optimisation.milp_algo import *

years = ['2018']
delta_auth = 0.2
meal_weight = 1400
print('delta_auth', delta_auth)
print('meal_weight', meal_weight)

filename = 'data/intermediate/df_algo.csv'
distrib_filename = 'data/food_categories.csv'
json_filename = 'data/output/results_sample.json'
res_path = 'data/output/'

map_label2_code1, map_code1_label1 = load_category_mappings(distrib_filename)
    

histo = pd.read_csv(filename, sep='\t', decimal='.', index_col = 0)
histo.rename(columns = {'P_Nom'       : 'Produit_Nom',
                        'P_food_group': 'labelAlim_2',
                        'CP_QuantiteTotale': 'quantity',
                        'Qty_val': 'weight_grams'}, inplace = True)
histo['EAN'] = 0

# Reduce to 2018 year
histo = histo.loc[histo['CO_DateCommande'].str.slice(0,4).isin(years), ]
distinct_id  = histo[['RC_Id', 'CO_DateCommande']].drop_duplicates(['RC_Id', 'CO_DateCommande'])

# Init output 
res_df = pd.DataFrame(columns = ('Produit_Nom', 'EAN', 'labelAlim_1', 'quantity', 
                                 'weight_grams', 'allocated_basket',
                                 'RC_Id', 'CO_DateCommande'))

for i in range(len(distinct_id)):
    
    print('step ' + str(i))
    print('Date commande :' + str(distinct_id.iloc[i, 1]))
    print('Recepteur :' + str(distinct_id.iloc[i, 0]))
    
    # Create df for one day and one receptor
    df_temp = histo.loc[(histo['RC_Id']==distinct_id.iloc[i, 0]) & 
                        (histo['CO_DateCommande']==distinct_id.iloc[i, 1]), ]
    
    # filter out forbidden categories
    df_temp = df_temp.loc[~(df_temp.labelAlim_2.isin(['Exclus','Plats préparés']) | 
                            df_temp.labelAlim_2.isnull()), ]
    df_temp = df_temp[~pd.isnull(df_temp['weight_grams'])]
    
    #### Meal balancing
    
    # output of classifier is level 2 labels but constraints are for level 1 codes
    df_temp['codeAlim_1'] = df_temp['labelAlim_2'].apply(lambda x: map_label2_code1[x])
    df_temp['labelAlim_1'] = df_temp['codeAlim_1'].apply(lambda x: map_code1_label1[x])
    df_temp['allocated_basket'] = 0

    df_temp = df_temp[['Produit_Nom', 'EAN', 'codeAlim_1', 'labelAlim_1', 'quantity', 
                       'weight_grams', 'allocated_basket',
                       'RC_Id', 'CO_DateCommande']]
    
    df_temp_algo = df_temp.copy()

    cat_distrib, cat_mandatory, cat_status = load_meal_balancing_parameters(distrib_filename, df_temp)

    if cat_status == 1:
        result = optimize_baskets(df_temp_algo, cat_distrib, cat_mandatory, delta_auth, 
                                  meal_weight, json_filename, solver='CBC')
        if result[0] is None:
            res_temp = df_temp.drop(columns = 'codeAlim_1')
        else:
            res_temp = result[0]
            res_temp['RC_Id'] = distinct_id.iloc[i, 0]
            res_temp['CO_DateCommande'] = distinct_id.iloc[i, 1]
    else:
        res_temp = df_temp.drop(columns = 'codeAlim_1')
        
    res_df = res_df.append(res_temp, ignore_index = True)    
    res_df.to_pickle(res_path + 'results_histo_mw' + str(meal_weight) \
                     + '_da' + str(delta_auth) + '_y' + '_'.join(years) + '.pkl')    

# Export csv
res_df.to_csv(res_path + 'results_histo_mw' + str(meal_weight) \
              + '_da' + str(delta_auth) + '_y' + '_'.join(years) + '.csv')
