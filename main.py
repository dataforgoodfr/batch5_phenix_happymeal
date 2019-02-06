# coding: utf-8
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meal-weight", help="meal weight in grams: 2000", type=float, required=True)
    parser.add_argument("--delta-auth", help="authorised difference between real and ideal distribution: 0.05", type=float, required=True)
    parser.add_argument("--csv-file-path", help="path to csv file: '/home/ubuntu/input/listing.csv'", type=str, required=True)
    parser.add_argument("--request-id", help="unique request id: 123456789", type=int, required=True)
    args = parser.parse_args()

    # access variables via args.meal_weight, args.csv_file_path etc.
    request_id = args.request_id

    test_result = {
           "pct_input_items": [
                   ["Viande, poisson, oeufs", 0.06121894571887815], 
                   ["Produits gras sucr\u00e9s sal\u00e9s", 0.24292811142750698], 
                   ["Produits laitiers", 0.17405525299186245], 
                   ["F\u00e9culents", 0.05400006425838616], 
                   ["Fruits et l\u00e9gumes", 0.46779762560336624]], 
                   
           "pct_allocated_items": [
                   ["Viande, poisson, oeufs", 0.03220187318160643], 
                   ["Produits gras sucr\u00e9s sal\u00e9s", 0.0],
                   ["Produits laitiers", 0.058458785160454746], 
                   ["F\u00e9culents", 0.050697308034631655], 
                   ["Fruits et l\u00e9gumes", 0.11917912945713675]], 
                   
           "pct_remaining_items": [
                   ["Viande, poisson, oeufs", 0.029017072537271725], 
                   ["Produits gras sucr\u00e9s sal\u00e9s", 0.24292811142750698], 
                   ["Produits laitiers", 0.11559646783140769], 
                   ["F\u00e9culents", 0.0033027562237545054], 
                   ["Fruits et l\u00e9gumes", 0.3486184961462295]], 
                   
           "nb_balanced_meals": 8, 
           "nb_allocated_items": 52, 
           "nb_remaining_items": 89, 
           "total_input_weight": 60555, 
           "total_allocated_weight": 15776, 
           "total_weight_remaining_items": 44778, 
           "pct_weight_allocated_items": 0.2605370958338296, 
           "pct_weight_remaining_items": 0.7394629041661704
           
            }
    
    print(json.dumps(test_result))


if __name__ == '__main__':
    main()
