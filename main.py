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
                ["Viande, poisson, oeufs", 0.055366335703336746], 
                ["Produits gras sucr\u00e9s sal\u00e9s", 0.24074123343360457], 
                ["Mati\u00e8res grasses ajout\u00e9es", 0.013074132356507211], 
                ["Produits laitiers", 0.06832168022871912], 
                ["F\u00e9culents", 0.02283490602723674], 
                ["Fruits et l\u00e9gumes", 0.5996617122505956]],
                
        "pct_allocated_items": [
                ["Viande, poisson, oeufs", 0.003735466387573489], 
                ["Produits gras sucr\u00e9s sal\u00e9s", 0.0], 
                ["Mati\u00e8res grasses ajout\u00e9es", 0.0], 
                ["Produits laitiers", 0.010496660549081504], 
                ["F\u00e9culents", 0.00864013375445748], 
                ["Fruits et l\u00e9gumes", 0.013292034562449]], 
                
        "pct_remaining_items": [
                ["Viande, poisson, oeufs", 0.05163086931576326], 
                ["Produits gras sucr\u00e9s sal\u00e9s", 0.24074123343360457], 
                ["Mati\u00e8res grasses ajout\u00e9es", 0.013074132356507211], 
                ["Produits laitiers", 0.057825019679637615], 
                ["F\u00e9culents", 0.014194772272779259], 
                ["Fruits et l\u00e9gumes", 0.5863696776881466]], 
                
        "nb_balanced_meals": 7, 
        "nb_allocated_items": 41, 
        "nb_remaining_items": 380, 
        "total_input_weight": 267704, 
        "total_allocated_weight": 9681, 
        "total_weight_remaining_items": 258022, 
        "pct_weight_allocated_items": 0.036164295253561476, 
        "pct_weight_remaining_items": 0.9638357047464385
        }

    print(json.dumps(test_result))


if __name__ == '__main__':
    main()
