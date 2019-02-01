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
        "allocated_items": [
            [
                "Produits gras sucr\u00e9s sal\u00e9s",
                0.2
            ],
            [
                "Mati\u00e8res grasses ajout\u00e9es",
                0.3
            ],
            [
                "Produits laitiers",
                0.1
            ],
            [
                "F\u00e9culents",
                0.2
            ],
            [
                "Fruits et l\u00e9gumes",
                0.2
            ]
        ],
        "remaining_items": [
            [
                "Produits gras sucr\u00e9s sal\u00e9s",
                0.35749502638180086
            ],
            [
                "Mati\u00e8res grasses ajout\u00e9es",
                0.03982354467606609
            ],
            [
                "Produits laitiers",
                0.3317706080788859
            ],
            [
                "F\u00e9culents",
                0.26174206383530835
            ],
            [
                "Fruits et l\u00e9gumes",
                0.00916875702793876
            ]
        ],
        "nb_balanced_meals": 5,
        "nb_remaining_items": 205.0,
        "pct_weight_allocated_items": 0.4,
        "pct_weight_remaining_items": 0.6
    }

    output_filename = 'output/{}_results.json'.format(request_id)
    with open(output_filename, 'w') as f:
        json.dump(test_result, f)

    return 0


if __name__ == '__main__':
    main()
