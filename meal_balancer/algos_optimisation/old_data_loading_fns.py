def old_example():

    use_real_test_data = True

    # TODO store data in df instead of dictionary
    if use_real_test_data:
        # load real test data
        filename = 'sample_offres.csv'
        pdt_info_dict = load_real_test_data(filename)
        n_products = len(pdt_info_dict)
    else:
        # create random fake test data
        n_products = 100
        pdt_info_dict = create_fake_test_data(n_products, n_meals, n_categories)


def create_fake_test_data(n_products, n_meals, n_categories):
    '''
    create n_products products with random weights and categories and
    number of items
    '''

    # weight of each item
    max_item_weight = 200
    weights = np.random.random(size=n_products) * max_item_weight + 1.0

    # category of each item
    categories = np.random.random_integers(low=1, high=n_categories, size=n_products)

    # number of items of each product
    # (based on observed data it is something like an exponential distribution)
    n_items = np.random.exponential(scale=2.0, size=n_products)

    pdt_info_dict = {'qty_gram': weights, 'code_cat': categories, 'n_items': n_items}

    return pdt_info_dict


def load_real_test_data(filename):
    '''
    Parse csv file extracted from Offres table in Phenix database
    Warning: not all weights are correctly parsed yet
    Temporary function for testing and benchmarking MILP algos
    '''

    # category to use for tests
    category_to_use = 'pnns_groups_1'

    df = pd.read_csv(filename, sep=',', encoding='iso-8859-1')
    df = df[~df[category_to_use].isnull()]

    # get weight in grams of each product
    df['weight_per_entry'] = df.quantity_off.apply(lambda s: convert_quantity(s))
    df = pd.concat([df.drop(['weight_per_entry'], axis=1), df['weight_per_entry'].apply(pd.Series)], axis=1)
    df = df[~df['val'].isnull()]
    df = df.rename(columns={'val': 'qty_gram'})

    # encode category of each product as an integer
    le = LabelEncoder()
    df['code_cat'] = le.fit_transform(df[category_to_use].values)
    # TODO return le.classes_ (label of each integer class) for later use

    # regroup duplicate products
    df['QuantiteValeur'] = df.groupby(['EAN', 'code_cat', 'qty_gram'])['QuantiteValeur'] \
                             .transform('sum') \
                             .astype(int)
    # Note: in some very rare cases QuantiteValeur is not an integer,
    # but we neglect this
    df = df[['EAN', 'Nom', 'code_cat', category_to_use, 'qty_gram', 'QuantiteValeur']] \
                         .drop_duplicates(['EAN', 'code_cat', 'qty_gram']) \
                         .reset_index(drop = True)

    df = df[df['QuantiteValeur'] > 0]

    return df.to_dict(orient='list')

