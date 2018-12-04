import numpy as np
from scipy.spatial.distance import euclidean

# Licence: GNU GPLv3 
# Author: Aoife Fogarty
# 2018

# objectif :
# distribuer un nombre fixe d'items dans un nombre fixe de paniers

# chaque item a deux attributs : categorie, poids

# Cost function :
# la somme des distances distribution_de_categories vs distribution_optimale_de_categories
# +
# la somme des distances poids vs poids_optimal

# Methode d'optimisation : 
# Local search
#     solution initiale
#     neighbourhood function
#     acceptance strategy
#     stop criterion


def main():

    panier_optimal = np.asarray([0.3, 0.6, 0.1]) #{1: 0.3, 2: 0.6, 3: 0.1}
    n_categories = len(panier_optimal)
    poids_optimal = 300.0  # grammes
    n_items = 50

    model = LocalSearchModel(n_categories, n_items, poids_optimal, panier_optimal)

    model.initialise_items()
    model.initialise_paniers()

    # model.initialise_items_paniers_random()
    model.initialise_items_paniers_par_poids()

    cost, cost_poids, cost_cat = model.calc_cost()
    print('Initial cost (total, cost for basket weight, cost for category distribution)')
    print(cost, cost_poids, cost_cat)

    cost_tolerance = 3.5
    while model.cost > cost_tolerance:
        model.update_panier()


class LocalSearchModel:

    def __init__(self, n_categories, n_items, poids_optimal, panier_optimal):
        self.n_categories = n_categories
        self.n_items = n_items
        self.poids_optimal = poids_optimal
        self.panier_optimal = panier_optimal


    def initialise_items(self, cat_initial_dist='easiest'):
        '''
        Construire un systeme de self.n_items items avec poids et categorie aleatoire
        '''
    
        poids_max = 200.0
        # poids de chaque item
        # (distribution uniforme de poids)
        self.items_poids = np.random.random(size=self.n_items) * poids_max + 1.0

        # categorie de chaque item
        if cat_initial_dist == 'uniform':
            # (distribution uniforme de categories)
            self.items_cat = np.random.random_integers(low=1, 
                                                       high=self.n_categories, 
                                                       size=self.n_items)  
        elif cat_initial_dist == 'easiest':
            # (distribution des categories tiree de la distn optimale)
            self.items_cat = np.random.choice(list(range(1, self.n_categories+1)), 
                                              size=self.n_items, 
                                              replace=True, 
                                              p=self.panier_optimal)
       
 
    def initialise_paniers(self):
        '''
        Construire des paniers vides
        Nombre de paniers estime base sur le poids total des items
        '''

        # estimation du nombre de paniers 
        poids_total = np.sum(self.items_poids)
        self.n_paniers = int(np.ceil(poids_total / self.poids_optimal))

        print('Trying to construct {} baskets'.format(self.n_paniers))
       
        # matrice pour suivre le contenu de chaque panier
        # 1: item in panier, 0: item not in panier
        self.matrice_items_paniers = np.zeros((self.n_paniers, self.n_items), dtype=np.int32)
   

    def initialise_items_paniers_random(self): 
        '''
        Placer chaque item dans un panier choisi de facon aleatoire
        Chaque item peut etre dans un panier seulement
        '''

        indices_panier = np.random.random_integers(low=0, 
                                                   high=self.n_paniers-1, 
                                                   size=self.n_items)
        self.matrice_items_paniers[indices_panier, np.arange(self.n_items)] = 1


    def initialise_items_paniers_par_poids(self): 
        '''
        Placer chaque item dans un panier par ordre de poids
        Chaque item peut etre dans un panier seulement
        '''

        # trier les items par poids
        indices_sort = np.argsort(self.items_poids)

        for i, i_item in enumerate(indices_sort):
            i_panier = i % self.n_paniers
            self.matrice_items_paniers[i_panier, i_item] = 1
    
      
    def update_panier(self):
        '''
        Local search update
        '''

        costs = []
        matrices = []   
        for i in range(self.n_items):
            matrice_items_paniers = self.matrice_items_paniers 

            item = matrice_items_paniers[:, i]
            
            current_index = np.where(item==1)[0][0]
            index = np.random.random_integers(low=0, high=self.n_paniers-1)
            if index != current_index:
                # transferer l'item dans un autre panier
                matrice_items_paniers[current_index, i] = 0
                matrice_items_paniers[index, i] = 1
                cost, cost_poids, cost_cat = self.calc_cost(matrice_items_paniers)
                costs.append(cost)
                matrices.append(matrice_items_paniers)

        index_meilleur_cost = np.argmin(np.asarray(costs))
        if costs[index_meilleur_cost] < self.cost:
            self.cost = costs[index_meilleur_cost]
            self.matrice_items_paniers = matrices[index_meilleur_cost]
            print(self.cost)
            if self.cost < 3.0:
                print(self.calc_distribution_par_panier(self.matrice_items_paniers))
                print(self.calc_poids_par_panier(self.matrice_items_paniers))
   
 
    def calc_cost(self, matrice_items_paniers=None):
        '''
        Calculer le cost du systeme matrice_items_paniers si fourni
        Sinon calculer le code de self.matrice_items_paniers
        '''

        update_cost = False

        if matrice_items_paniers is None:
            matrice_items_paniers = self.matrice_items_paniers
            update_cost = True
      
        cost_poids = self.calc_cost_poids(matrice_items_paniers)
        cost_cat = self.calc_cost_cat(matrice_items_paniers)
        cost = 0.00 * cost_poids + cost_cat

        if update_cost:
            self.cost = cost

        return cost, cost_poids, cost_cat
       
 
    def calc_poids_par_panier(self, matrice_items_paniers):
        '''
        Calculer le poids de chaque panier dans matrice_items_paniers
        '''

        poids_par_panier = np.row_stack([self.items_poids] * self.n_paniers) * matrice_items_paniers
        poids_par_panier = np.sum(poids_par_panier, axis=1)
    
        assert np.isclose(poids_par_panier.sum(), self.items_poids.sum())

        return poids_par_panier

 
    def calc_cost_poids(self, matrice_items_paniers):
        '''
        cost = somme sur tous les paniers de (poids_panier - poids_optimal)^2
        '''
    
        poids_par_panier = self.calc_poids_par_panier(matrice_items_paniers)
 
        distance_poids = poids_par_panier - self.poids_optimal
    
        return np.sqrt(np.sum(distance_poids * distance_poids))
   

    def calc_distribution_par_panier(self, matrice_items_paniers): 
        '''
        Calculer la distribution de categories de chaque panier dans matrice_items_paniers
        '''
    
        # distributions des categories pour chaque panier
        cat_par_panier = np.row_stack([self.items_cat] * self.n_paniers) * matrice_items_paniers
        distribution_par_panier = np.apply_along_axis(distribution_panier, 
                                                      axis=1, 
                                                      arr=cat_par_panier, 
                                                      n_categories=self.n_categories)
    
        # verifier que somme des probabilites = 1
        assert np.all(np.logical_or(np.isclose(distribution_par_panier.sum(axis=1), 1.0), (distribution_par_panier.sum(axis=1) == 0.0)))

        return distribution_par_panier

    
    def calc_cost_cat(self, matrice_items_paniers):
        '''
        cost = somme sur tous les paniers de la distance entre la distribution du panier
        et la distribution d'un panier optimal
        '''

        # distributions des categories pour chaque panier
        distribution_par_panier = self.calc_distribution_par_panier(matrice_items_paniers)

        # distribution_par_panier = np.asarray([[0.32, 0.6, 0.08], [0.0, 0.0, 1.0]])
        distance_panier = np.apply_along_axis(euclidean, 
                                              axis=1, 
                                              arr=distribution_par_panier, 
                                              v=self.panier_optimal.reshape(-1, 1))
    
        return np.sum(distance_panier)
    
    
def distribution_panier(a, n_categories):
    '''
    Convertir entre une liste d'items et une distribution de categories
    0 = cet item n'est pas dans le panier
    Par exemple [0, 2, 3, 1, 0, 0, 0, 0] => [0.333, 0.333, 0.333]
    '''

    a = a[a > 0]

    n_items = float(len(a))
    unique, counts = np.unique(a, return_counts=True)

    d = np.zeros(n_categories, dtype=np.float64)
    d[unique - 1] = counts / n_items

    return d


if __name__ == '__main__':
    main()
