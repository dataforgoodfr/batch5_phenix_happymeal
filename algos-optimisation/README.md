# Optimisation discrete

### Mixed Integer Linear Programming

**Inputs:** given to the algorithm

* M: number of meals
* W: meal weight
* <a href="https://www.codecogs.com/eqnedit.php?latex=$\tau_{c}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\tau_{c}$" title="$\tau_{c}$" /></a>: weight proportion of category c, <a href="https://www.codecogs.com/eqnedit.php?latex=$c&space;=&space;1,&space;\dots,&space;C$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$c&space;=&space;1,&space;\dots,&space;C$" title="$c = 1, \dots, C$" /></a>
* <a href="https://www.codecogs.com/eqnedit.php?latex=$\delta$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\delta$" title="$\delta$" /></a>: authorized gap between optimal and output proportions of categories

**Inputs:** inducted by product listings or previous
* p: product, <a href="https://www.codecogs.com/eqnedit.php?latex=$p&space;=&space;1,&space;\dots,&space;P$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$p&space;=&space;1,&space;\dots,&space;P$" title="$p = 1, \dots, P$" /></a>
* <a href="https://www.codecogs.com/eqnedit.php?latex=$w_{p}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$w_{p}$" title="$w_{p}$" /></a>: weight of product *p*
* <a href="https://www.codecogs.com/eqnedit.php?latex=$Q_{p}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$Q_{p}$" title="$Q_{p}$" /></a>: number of items available for product *p*
* m: meal, <a href="https://www.codecogs.com/eqnedit.php?latex=$m&space;=&space;1,&space;\dots,&space;M$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$m&space;=&space;1,&space;\dots,&space;M$" title="$m = 1, \dots, M$" /></a>

**Integers to find:**
* <a href="https://www.codecogs.com/eqnedit.php?latex=$q_{m,p}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$q_{m,p}$" title="$q_{m,p}$" /></a>: number of items *p* in meal *m*

**Function to optimize:** must be linear, to be defined

**Constraints:**
* <a href="https://www.codecogs.com/eqnedit.php?latex=$\forall&space;p,&space;\sum_{m=1,\dots,M}&space;q_{m,p}\leq&space;Q_{p}&space;$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\forall&space;p,&space;\sum_{m=1,\dots,M}&space;q_{m,p}\leq&space;Q_{p}&space;$" title="$\forall p, \sum_{m=1,\dots,M} q_{m,p}\leq Q_{p} $" /></a>: can't distribute more than available
* <a href="https://www.codecogs.com/eqnedit.php?latex=$\forall&space;m,&space;\forall&space;c,&space;\sum_{p&space;\in&space;c}&space;q_{m,p}&space;\times&space;w_{p}&space;\geq&space;(\tau_{c}&space;-&space;\delta$)&space;\times&space;W" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\forall&space;m,&space;\forall&space;c,&space;\sum_{p&space;\in&space;c}&space;q_{m,p}&space;\times&space;w_{p}&space;\geq&space;(\tau_{c}&space;-&space;\delta$)&space;\times&space;W" title="$\forall m, \forall c, \sum_{p \in c} q_{m,p} \times w_{p} \geq (\tau_{c} - \delta$) \times W" /></a>: *c* proportion in meal *m* must be greater than
* <a href="https://www.codecogs.com/eqnedit.php?latex=$\forall&space;m,&space;\forall&space;c,&space;\sum_{p&space;\in&space;c}&space;q_{m,p}&space;\times&space;w_{p}&space;\leq&space;(\tau_{c}&space;&plus;&space;\delta$)&space;\times&space;W" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\forall&space;m,&space;\forall&space;c,&space;\sum_{p&space;\in&space;c}&space;q_{m,p}&space;\times&space;w_{p}&space;\leq&space;(\tau_{c}&space;&plus;&space;\delta$)&space;\times&space;W" title="$\forall m, \forall c, \sum_{p \in c} q_{m,p} \times w_{p} \leq (\tau_{c} + \delta$) \times W" /></a>: *c* proportion in meal *m* must be lower than

### Methodes exactes - beaucoup trop lentes

### Methodes heuristiques - probablement trop lentes aussi ???

Par exemple, Local Search : voir `panier_localsearch.py`

Points d'amelioration :

* initialisation 

* cost function

* choice of neighbourhood moves

* stopping criterion

### Solution possible : initialisation par approche tres approximative + amelioration par methode heuristique ?
