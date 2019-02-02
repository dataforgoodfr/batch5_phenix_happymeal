# Optimisation discrete

### Mixed Integer Linear Programming - V1

**Inputs:** given to the algorithm

* M: number of meals
* W: meal weight
* <a href="https://www.codecogs.com/eqnedit.php?latex=c_{j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_{j}" title="c_{j}" /></a>: level 1 food categories, <a href="https://www.codecogs.com/eqnedit.php?latex=c_j&space;\in&space;\{c_{10},&space;c_{20},&space;c_{30},&space;c_{40},&space;c_{50},&space;c_{60}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_j&space;\in&space;\{c_{10},&space;c_{20},&space;c_{30},&space;c_{40},&space;c_{50},&space;c_{60}\}" title="c_j \in \{c_{10}, c_{20}, c_{30}, c_{40}, c_{50}, c_{60}\}" /></a>
* <a href="https://www.codecogs.com/eqnedit.php?latex=c_{jk}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_{jk}" title="c_{jk}" /></a>: level 2 food categories, <a href="https://www.codecogs.com/eqnedit.php?latex=c_{jk}&space;\in&space;\{c_{11},&space;c_{12},&space;c_{41},&space;c_{42},&space;c_{51},&space;c_{52},&space;c_{61},&space;c_{62}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_{jk}&space;\in&space;\{c_{11},&space;c_{12},&space;c_{41},&space;c_{42},&space;c_{51},&space;c_{52},&space;c_{61},&space;c_{62}\}" title="c_{jk} \in \{c_{11}, c_{12}, c_{41}, c_{42}, c_{51}, c_{52}, c_{61}, c_{62}\}" /></a>
* <a href="https://www.codecogs.com/eqnedit.php?latex=\tau_{c_j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau_{c_j}" title="\tau_{c_j}" /></a>: weight proportion of category <a href="https://www.codecogs.com/eqnedit.php?latex=c_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_j" title="c_j" /></a> in a meal
* <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\tau_{c_{10}}=&space;0.12,&space;\tau_{c_{20}}=0.025,&space;\tau_{c_{30}}=0.025" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\tau_{c_{10}}=&space;0.12,&space;\tau_{c_{20}}=0.025,&space;\tau_{c_{30}}=0.025" title="\tau_{c_{10}}= 0.12, \tau_{c_{20}}=0.025, \tau_{c_{30}}=0.025" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\tau_{c_{40}}=0.25,&space;\tau_{c_{50}}=0.25,&space;\tau_{c_{60}}=0.33" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\tau_{c_{40}}=0.25,&space;\tau_{c_{50}}=0.25,&space;\tau_{c_{60}}=0.33" title="\tau_{c_{40}}=0.25, \tau_{c_{50}}=0.25, \tau_{c_{60}}=0.33" /></a>
* <a href="https://www.codecogs.com/eqnedit.php?latex=\tau_{c_{jk}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau_{c_{jk}}" title="\tau_{c_{jk}}" /></a>: weight proportion of category <a href="https://www.codecogs.com/eqnedit.php?latex=c_{jk}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_{jk}" title="c_{jk}" /></a> in category <a href="https://www.codecogs.com/eqnedit.php?latex=c_{j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_{j}" title="c_{j}" /></a>
* <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\tau_{c_{12}}=0.25,&space;\tau_{c_{42}}=0.1,&space;\tau_{c_{52}}=0.5,&space;\tau_{c_{61}}=0.5" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\tau_{c_{12}}=0.25,&space;\tau_{c_{42}}=0.1,&space;\tau_{c_{52}}=0.5,&space;\tau_{c_{61}}=0.5" title="\tau_{c_{12}}=0.25, \tau_{c_{42}}=0.1, \tau_{c_{52}}=0.5, \tau_{c_{61}}=0.5" /></a>
* <a href="https://www.codecogs.com/eqnedit.php?latex=$\delta$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\delta$" title="$\delta$" /></a>: authorized gap between optimal and output proportions of categories (a percentage of <a href="https://www.codecogs.com/eqnedit.php?latex=\tau_c" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau_c" title="\tau_c" /></a>)

**Inputs:** induced by product listings or previous
* p: product, <a href="https://www.codecogs.com/eqnedit.php?latex=$p&space;=&space;1,&space;\dots,&space;P$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$p&space;=&space;1,&space;\dots,&space;P$" title="$p = 1, \dots, P$" /></a>
* <a href="https://www.codecogs.com/eqnedit.php?latex=$w_{p}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$w_{p}$" title="$w_{p}$" /></a>: weight of product *p*
* <a href="https://www.codecogs.com/eqnedit.php?latex=$Q_{p}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$Q_{p}$" title="$Q_{p}$" /></a>: number of items available for product *p*
* m: meal, <a href="https://www.codecogs.com/eqnedit.php?latex=$m&space;=&space;1,&space;\dots,&space;M$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$m&space;=&space;1,&space;\dots,&space;M$" title="$m = 1, \dots, M$" /></a>

**Integers to find:**
* <a href="https://www.codecogs.com/eqnedit.php?latex=$q_{m,p}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$q_{m,p}$" title="$q_{m,p}$" /></a>: number of items *p* in meal *m*

**Function to optimize:** must be linear, to be defined

**Constraints:**
* <a href="https://www.codecogs.com/eqnedit.php?latex=$\forall&space;p,&space;\sum_{m=1,\dots,M}&space;q_{m,p}\leq&space;Q_{p}&space;$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\forall&space;p,&space;\sum_{m=1,\dots,M}&space;q_{m,p}\leq&space;Q_{p}&space;$" title="$\forall p, \sum_{m=1,\dots,M} q_{m,p}\leq Q_{p} $" /></a>: can't distribute more than available
* <a href="https://www.codecogs.com/eqnedit.php?latex=\forall&space;m,&space;\forall&space;c_j&space;\in&space;\{c_{10},&space;c_{40},&space;c_{50},&space;c_{60}\}&space;\sum_{p&space;\in&space;c_j}&space;q_{m,p}&space;\times&space;w_p&space;\geq&space;\tau_{c_j}\times(1-\delta)\times&space;W" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\forall&space;m,&space;\forall&space;c_j&space;\in&space;\{c_{10},&space;c_{40},&space;c_{50},&space;c_{60}\}&space;\sum_{p&space;\in&space;c_j}&space;q_{m,p}&space;\times&space;w_p&space;\geq&space;\tau_{c_j}\times(1-\delta)\times&space;W" title="\forall m, \forall c_j \in \{c_{10}, c_{40}, c_{50}, c_{60}\} \sum_{p \in c_j} q_{m,p} \times w_p \geq \tau_{c_j}\times(1-\delta)\times W" /></a>
* <a href="https://www.codecogs.com/eqnedit.php?latex=\forall&space;m,&space;\forall&space;c_j&space;\in&space;\{c_{20},&space;c_{30}\}&space;\sum_{p&space;\in&space;c_j}&space;q_{m,p}&space;\times&space;w_p&space;\leq&space;\tau_{c_j}&space;\times&space;(1&plus;\delta)\times&space;W" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\forall&space;m,&space;\forall&space;c_j&space;\in&space;\{c_{20},&space;c_{30}\}&space;\sum_{p&space;\in&space;c_j}&space;q_{m,p}&space;\times&space;w_p&space;\leq&space;\tau_{c_j}&space;\times&space;(1&plus;\delta)\times&space;W" title="\forall m, \forall c_j \in \{c_{20}, c_{30}\} \sum_{p \in c_j} q_{m,p} \times w_p \leq \tau_{c_j} \times (1+\delta)\times W" /></a>
* <a href="https://www.codecogs.com/eqnedit.php?latex=\forall&space;m,&space;\forall&space;c_{jk}&space;\in&space;\{c_{12},&space;c_{52},&space;c_{61}\}&space;\sum_{p&space;\in&space;c_{jk}}&space;q_{m,p}&space;\times&space;w_p&space;\geq&space;\tau_{c_{jk}}\times&space;\tau_{c_j}\times(1-\delta)&space;\times&space;W" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\forall&space;m,&space;\forall&space;c_{jk}&space;\in&space;\{c_{12},&space;c_{52},&space;c_{61}\}&space;\sum_{p&space;\in&space;c_{jk}}&space;q_{m,p}&space;\times&space;w_p&space;\geq&space;\tau_{c_{jk}}\times&space;\tau_{c_j}\times(1-\delta)&space;\times&space;W" title="\forall m, \forall c_{jk} \in \{c_{12}, c_{52}, c_{61}\} \sum_{p \in c_{jk}} q_{m,p} \times w_p \geq \tau_{c_{jk}}\times \tau_{c_j}\times(1-\delta) \times W" /></a>
* <a href="https://www.codecogs.com/eqnedit.php?latex=\forall&space;m,&space;\forall&space;c_{jk}&space;\in&space;\{c_{42}\}&space;\sum_{p&space;\in&space;c_{jk}}&space;q_{m,p}&space;\times&space;w_p&space;\leq&space;\tau_{c_{jk}}\times&space;\tau_{c_j}\times(1&plus;\delta)&space;\times&space;W" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\forall&space;m,&space;\forall&space;c_{jk}&space;\in&space;\{c_{42}\}&space;\sum_{p&space;\in&space;c_{jk}}&space;q_{m,p}&space;\times&space;w_p&space;\leq&space;\tau_{c_{jk}}\times&space;\tau_{c_j}\times(1&plus;\delta)&space;\times&space;W" title="\forall m, \forall c_{jk} \in \{c_{42}\} \sum_{p \in c_{jk}} q_{m,p} \times w_p \leq \tau_{c_{jk}}\times \tau_{c_j}\times(1+\delta) \times W" /></a>



### Mixed Integer Linear Programming - V0

**Inputs:** given to the algorithm

* M: number of meals
* W: meal weight
* <a href="https://www.codecogs.com/eqnedit.php?latex=$\tau_{c}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\tau_{c}$" title="$\tau_{c}$" /></a>: weight proportion of category c, <a href="https://www.codecogs.com/eqnedit.php?latex=$c&space;=&space;1,&space;\dots,&space;C$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$c&space;=&space;1,&space;\dots,&space;C$" title="$c = 1, \dots, C$" /></a>
* <a href="https://www.codecogs.com/eqnedit.php?latex=$\delta$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\delta$" title="$\delta$" /></a>: authorized gap between optimal and output proportions of categories

**Inputs:** induced by product listings or previous
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

### Install non-commercial solver (alternative to gurobi)

#### Install of OR-Tools (Coin or Branch and Cut)

See here installation instructions:
https://developers.google.com/optimization/install/python/


#### Install of GLPK

If conda is not already installed, install anaconda or miniconda
For example under linux
```
curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source .bashrc
```
Other OS: https://conda.io/docs/user-guide/install/index.html
Warning: Anaconda will take up 3 GB of disk space, so better install Miniconda

Now install glpk (LP and MIP solver) + cvxopt (python wrapper) + cvxpy (python wrapper)
```
# install glpk
conda install -c conda-forge glpk
# check if glpk is installed
glpsol -v
# install cvxopt
conda install -c conda-forge cvxopt
# check if cvxopt is able to find glpk by running 'from cvxopt.glpk import ilp' in python interpretor
# install cvxpy
conda install -c conda-forge lapack
conda install -c cvxgrp cvxpy
# check if cvxpy is able to find glpk and cvxopt by running milp_cvxpy_glpk.py
```
