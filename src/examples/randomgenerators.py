import numpy as np

import src.utils as ut

def set_seed(seed):
     np.random.seed(seed)

def generate_random_alpha(dom,codom):
    surjective_values = np.arange(0,codom,dtype=int)
    non_surjective_values = np.random.randint(0,codom,(dom-codom),dtype=int)
    random_values = np.concatenate((surjective_values,non_surjective_values))
    np.random.shuffle(random_values)
    return ut.map_vect2matrix(random_values)

def generate_random_alphas(M0,M1,a):
    M1names = list(M1.nodes)
    
    alphas = {}
    for m in M1names:
        dom,codom = ut.get_cardinalities_Falpha(a,m,M0.get_cardinality(),M1.get_cardinality())
        alphas[m] = generate_random_alpha(dom,codom)
        
    return alphas