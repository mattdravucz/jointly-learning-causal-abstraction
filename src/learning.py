
import numpy as np
import itertools

import src.utils as ut
from src.SCMMappings_1_1 import Abstraction
from src.evaluating import AbstractionErrorEvaluator


def enumerate_all_surjective_maps(dom,codom):
    diff = dom-codom
    
    surjective_values = np.arange(0,codom,dtype=int)
    nonsurjective_values = np.array(list(itertools.product(surjective_values,repeat=diff)),dtype=int)
    
    surjective_maps = []
    for nsv in nonsurjective_values:
        map_values = np.concatenate((surjective_values,nsv))
        surjective_maps.extend(list(itertools.permutations(np.concatenate((surjective_values,nsv)))))
        
    return list(set(surjective_maps))

def get_all_surjective_matrices(dom,codom):
    vs = enumerate_all_surjective_maps(dom,codom)
    Ms = []
    for v in vs:
        Ms.append(ut.map_vect2matrix(v))
    return Ms

def learn_alpha_by_enumeration(A,J=None):
    alphanames = list(A.M1.nodes)
    candidate_alphas = {}
    for X_ in alphanames:
        dom,codom = A.get_cardinalities_alpha(X_)
        candidates = get_all_surjective_matrices(dom,codom)
        candidate_alphas[X_] = candidates
    
    c_alphas=[]
    c_errors=[]
    for c_alpha in itertools.product(*candidate_alphas.values()):
        alphas = {}
        for i in range(len(alphanames)):
            alphas[alphanames[i]] = c_alpha[i]
        c_A = Abstraction(A.M0,A.M1,A.R,A.a,alphas)
        c_A_ev = AbstractionErrorEvaluator(c_A)

        c_errors.append(c_A_ev.evaluate_overall_abstraction_error(J=J))
        c_alphas.append(alphas)
        
    return np.min(c_errors),c_alphas[np.argmin(c_errors)]

def list_all_alphas_and_errors(A,J=None):
    alphanames = list(A.M1.nodes)
    candidate_alphas = {}
    for X_ in alphanames:
        dom,codom = A.get_cardinalities_alpha(X_)
        candidates = get_all_surjective_matrices(dom,codom)
        candidate_alphas[X_] = candidates
    
    c_alphas=[]
    c_errors=[]
    for c_alpha in itertools.product(*candidate_alphas.values()):
        alphas = {}
        for i in range(len(alphanames)):
            alphas[alphanames[i]] = c_alpha[i]
        c_A = Abstraction(A.M0,A.M1,A.R,A.a,alphas)
        c_A_ev = AbstractionErrorEvaluator(c_A)

        c_errors.append(c_A_ev.evaluate_overall_abstraction_error(J=J))
        c_alphas.append(alphas)
        
    return c_errors,c_alphas