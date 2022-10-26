
import numpy as np
import networkx as nx
import itertools
import src.utils as ut

    
def check_path_between_sets(G,sources,targets):
    """
    It computes whether there is a path between a set of source nodes and a set of target nodes.

    Args:
        G: a networkx graph
        sources: a set of source nodes in G
        targets: a set of target nodes in G

    Returns:
        True if there is a path in G between source and target
    """
    augmentedG = G.copy()

    augmented_s = 'augmented_s_'+str(np.random.randint(10**6))
    augmented_t = 'augmented_t_'+str(np.random.randint(10**6))
    augmentedG.add_node(augmented_s)
    augmentedG.add_node(augmented_t)

    [augmentedG.add_edge(augmented_s,s) for s in sources]
    [augmentedG.add_edge(t,augmented_t) for t in targets]

    return nx.has_path(augmentedG,augmented_s,augmented_t)


def check_path_between_each_source_and_target(G,sources,targets):
    """
    It computes whether for each source there is a path going to (at least) a target AND for each target if there is a path arriving at that target

    Args:
        G: a networkx graph
        sources: a set of source nodes in G
        targets: a set of target nodes in G

    Returns:
        True if there is a path in G for each source and target
    """
    for s in sources:
        has_path = False
        for t in targets:
            if nx.has_path(G,s,t):
                has_path=True
                break
        if not(has_path): return False
    
    for t in targets:
        has_path = False
        for s in sources:
            if nx.has_path(G,s,t):
                has_path=True
                break
        if not(has_path): return False
        
    return True


def get_all_pairs(M):
    """
    It computes all the pairs of nodes

    Args:
        M: a pgmpy BN model
        
    Returns:
        J: list of all the pairs of nodes
    """
    J = list(itertools.permutations(M1.nodes(),2))
    return J
    
def get_pairs_with_directed_path(M):
    """
    For each pair (s,t) of nodes in M, the pair is added to the list if there is path from s to t.

    Args:
        M: a pgmpy BN model
        
    Returns:
        J: list of all the pairs of connected nodes
    """
    J = []
    sources = list(M.nodes())
    targets = list(M.nodes())
    for s in sources:
        for t in list(set(targets)-{s}):
            if nx.has_path(M,s,t):
                J.append((s,t))
    return J
        
def get_pairs_in_M1_with_directed_path_in_M1_or_M0(M0,M1,a):
    """
    For each pair (s,t) of nodes in M1, the pair is added to the list if (i) there is path from s to t in M1; OR (ii) there is path from a^-1(s) to a^-1(t)  

    Args:
        M0: a pgmpy BN model
        M1: a pgmpy BN model
        a: dictionary containing a surjective mapping from the variables of M0 to the variables of M1
        
    Returns:
        J: list of all the pairs of connected nodes either in M1 or in M0
    """
    J = []
    sources = list(M1.nodes())
    targets = list(M1.nodes())
    for s in sources:
        for t in list(set(targets)-{s}):
            if nx.has_path(M1,s,t):
                J.append((s,t))
            else:
                M0_sources = ut.inverse_fx(a,s)
                M0_targets = ut.inverse_fx(a,t)
                if check_path_between_sets(M0,M0_sources,M0_targets):
                    J.append((s,t))
    return J
    
def get_sets_in_M1_with_directed_path_in_M1_or_M0(M0,M1,a,verbose=False):
    """
    For each pair of disjoint sets (S,T) of nodes in M1, the pair is added to the list if (i) there is at least one path from S to T in M1; OR (ii) there is at least one path from a^-1(S) to a^-1(T)  

    Args:
        M0: a pgmpy BN model
        M1: a pgmpy BN model
        a: dictionary containing a surjective mapping from the variables of M0 to the variables of M1
        
    Returns:
        J: list of all the sets of connected nodes either in M1 or in M0
    """
    J = []
    sets = list(ut.powerset(M1.nodes()))
    sets.remove(())

    for i in sets:
        for j in sets:
            M1_sources = list(i)
            M1_targets = list(j)
            if not(any(x in M1_sources for x in M1_targets)):            
                if check_path_between_sets(M1,M1_sources,M1_targets):
                    if verbose: print('- Checking {0} -> {1}: True'.format(M1_sources,M1_targets))
                    J.append([M1_sources,M1_targets])
                else:
                    if verbose: print('- Checking {0} -> {1}: False'.format(M1_sources,M1_targets))
                    M0_sources = ut.inverse_fx(a,M1_sources)
                    M0_targets = ut.inverse_fx(a,M1_targets)
                    if check_path_between_sets(M0,M0_sources,M0_targets):
                        if verbose: print('---- Checking {0} -> {1}: True'.format(M0_sources,M0_targets))
                        J.append([M1_sources,M1_targets])
                    else:
                        if verbose: print('---- Checking {0} -> {1}: False'.format(M0_sources,M0_targets))
    if verbose: print('\n {0} legitimate pairs of sets out of {1} possbile pairs of sets'.format(len(J),len(sets)**2))  

    return J

def get_sets_in_M1_with_directed_path_in_M1_and_M0(M0,M1,a,verbose=False):
    """
    For each pair of disjoint sets (S,T) of nodes in M1, the pair is added to the list if (i) there is at least one path from S to T in M1; AND (ii) there is at least one path from a^-1(S) to a^-1(T)  

    Args:
        M0: a pgmpy BN model
        M1: a pgmpy BN model
        a: dictionary containing a surjective mapping from the variables of M0 to the variables of M1
        
    Returns:
        J: list of all the sets of connected nodes both in M1 and in M0
    """
    J = []
    sets = list(ut.powerset(M1.nodes()))
    sets.remove(())

    for i in sets:
        for j in sets:
            M1_sources = list(i)
            M1_targets = list(j)
            if not(any(x in M1_sources for x in M1_targets)):            
                if check_path_between_sets(M1,M1_sources,M1_targets):
                    if verbose: print('- Checking {0} -> {1}: True'.format(M1_sources,M1_targets))
                    M0_sources = ut.inverse_fx(a,M1_sources)
                    M0_targets = ut.inverse_fx(a,M1_targets)
                    if check_path_between_sets(M0,M0_sources,M0_targets):
                        if verbose: print('---- Checking {0} -> {1}: True'.format(M0_sources,M0_targets))
                        J.append([M1_sources,M1_targets])
                    else:
                        if verbose: print('---- Checking {0} -> {1}: False'.format(M0_sources,M0_targets))
                        if verbose: print('Found an inconsistent diagram. Returning None')
                        return None
                else:
                    if verbose: print('- Checking {0} -> {1}: False'.format(M1_sources,M1_targets))
                    
    if verbose: print('\n {0} legitimate pairs of sets out of {1} possbile pairs of sets'.format(len(J),len(sets)**2))  

    return J


def get_causal_sets_in_M1_with_directed_path_in_M1_and_M0(M0,M1,a,verbose=False):
    """
    For each pair of disjoint sets (S,T) of nodes in M1, the pair is added to the list if (i) every node in S reaches a node in T and every node in T is reached by a node in S; AND (ii) every node in a^-1(S) reaches a node in a^-1(T) and every node in a^-1(T) is reached by a node in a^-1(S) 

    Args:
        M0: a pgmpy BN model
        M1: a pgmpy BN model
        a: dictionary containing a surjective mapping from the variables of M0 to the variables of M1
        
    Returns:
        J: list of all the sets of connected nodes both in M1 and in M0
    """
    J = []
    sets = list(ut.powerset(M1.nodes()))
    sets.remove(())

    for i in sets:
        for j in sets:
            M1_sources = list(i)
            M1_targets = list(j)
            if not(any(x in M1_sources for x in M1_targets)):            
                if check_path_between_each_source_and_target(M1,M1_sources,M1_targets):
                    if verbose: print('- Checking {0} -> {1}: True'.format(M1_sources,M1_targets))
                    M0_sources = ut.inverse_fx(a,M1_sources)
                    M0_targets = ut.inverse_fx(a,M1_targets)
                    if check_path_between_each_source_and_target(M0,M0_sources,M0_targets):
                        if verbose: print('---- Checking {0} -> {1}: True'.format(M0_sources,M0_targets))
                        J.append([M1_sources,M1_targets])
                    else:
                        if verbose: print('---- Checking {0} -> {1}: False'.format(M0_sources,M0_targets))
                        if verbose: print('Found an inconsistent diagram. Returning None')
                        return None
                else:
                    if verbose: print('- Checking {0} -> {1}: False'.format(M1_sources,M1_targets))
                    
    if verbose: print('\n {0} legitimate pairs of sets out of {1} possbile pairs of sets'.format(len(J),len(sets)**2))  

    return J

def get_default_sets(M0,M1,a, J=None, J_algorithm=None, verbose=False):
    if J is None and J_algorithm is None:
        return get_causal_sets_in_M1_with_directed_path_in_M1_and_M0(M0,M1,a,verbose=verbose)
    elif J is None:
        return J_algorithm(M0,M1,a,verbose=verbose)
    else:
        return J

def get_causal_sets_in_M1_M0_with_directed_path_in_M1_and_M0(M0,M1,a,verbose=False):
    J = []
    O = []
    sets = list(ut.powerset(M1.nodes()))
    sets.remove(())

    for i in sets:
        for j in sets:
            M1_sources = list(i)
            M1_targets = list(j)
            if not(any(x in M1_sources for x in M1_targets)):            
                if check_path_between_each_source_and_target(M1,M1_sources,M1_targets):
                    if verbose: print('- Checking {0} -> {1}: True'.format(M1_sources,M1_targets))
                    M0_sources = ut.inverse_fx(a,M1_sources)
                    M0_targets = ut.inverse_fx(a,M1_targets)
                    if check_path_between_each_source_and_target(M0,M0_sources,M0_targets):
                        if verbose: print('---- Checking {0} -> {1}: True'.format(M0_sources,M0_targets))
                        J.append([M1_sources,M1_targets])
                        O.append([M0_sources,M0_targets])
                    else:
                        if verbose: print('---- Checking {0} -> {1}: False'.format(M0_sources,M0_targets))
                        if verbose: print('Found an inconsistent diagram. Returning None')
                        return None
                else:
                    if verbose: print('- Checking {0} -> {1}: False'.format(M1_sources,M1_targets))
                    
    if verbose: print('\n {0} legitimate pairs of sets out of {1} possbile pairs of sets'.format(len(J),len(sets)**2))  

    return J, O