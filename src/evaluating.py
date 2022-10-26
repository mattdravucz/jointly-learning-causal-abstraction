
import numpy as np
from scipy.spatial import distance

import src.evaluationsets as es

class SCMMappingEvaluator():
    def __init__(self,A):
        self.A = A

class AbstractionEvaluator(SCMMappingEvaluator):
    def __init__(self,A):
        super().__init__(A)
            
class AbstractionErrorEvaluator(AbstractionEvaluator):
    def __init__(self,A):
        super().__init__(A)
        
    def evaluate_abstraction_errors(self, metric=None, J=None,J_algorithm=None, verbose=False, debug=False):
        J = es.get_default_sets(self.A.M0,self.A.M1,self.A.a, J,J_algorithm)
            
        if metric is None:
            metric = distance.jensenshannon
        
        abstraction_errors = []

        for pair in J:
            # Get nodes in the abstracted model
            M1_sources = pair[0]
            M1_targets = pair[1]
            if verbose: print('\nM1: {0} -> {1}'.format(M1_sources,M1_targets))

            # Get nodes in the base model
            M0_sources = self.A.invert_a(M1_sources)
            M0_targets = self.A.invert_a(M1_targets)
            if verbose: print('M0: {0} -> {1}'.format(M0_sources,M0_targets))

            # Compute the high-level mechanisms
            M1_cond_TS_val = self.A.compute_mechanisms(self.A.M1,M1_sources,M1_targets)
            if verbose: print('M1 mechanism shape: {}'.format(M1_cond_TS_val.shape))

            # Compute the low-level mechanisms
            M0_cond_TS_val = self.A.compute_mechanisms(self.A.M0,M0_sources,M0_targets)
            if verbose: print('M0 mechanism shape: {}'.format(M0_cond_TS_val.shape))

            # Compute the alpha for sources
            alpha_S = self.A.compute_abstractions(M1_sources)
            if verbose: print('Alpha_s shape: {}'.format(alpha_S.shape))

            # Compute the alpha for targets
            alpha_T = self.A.compute_abstractions(M1_targets)
            if verbose: print('Alpha_t shape: {}'.format(alpha_T.shape))

            # Evaluate the paths on the diagram
            lowerpath = np.dot(M1_cond_TS_val,alpha_S)
            upperpath = np.dot(alpha_T,M0_cond_TS_val)

            # Compute abstraction error for every possible intervention
            distances = []
            if debug: print('{0} \n\n {1}'.format(lowerpath,upperpath))
            for c in range(lowerpath.shape[1]):
                distances.append( metric(lowerpath[:,c],upperpath[:,c]) )
            if verbose: print('All JS distances: {0}'.format(distances))

            # Select the greatest distance over all interventions
            if verbose: print('\nAbstraction error: {0}'.format(np.max(distances)))
            abstraction_errors.append(np.max(distances))

        # Select the greatest distance over all pairs considered
        if verbose: print('\n\nOVERALL ABSTRACTION ERROR: {0}'.format(np.max(abstraction_errors)))
            
        return abstraction_errors
    
    def evaluate_overall_abstraction_error(self, metric=None, J=None,J_algorithm=None, verbose=False):
        errors = np.array(self.evaluate_abstraction_errors(metric=metric,J=J,J_algorithm=J_algorithm,verbose=verbose))
        errors[np.argwhere(np.isnan(errors))] = 0
        return np.max(errors)
    
    def evaluate_cumulative_abstraction_errors(self, metric=None, J=None,J_algorithm=None, verbose=False):
        errors = np.array(self.evaluate_abstraction_errors(metric=metric,J=J,J_algorithm=J_algorithm,verbose=verbose))
        errors[np.argwhere(np.isnan(errors))] = 0
        return np.sum(errors)
    
    def is_exact(self, metric=None,J=None,J_algorithm=None,verbose=False, rtol=1e-05, atol=1e-08):
        error = self.evaluate_overall_abstraction_error(metric=metric,J=J,J_algorithm=J_algorithm,verbose=verbose)
        return np.isclose(0,error,rtol=rtol,atol=atol)
        
class AbstractionInfoLossEvaluator(AbstractionEvaluator):
    def __init__(self,A):
        super().__init__(A)
        
    def evaluate_info_loss(self, metric=None, invalpha_algorithm=None, verbose=False):
        if metric is None:
            metric = distance.jensenshannon
        
        joint_M0,joint_M1 = self.A.compute_joints(verbose=verbose)
        inverse_joint_M1 = self.A.compute_inverse_joint_M1(invalpha_algorithm=invalpha_algorithm, verbose=verbose)
        info_loss = metric( joint_M0, inverse_joint_M1 )
        
        return info_loss
    