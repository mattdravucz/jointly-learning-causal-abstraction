
import numpy as np

from pgmpy.models import BayesianNetwork as BN
from pgmpy.factors.discrete import TabularCPD as cpd


def M_pgmpy_chain_STC(MphiS,MphiT,MphiC,S='Smoking',T='Tar',C='Cancer'):
    M = BN([(S,T),(T,C)])

    cpdS = cpd(variable=S,
              variable_card=MphiS.shape[0],
              values=MphiS,
              evidence=None,
              evidence_card=None)
    cpdT = cpd(variable=T,
              variable_card=MphiT.shape[0],
              values=MphiT,
              evidence=[S],
              evidence_card=[MphiT.shape[1]])
    cpdC = cpd(variable=C,
              variable_card=MphiC.shape[0],
              values=MphiC,
              evidence=[T],
              evidence_card=[MphiC.shape[1]])

    M.add_cpds(cpdS,cpdT,cpdC)
    if M.check_model(): return M
    
def M_pgmpy_chain_SC(MphiS,MphiC,S='Smoking_',C='Cancer_'):
    M = BN([(S,C)])

    cpdS = cpd(variable=S,
              variable_card=MphiS.shape[0],
              values=MphiS,
              evidence=None,
              evidence_card=None)
    cpdC = cpd(variable=C,
              variable_card=MphiC.shape[0],
              values=MphiC,
              evidence=[S],
              evidence_card=[MphiC.shape[1]])

    M.add_cpds(cpdS,cpdC)
    if M.check_model(): return M
    
def M_pgmpy_indep_SC(S='Smoking__',C='Cancer__'):
    M = BN()
    M.add_node(S)
    M.add_node(C)

    cpdS = cpd(variable=S,
              variable_card=2,
              values=[[.8],[.2]],
              evidence=None,
              evidence_card=None)
    cpdC = cpd(variable=C,
              variable_card=2,
              values=[[.852],[.148]],
              evidence=None,
              evidence_card=None)

    M.add_cpds(cpdS,cpdC)
    if M.check_model(): return M
    
def M_pgmpy_singleton():
    M = BN()
    M.add_node('*')

    cpdX = cpd(variable='*',
              variable_card=1,
              values=[[1.]],
              evidence=None,
              evidence_card=None)

    M.add_cpds(cpdX)
    if M.check_model(): return M

def M_pgmpy_chain_ESTC(MphiE,MphiS,MphiT,MphiC,E='Environment',S='Smoking',T='Tar',C='Cancer'):
    M = BN([(E,S),(S,T),(T,C)])
    
    cpdE = cpd(variable=E,
              variable_card=MphiE.shape[0],
              values=MphiE,
              evidence=None,
              evidence_card=None)
    cpdS = cpd(variable=S,
              variable_card=MphiS.shape[0],
              values=MphiS,
              evidence=[E],
              evidence_card=[MphiS.shape[1]])
    cpdT = cpd(variable=T,
              variable_card=MphiT.shape[0],
              values=MphiT,
              evidence=[S],
              evidence_card=[MphiT.shape[1]])
    cpdC = cpd(variable=C,
              variable_card=MphiC.shape[0],
              values=MphiC,
              evidence=[T],
              evidence_card=[MphiC.shape[1]])

    M.add_cpds(cpdE,cpdS,cpdT,cpdC)
    if M.check_model(): return M
    
def M_pgmpy_vstruct_SGCFH(MphiS,MphiG,MphiC,MphiH,MphiF,
                          S='Smoking',G='Genetics',C='Cancer',H='Coughing',F='Fatigue'):
    M = BN([(S,C), (G,C), (C,H), (C,F), (H,F)])

    cpdS = cpd(variable=S,
              variable_card=MphiS.shape[0],
              values=MphiS,
              evidence=None,
              evidence_card=None)
    cpdG = cpd(variable=G,
              variable_card=MphiG.shape[0],
              values=MphiG,
              evidence=None,
              evidence_card=None)
    cpdC = cpd(variable=C,
              variable_card=MphiC.shape[0],
              values=MphiC,
              evidence=[S,G],
              evidence_card=[MphiS.shape[0],MphiG.shape[0]])
    cpdH = cpd(variable=H,
              variable_card=MphiH.shape[0],
              values=MphiH,
              evidence=[C],
              evidence_card=[MphiH.shape[1]])
    cpdF = cpd(variable=F,
              variable_card=MphiF.shape[0],
              values=MphiF,
              evidence=[C,H],
              evidence_card=[MphiC.shape[0],MphiF.shape[0]])

    M.add_cpds(cpdS,cpdG,cpdC,cpdF,cpdH)
    if M.check_model(): return M
    

M_chain_STC_a = M_pgmpy_chain_STC(MphiS = np.array([[.8],[.2]]), 
                                  MphiT = np.array([[1,.2],[0,.8]]), 
                                  MphiC = np.array([[.9,.6],[.1,.4]]))
M_chain_STC_b = M_pgmpy_chain_STC(MphiS = np.array([[.8],[.2]]), 
                                  MphiT = np.array([[.5,.1],[.5,.1],[0.,.8]]), 
                                  MphiC = np.array([[.9,.9,.6],[.1,.1,.4]]))
M_chain_STC_c = M_pgmpy_chain_STC(MphiS = np.array([[.8],[.2]]), 
                                  MphiT = np.array([[1,.2],[0,.8]]), 
                                  MphiC = np.array([[.5,.1],[.4,.5],[.1,.4]]))
    
M_chain_ST_a = M_pgmpy_chain_SC(MphiS = np.array([[.8],[.2]]), 
                                  MphiC = np.array([[.9,.66],[.1,.34]]))    
M_chain_ST_b = M_pgmpy_chain_SC(MphiS = np.array([[.8],[.2]]), 
                                  MphiC = np.array([[.9,.6],[.1,.4]]))
M_chain_ST_c = M_pgmpy_chain_SC(MphiS = np.array([[.7],[.3]]), 
                                  MphiC = np.array([[.9,.66],[.1,.34]])) 
M_chain_ST_d = M_pgmpy_chain_SC(MphiS = np.array([[.2],[.8]]), 
                                  MphiC = np.array([[.9,.66],[.1,.34]]))
M_chain_ST_e = M_pgmpy_chain_SC(MphiS = np.array([[.2],[.8]]), 
                                  MphiC = np.array([[.9,.26],[.1,.74]]))

M0_multidiagramtest = M_pgmpy_chain_STC(MphiS = np.array([[.25],[.25],[.25],[.25]]), 
                                  MphiT = np.array([[.6,.55,.1,.1],[.3,.25,.4,.4],[.1,.2,.5,.5]]), 
                                  MphiC = np.array([[.7,.7,.4],[.3,.3,.6]]),
                                  S='Smoking',T='Tar',C='Cancer')
M1_multidiagramtest = M_pgmpy_chain_STC(MphiS = np.array([[.25],[.5],[.25]]), 
                                  MphiT = np.array([[.9,.8,.5],[.1,.2,.5]]), 
                                  MphiC = np.array([[.7,.4],[.3,.6]]),
                                  S='Smoking_',T='Tar_',C='Cancer_')

M_chain_ESTC_a = M_pgmpy_chain_ESTC(MphiE = np.array([[.45],[.55]]),
                                    MphiS = np.array([[.9,.7],[.1,.3]]),
                                    MphiT = np.array([[.95,.2],[.05,.8]]),
                                    MphiC = np.array([[.9,.6],[.1,.4]]),
                                    E='Environment',S='Smoking',T='Tar',C='Cancer')
M_chain_STC_d = M_pgmpy_chain_STC(MphiS = np.array([[.8],[.2]]), 
                                  MphiT = np.array([[1,.2],[0,.8]]), 
                                  MphiC = np.array([[.9,.6],[.1,.4]]),
                                  S='Smoking_',T='Tar_',C='Cancer_')

M_vstruct_SGCFH_a = M_pgmpy_vstruct_SGCFH(MphiS = np.array([[.8],[.2]]),
                                    MphiG = np.array([[.7],[.3]]),
                                    MphiC = np.array([[.15,.85,.65,.75],[.85,.15,.35,.25]]),
                                    MphiH = np.array([[1,.2],[0,.8]]),
                                    MphiF = np.array([[.42,.75,.65,.33],[.58,.25,.35,.67]]),
                                    S='Smoking',G='Genetics',C='Cancer',H='Coughing',F='Fatigue')
M_chain_SCF_a = M_pgmpy_chain_STC(MphiS = np.array([[.8],[.2]]), 
                                  MphiT = np.array([[.9,.66],[.1,.34]]), 
                                  MphiC = np.array([[.8,.5],[.2,.5]]),
                                  S='Smoking_',T='Cancer_',C='Fatigue_')

    
def standardA_M0chainSTC_M1chainSC():
    M0 = M_chain_STC_a
    M1 = M_chain_ST_a   
    R = ['Smoking','Cancer']
    a = {'Smoking': 'Smoking_',
        'Cancer': 'Cancer_'}
    alphas = {'Smoking_': np.eye(2),
             'Cancer_': np.eye(2)}
    return M0,M1,R,a,alphas
abs1a = standardA_M0chainSTC_M1chainSC
basic_lung_cancer = standardA_M0chainSTC_M1chainSC

def standardA_M0chainSC_M1indepSC():
    M0 = M_chain_ST_a
    M1 = M_pgmpy_indep_SC()    
    R = ['Smoking_','Cancer_']
    a = {'Smoking_': 'Smoking__',
        'Cancer_': 'Cancer__'}
    alphas = {'Smoking__': np.eye(2),
             'Cancer__': np.eye(2)}
    return M0,M1,R,a,alphas
abs2a = standardA_M0chainSC_M1indepSC

def partitioningA_M0chainSTC_M1chainSC_S_TC():
    M0 = M_chain_STC_a
    M1 = M_chain_ST_a    
    R = ['Smoking','Tar','Cancer']
    a = {'Smoking': 'Smoking_',
         'Tar': 'Cancer_',
        'Cancer': 'Cancer_'}
    alphas = {'Smoking_': np.eye(2),
             'Cancer_': np.array([[1,0,1,0],[0,1,0,1.]])}    
    return M0,M1,R,a,alphas
abs1b = partitioningA_M0chainSTC_M1chainSC_S_TC

def partitioningA_M0chainSTC_M1chainSC_ST_C():
    M0 = M_chain_STC_a
    M1 = M_chain_ST_a   
    R = ['Smoking','Tar','Cancer']
    a = {'Smoking': 'Smoking_',
         'Tar': 'Smoking_',
        'Cancer': 'Cancer_'}
    alphas = {'Smoking_': np.array([[1,1,0,0],[0,0,1,1.]]),
             'Cancer_': np.eye(2)}    
    return M0,M1,R,a,alphas
abs1c = partitioningA_M0chainSTC_M1chainSC_ST_C

def partitioningA_M0chainSTC_M1chainSCb_ST_C():
    M0 = M_chain_STC_a
    M1 = M_chain_ST_b    
    R = ['Smoking','Tar','Cancer']
    a = {'Smoking': 'Smoking_',
         'Tar': 'Smoking_',
        'Cancer': 'Cancer_'}
    alphas = {'Smoking_': np.array([[1,1,0,0],[0,0,1,1.]]),
             'Cancer_': np.eye(2)}   
    return M0,M1,R,a,alphas

def singletonA_M0chainSTC_star_STC():
    M0 = M_chain_STC_a
    M1 = M_pgmpy_singleton()    
    R = ['Smoking','Tar','Cancer']
    a = {'Smoking': '*',
         'Tar': '*',
        'Cancer': '*'}
    alphas = {'*': np.ones(shape=(1,8))}    
    return M0,M1,R,a,alphas
abs1s = singletonA_M0chainSTC_star_STC

def singletonA_M0chainSTC_star_TC():
    M0 = M_chain_STC_a
    M1 = M_pgmpy_singleton()    
    R = ['Tar','Cancer']
    a = {'Tar': '*',
         'Cancer': '*'}
    alphas = {'*': np.ones(shape=(1,4))}    
    return M0,M1,R,a,alphas

def singletonA_M0chainSTC_star_C():
    M0 = M_chain_STC_a
    M1 = M_pgmpy_singleton()    
    R = ['Cancer']
    a = {'Cancer': '*'}
    alphas = {'*': np.ones(shape=(1,2))}   
    return M0,M1,R,a,alphas

def identityA_M0chainSTC_M0chainSTC():
    M0 = M_chain_STC_a
    M1 = M_chain_STC_a   
    R = ['Smoking','Tar','Cancer']
    a = {'Smoking': 'Smoking',
         'Tar': 'Tar',
        'Cancer': 'Cancer'}
    alphas = {'Smoking': np.eye(2),
              'Tar': np.eye(2),
             'Cancer': np.eye(2)}
    return M0,M1,R,a,alphas
abs1z = identityA_M0chainSTC_M0chainSTC

def nonobservationalA_M0chainSTC_M1chainSCc():
    M0 = M_chain_STC_a
    M1 = M_chain_ST_c    
    R = ['Smoking','Cancer']
    a = {'Smoking': 'Smoking_',
        'Cancer': 'Cancer_'}
    alphas = {'Smoking_': np.eye(2),
             'Cancer_': np.eye(2)}    
    return M0,M1,R,a,alphas
abs1d = nonobservationalA_M0chainSTC_M1chainSCc

def nonobservationalA_M0chainSTC_M1chainSCd():
    M0 = M_chain_STC_a
    M1 = M_chain_ST_d    
    R = ['Smoking','Cancer']
    a = {'Smoking': 'Smoking_',
        'Cancer': 'Cancer_'}
    alphas = {'Smoking_': np.eye(2),
             'Cancer_': np.eye(2)}    
    return M0,M1,R,a,alphas
abs1e = nonobservationalA_M0chainSTC_M1chainSCd

def largerdomainA_M0chainSTCb_M1chainST_a():
    M0 = M_chain_STC_b
    M1 = M_chain_ST_a    
    R = ['Smoking','Cancer']
    a = {'Smoking': 'Smoking_',
        'Cancer': 'Cancer_'}
    alphas = {'Smoking_': np.eye(2),
             'Cancer_': np.eye(2)}   
    return M0,M1,R,a,alphas
abs3a = largerdomainA_M0chainSTCb_M1chainST_a

def largerdomainA_M0chainSTCb_M1chainST_b():
    M0 = M_chain_STC_b
    M1 = M_chain_ST_a  
    R = ['Smoking','Tar','Cancer']
    a = {'Smoking': 'Smoking_',
         'Tar': 'Cancer_',
        'Cancer': 'Cancer_'}
    alphas = {'Smoking_': np.eye(2),
             'Cancer_': np.array([[1,0,1,0,1,0.],[0,1.,0,1.,0,1]])}   
    return M0,M1,R,a,alphas
abs3b = largerdomainA_M0chainSTCb_M1chainST_b

def largerdomainA_M0chainSTCb_M1chainST_e():
    M0 = M_chain_STC_c
    M1 = M_chain_ST_e    
    R = ['Smoking','Cancer']
    a = {'Smoking': 'Smoking_',
        'Cancer': 'Cancer_'}
    alphas = {'Smoking_': np.eye(2),
             'Cancer_': np.array([[1,0,0],[0,1.,1.]])}    
    return M0,M1,R,a,alphas
abs4a = largerdomainA_M0chainSTCb_M1chainST_e

def largerdomainA_M0chainSTCb_M1chainST_f():
    M0 = M_chain_STC_c
    M1 = M_chain_ST_e    
    R = ['Smoking','Tar','Cancer']
    a = {'Smoking': 'Smoking_',
         'Tar': 'Cancer_',
        'Cancer': 'Cancer_'}
    alphas = {'Smoking_': np.eye(2),
             'Cancer_': np.array([[1,1,0,1,0,0.],[0,0.,1,0.,1,1]])}    
    return M0,M1,R,a,alphas
abs4b = largerdomainA_M0chainSTCb_M1chainST_f

def partitioningA_M0chainESTC_M1chainSTC_ES_T_C():
    M0 = M_chain_ESTC_a
    M1 = M_chain_STC_d
    R = ['Environment','Smoking','Tar','Cancer']
    a = {'Environment': 'Smoking_',
         'Smoking': 'Smoking_',
         'Tar': 'Tar_',
        'Cancer': 'Cancer_'}
    alphas = {'Smoking_': np.array([[1,0,1,0],[0,1,0,1.]]),
              'Tar_': np.eye(2),
             'Cancer_': np.eye(2)}    
    return M0,M1,R,a,alphas
collapsing_lung_cancer = partitioningA_M0chainESTC_M1chainSTC_ES_T_C

def complexA_M0vstructSGCHF_M1chainSCF():
    M0 = M_vstruct_SGCFH_a
    M1 = M_chain_SCF_a
    R = ['Smoking','Cancer','Coughing','Fatigue']
    a = {'Smoking': 'Smoking_',
        'Cancer': 'Cancer_',
        'Coughing': 'Fatigue_',
        'Fatigue': 'Fatigue_'}
    alphas = {'Smoking_': np.array([[0,1],[1,0]]),
             'Cancer_': np.eye(2),
             'Fatigue_': np.array([[0,1,0,1],[1,0,1.,0]])}    
    return M0,M1,R,a,alphas
vstruct_lung_cancer = complexA_M0vstructSGCHF_M1chainSCF

def A_multidiagramtest(alphas=None):
    M0 = M0_multidiagramtest
    M1 = M1_multidiagramtest
    R = ['Smoking','Tar','Cancer']
    a = {'Smoking': 'Smoking_',
         'Tar': 'Tar_',
         'Cancer': 'Cancer_'}
    if alphas is None:
        alphas = {"Smoking_": np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,1]]),
                 "Tar_": np.array([[1,1,0],
                        [0,0,1]]),
                 "Cancer_": np.array([[1,0],
                        [0,1]])}
    return M0,M1,R,a,alphas
extended_lung_cancer = A_multidiagramtest






