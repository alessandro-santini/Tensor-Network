import MPS_class as MPS
import numpy as np
import numpy.linalg as LA
from ncon import ncon
import contraction_utilities as contract


def compute_corr(MPS_,op):
    " Computes \sum_i,j <op_i op_j> - <op_i><op_j>"
    L = MPS_.L
    MPS_temp = MPS.MPS(L,2,2)
    MPS_temp.M = MPS_.M.copy()
    
    op1 = np.zeros(L).reshape(L,1)
    op2 = np.zeros((L,L))
    
    opTEN = op.reshape(1,1,op.shape[0],op.shape[1])
    for i in range(L):
        if i != 0:
            shpM1 = MPS_temp.M[i-1].shape; shpM2 = MPS_temp.M[i].shape
            M1M2  = ncon([MPS_temp.M[i-1],MPS_temp.M[i]],[[-1,-2,1],[1,-3,-4]])
            M1M2  = M1M2.reshape(shpM1[0]*shpM1[1],shpM2[1]*shpM2[2])
            U,S,V = LA.svd(M1M2,full_matrices=False)
            MPS_temp.M[i-1] = U.reshape(shpM1[0],shpM1[1],S.size)
            MPS_temp.M[i] = (np.diag(S)@V).reshape(S.size,shpM2[1],shpM2[2])
        op1[i] = ncon([MPS_temp.M[i],op,MPS_temp.M[i].conjugate()],\
                      [[1,3,2],[3,4],[1,4,2]]).real.item()
        op2[i,i] = ncon([MPS_temp.M[i],op,op,MPS_temp.M[i].conjugate()],\
                      [[1,3,2],[3,4],[4,5],[1,5,2]]).real.item()
        for j in range(i+1,L):
            Ltemp = ncon([MPS_temp.M[i],op.reshape(op.shape[0],op.shape[1],1),MPS_temp.M[i].conjugate()],\
                         [[1,2,-1],[2,3,-2],[1,3,-3]])
            for x in range(i+1,j):
                Ltemp = contract.contract_left(MPS_temp.M[x], np.eye(2).reshape(1,1,2,2), MPS_temp.M[x].conj(), Ltemp)
            Ltemp = contract.contract_left(MPS_temp.M[j], opTEN, MPS_temp.M[j].conj(), Ltemp)
            op2[i,j] = ncon(Ltemp,[1,-1,1]).real.item()
            op2[j,i] = op2[i,j]
    
    G = 1/L**2*( op2 - op1@op1.T ).sum()
    
    return G, (op1,op2)