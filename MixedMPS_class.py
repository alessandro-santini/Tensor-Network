import numpy as np
import numpy.linalg as LA
from ncon import ncon

import contraction_utilities as contract

class MPS:
    def __init__(self,L,d):
        # L: length of the tensor train
        # chim: maximum bond dimension
        # d: local Hilbert Space dimension
        
        #   Index order
        #   0-- M --3
        #      | |
        #      1 2
        self.L = L
        self.d = d
        self.M = [0 for x in range(L)]
        # Singular-Values
        self.Svr = [0 for x in range(L+1)]
        self.Svr[0] = np.array([1])

    def set_Identity(self):
        d = self.d
        L = self.L
        for i in range(L):
            self.M[i] = 1/np.sqrt(d)*np.eye(d,dtype=complex).reshape(1,d,d,1)

    def initializeMPS(self, chi, kraus = 1):
        """Initialize a random MPS with bond dimension chi
        local hilbert space dim d and length L"""
        d = self.d
        L = self.L
        self.M[0]   = np.random.rand(1,   d,kraus, chi)
        self.M[L-1] = np.random.rand(chi, d,kraus, 1)
        for i in range(1,L-1):
            self.M[i] = np.random.rand(chi,d,kraus,chi)   
    
    def right_normalize(self):
        for i in range(self.L-1,-1,-1):
            M = self.M[i]
            shpM = M.shape
            U, S, V = LA.svd(M.reshape(shpM[0], shpM[1]*shpM[2]*shpM[3]), full_matrices=False)
            S /= LA.norm(S)
            self.M[i] = V.reshape(S.size, shpM[1],shpM[2], shpM[3])
            if i != 0:
                self.M[i-1] = ncon([self.M[i-1],U*S],[[-1,-2,-3,1],[1,-4]])
            self.Svr[i+1] = np.array(S)
    
    def left_normalize(self):
        for i in range(0, self.L-1):
            M = self.M[i]
            shpM = M.shape
            U, S, V = LA.svd(M.reshape(shpM[0]*shpM[1]*shpM[2], shpM[3]), full_matrices=False)
            S /= LA.norm(S)
            self.M[i] =  U.reshape(shpM[0], shpM[1],shpM[2], S.size)
            self.M[i+1] = ncon([np.diag(S)@V, self.M[i+1]],[[-1,1],[1,-2,-3,-4]])
    
    def mix_normalize(self, j):
        for i in range(0, j):
            M = self.M[i]
            shpM = M.shape
            U, S, V = LA.svd(M.reshape(shpM[0]*shpM[1], shpM[2]), full_matrices=False)
            S /= LA.norm(S)
            self.M[i] =  U.reshape(shpM[0], shpM[1], S.size)
            self.M[i+1] = ncon([np.diag(S)@V, self.M[i+1]],[[-1,1],[1,-2,-3]])            
        for i in range(self.L-1,j,-1):
            M = self.M[i]
            shpM = M.shape
            U, S, V = LA.svd(M.reshape(shpM[0], shpM[1]*shpM[2]), full_matrices=False)
            S /= LA.norm(S)
            self.M[i] = V.reshape(S.size, shpM[1], shpM[2])
            self.M[i-1] = ncon([self.M[i-1],U*S],[[-1,-2,1],[1,-3]])
    
    
    def check_normalization(self, which='R'):
        if which == 'R':
            for i in range(self.L):
                shpM = self.M[i].shape
                K = self.M[i].reshape(shpM[0], shpM[1]*shpM[2], shpM[3])
                X = [K[:,j,:]@K[:,j,:].T.conj() for j in range(shpM[1]*shpM[2])]
                print('site',i,np.allclose(sum(X),np.eye(self.M[i].shape[0]))) 
        if which == 'L':
            for i in range(self.L):
                shpM = self.M[i].shape
                K = self.M[i].reshape(shpM[0],shpM[1]*shpM[2], shpM[3])
                X = [K[:,j,:].T.conj()@K[:,j,:] for j in range(shpM[1]*shpM[2])]
                print('site',i,np.allclose(sum(X),np.eye(self.M[i].shape[3])))
    
    def compute_EntEntropy(self):
        Sent = np.zeros(self.L-1)
        Mlist = self.M.copy()
        
        for i in range(0, self.L-1):
            M = Mlist[i]
            shpM = M.shape
            U, S, V = LA.svd(M.reshape(shpM[0]*shpM[1]*shpM[2], shpM[3]), full_matrices=False)
            S /= LA.norm(S)
            Mlist[i] =  U.reshape(shpM[0], shpM[1],shpM[2], S.size)
            if i!= self.L-1:
                Mlist[i+1] = ncon([np.diag(S)@V, Mlist[i+1]],[[-1,1],[1,-2,-3,-4]])
            Sent[i] = (-S**2*np.log(S**2)).sum()
        return Sent
    
    def save_hdf5(self,file_pointer, n):
        subgroup = "/MPS/n/%d/M"%(n)
        file_pointer.create_group(subgroup)
        for idx, arr in enumerate(self.M):
            file_pointer.create_dataset(subgroup+'/'+str(idx), shape=arr.shape, data=arr,compression='gzip',compression_opts=9)

    def load_hdf5(self, file_pointer, n):
        subgroup = "/MPS/n/%d/M"%(n)
        for idx in range(self.L):
            self.M[idx] = file_pointer[subgroup+'/'+str(idx)][...].copy()

    
    def contractMPOmixMPS(self, MPO):
        if(self.L != MPO.L): raise Exception('MPS MPO length are different')
        Rtemp = np.ones((1,1,1),dtype=np.complex128)
        for i in range(self.L-1,0,-1):
            Rtemp = contract.mix_contract_right(self.M[i], MPO.W[i], self.M[i].conj(), Rtemp)
        return contract.mix_contract_right(self.M[0], MPO.W[0], self.M[0].conj(), Rtemp)[0][0][0]
            
def svdtruncate(mat,etrunc,chiMAX,info=True):
    U,S,V = np.linalg.svd(mat,full_matrices=False)
    S /= np.linalg.norm(S)
    S = S[S>1e-16]
    chi = S.size
    if info == True:   
        indices = np.where( (1-np.cumsum(S**2) < etrunc ))[0]
        if len(indices) > 0:
            chi = indices[0]+1
        else:
            chi = S.size
        if chi > chiMAX:
            chi = chiMAX
    U = U[:,:chi]
    S = S[:chi]
    V = V[:chi,:]
    S /= np.linalg.norm(S)
    return U,S,V


def mix_compute_corr(MPS_,op):
    "Computes \sum_i,j <op_i op_j> - <op_i><op_j>"
    L = MPS_.L
    MPS_temp = MPS(L,2)
    MPS_temp.M = MPS_.M.copy()
    
    op1 = np.zeros(L).reshape(L,1)
    op2 = np.zeros((L,L))
    
    opTEN = op.reshape(1,1,op.shape[0],op.shape[1])
    for i in range(L):
        if i != 0:
            shpM1 = MPS_temp.M[i-1].shape; shpM2 = MPS_temp.M[i].shape
            M1M2  = ncon([MPS_temp.M[i-1],MPS_temp.M[i]],[[-1,-2,-3,1],[1,-4,-5,-6]])
            M1M2  = M1M2.reshape(shpM1[0]*shpM1[1]*shpM1[2],shpM2[1]*shpM2[2]*shpM2[3])
            U,S,V = LA.svd(M1M2,full_matrices=False)
            MPS_temp.M[i-1] = U.reshape(shpM1[0],shpM1[1],shpM1[2],S.size)
            MPS_temp.M[i]   = (np.diag(S)@V).reshape(S.size,shpM2[1],shpM2[2],shpM2[3])
        op1[i] = ncon([MPS_temp.M[i],op,MPS_temp.M[i].conjugate()],\
                      [[1,4,3,2],[4,5],[1,5,3,2]]).real.item()
        op2[i,i] = ncon([MPS_temp.M[i],op,op,MPS_temp.M[i].conjugate()],\
                      [[1,4,3,2],[4,5],[5,6],[1,6,3,2]]).real.item()
        for j in range(i+1,L):
            Ltemp = ncon([MPS_temp.M[i],op.reshape(op.shape[0],op.shape[1],1),MPS_temp.M[i].conjugate()],\
                         [[1,2,4,-1],[2,3,-2],[1,3,4,-3]])
            for x in range(i+1,j):
                Ltemp = contract.mix_contract_left(MPS_temp.M[x], np.eye(2).reshape(1,1,2,2), MPS_temp.M[x].conj(), Ltemp)
            Ltemp = contract.mix_contract_left(MPS_temp.M[j], opTEN, MPS_temp.M[j].conj(), Ltemp)
            op2[i,j] = ncon(Ltemp,[1,-1,1]).real.item()
            op2[j,i] = op2[i,j]
    
    G = 1/L*( op2 - op1@op1.T ).sum()
    
    return G, (op1,op2)