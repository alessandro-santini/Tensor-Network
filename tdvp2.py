import MPS_class as MPS

import contraction_utilities as contract
import numpy as np
import numpy.linalg as LA
from ncon import ncon

from LanczosRoutines import expm_krylov_lanczos

## tensor contraction for minimization
##         +--M--+
##         |  |  |
## H_eff = L--H--R
##         |  |  |
##         +--M--+       
def apply_Heff2(L,H,R,M):
    return ncon([L, H, R,M], [[1,2,-1],[2,6,3,4,-2,-3],[5,6,-4],[1,3,4,5]])
def apply_Heff1(L,H,R,M):
    return ncon([L,H,R,M],[[1,2,-1],[2,5,3,-2],[4,5,-3],[1,3,4]])

def local_exponentiation(method,M,L,H,R,delta,maxit=10):
    if method == 'H2':
            Afunc = lambda x: apply_Heff2(L, H, R, x.reshape(M.shape)).ravel()
    if method == 'H1':
            Afunc = lambda x: apply_Heff1(L, H, R, x.reshape(M.shape)).ravel()
    v = expm_krylov_lanczos(Afunc, M.ravel(), 1j*delta/2, maxit)
    return v/LA.norm(v)
    
class TDVP2:
    def __init__(self,MPS_,H,chi_MAX=64,chi_min=0,truncate_info=True):
        self.MPS = MPS.MPS(MPS_.L,MPS_.chim,MPS_.d)
        self.MPS.M = MPS_.M.copy()
        self.MPO = H
        self.L = self.MPS.L
        self.chi_MAX = chi_MAX
        self.end_max = False
        self.chi_min = chi_min
        self.truncate_info = truncate_info
        
        self.W12 = [0 for i in range(self.L-1)]
        for i in range(self.L-1):
            self.W12[i] = ncon([H.W[i],H.W[i+1]],[[-1,1,-3,-5],[1,-2,-4,-6]])

    def initialize(self):
        L = self.L

        self.RT = [0 for x in range(self.L+1)]
        self.LT = [0 for x in range(self.L+1)]
        
        self.RT[L]  = np.ones((1,1,1))
        self.LT[-1] = np.ones((1,1,1))
        
        # Generates R tensors
        for j in range(L-1,1,-1):
            self.RT[j] = contract.contract_right(self.MPS.M[j], self.MPO.W[j], self.MPS.M[j].conj(), self.RT[j+1])
       
    def right_sweep(self,delta,krydim=10):
        for i in range(self.L-1):
            
            M = ncon([self.MPS.M[i],self.MPS.M[i+1]],[[-1,-2,1],[1,-3,-4]])
            
            shpMi = self.MPS.M[i].shape
            shpMj = self.MPS.M[i+1].shape
            
            psi = local_exponentiation('H2',M, self.LT[i-1], self.W12[i], self.RT[i+2], -delta,krydim)
            
            M = psi.reshape(shpMi[0]*shpMi[1],shpMj[1]*shpMj[2])
            
            U,S,V = LA.svd(M, full_matrices=False)
            S /= np.linalg.norm(S)
            
            S = S[S>1e-16]
            chi = S.size
            if self.truncate_info:
                indices = np.where( (1-np.cumsum(S**2) < self.etrunc ))[0]
                if len(indices) > 0:
                      chi = indices[0]+1
                else:
                        chi = S.size
                if chi > self.chi_MAX:
                    chi = self.chi_MAX
                    self.end_max = True
                chi = np.max([chi,self.chi_min])
            
            chi = np.min([chi,self.chi_MAX])
            # Truncation
            U = U[:,:chi]
            S = S[:chi]
            V = V[:chi,:]
            
            S /= LA.norm(S)
            A = U.reshape(shpMi[0],shpMi[1], S.size)
            
            self.MPS.M[i] = A
            self.MPS.M[i+1] = (np.diag(S)@V).reshape(S.size,shpMj[1],shpMj[2])
            
            if i != self.L-2:
                self.LT[i]  = contract.contract_left(A, self.MPO.W[i], A.conj(), self.LT[i-1])
                shpMj = self.MPS.M[i+1].shape
                self.MPS.M[i+1] = local_exponentiation('H1', self.MPS.M[i+1], self.LT[i], self.MPO.W[i+1], self.RT[i+2], delta,krydim).reshape(shpMj)
            
    def left_sweep(self,delta,krydim):
        for i in range(self.L-1, 0, -1):
            
            M = ncon([self.MPS.M[i-1],self.MPS.M[i]],[[-1,-2,1],[1,-3,-4]])
            shpMi = self.MPS.M[i-1].shape
            shpMj = self.MPS.M[i].shape
            
            psi = local_exponentiation('H2',M, self.LT[i-2], self.W12[i-1], self.RT[i+1],-delta,krydim)         
            M = psi.reshape(shpMi[0]*shpMi[1],shpMj[1]*shpMj[2])
            
            U,S,V = LA.svd(M,full_matrices=False)
            S /= LA.norm(S)
            
            S = S[S>1e-16]
            chi = S.size
            if self.truncate_info:
                indices = np.where( (1-np.cumsum(S**2) < self.etrunc ))[0]
                if len(indices) > 0:
                      chi = indices[0]+1
                else:
                        chi = S.size
                if chi > self.chi_MAX:
                    chi = self.chi_MAX
                    self.end_max = True
                chi = np.max([chi,self.chi_min])
            
            chi = np.min([chi,self.chi_MAX])
            # Truncation
            U = U[:,:chi]
            S = S[:chi]
            V = V[:chi,:]
            S /= LA.norm(S)
            B  = V.reshape(S.size, shpMj[1], shpMj[2])
            
            self.MPS.M[i] = B
            self.MPS.M[i-1] = (U@np.diag(S)).reshape(shpMi[0], shpMi[1], S.size)
            
            if i != 1:
                self.RT[i]  = contract.contract_right(B, self.MPO.W[i], B.conj(), self.RT[i+1])
                shpMi = self.MPS.M[i-1].shape
                self.MPS.M[i-1] = local_exponentiation('H1',self.MPS.M[i-1], self.LT[i-2],self.MPO.W[i-1], self.RT[i],delta,krydim).reshape(shpMi)       
                
    def time_step(self, delta, etrunc, krydim=10):
        self.etrunc = etrunc
        self.right_sweep(delta,krydim)
        self.left_sweep(delta,krydim)