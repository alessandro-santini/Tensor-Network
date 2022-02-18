import MPS_class as MPS
import MPO_class as MPO
import contraction_utilities as contract

import numpy as np
import numpy.linalg as LA
from ncon import ncon

from LanczosRoutines import optimize_lanczos

## tensor contraction for minimization
##         +--M--+
##         |  |  |
## H_eff = L--H--R
##         |  |  |
##         +--M--+       
def apply_Heff(L,H,R,M):
    return ncon([L,H,R,M.conj()],[[-1,2,1],[2,5,-2,3],[-3,5,4],[1,3,4]])
def apply_overlap(L1,Mgs,R1):
    return ncon([L1,Mgs,R1],[[-1,1],[1,-2,2],[-3,2]])
def overlap(L1,Mgs,M,R1):
    return ncon([L1,Mgs,R1,M],[[4,1],[1,5,2],[6,2],[4,5,6]])

def local_minimization(M,L,H,R, Mgs, LT1, RT1,Lsteps=10):
    Afunc = lambda x: apply_Heff(L, H, R, x.reshape(M.shape)).ravel() + overlap(LT1,Mgs,M,RT1)*apply_overlap(LT1,Mgs,RT1).ravel()
    return optimize_lanczos(Afunc, M.ravel(), Lsteps)

class DMRG1:
    def __init__(self, H, psi_gs):
        chim = 200
        self.MPS = MPS.MPS(H.L, chim, H.d)
        self.MPS_gs = psi_gs
        self.MPO  = H
        # check H.L == MPS.L
        self.L = self.MPS.L
        self.E = 0.
        
    def initialize(self,chi):
        # Generate a randomMPS and put it in right
        # canonical form
        self.MPS.initializeMPS(chi)      
        self.MPS.right_normalize()
        L = self.L

        self.RT = [0 for x in range(self.L+1)]
        self.LT = [0 for x in range(self.L+1)]
        
        self.RT[L]  = np.ones((1,1,1))
        self.LT[-1] = np.ones((1,1,1))
        
        # Generates R tensors
        for j in range(L-1,0,-1):
            self.RT[j] = contract.contract_right(self.MPS.M[j], self.MPO.W[j], self.MPS.M[j].conj(), self.RT[j+1])
        
        self.RT1 = [0 for x in range(self.L+1)]
        self.LT1 = [0 for x in range(self.L+1)]
        
        self.RT1[L] = np.ones((1,1))
        self.LT1[-1] = np.ones((1,1))
        
        for j in range(L-1,0,-1):
            self.RT1[j] = ncon([self.MPS.M[j],self.MPS_gs.M[j].conj(),self.RT1[j+1]],[[-1,3,1],[-2,3,2],[1,2]])

    def check_convergence(self):
        self.H2 = MPO.MPO(self.MPO.L,self.MPO.d)
        for x in range(self.L):
            shpW = self.MPO.W[x].shape
            self.H2.W[x] = ncon([self.MPO.W[x],self.MPO.W[x]],[[-1,-3,1,-6],[-2,-4,-5,1]])
            self.H2.W[x] = self.H2.W[x].reshape(shpW[0]*shpW[0],shpW[1]*shpW[1],shpW[2],shpW[3])
        E2 = self.H2.contractMPOMPS(self.MPS)
        E  = self.MPO.contractMPOMPS(self.MPS)
        return (E2 - E**2)

    def right_sweep(self):
        for i in range(self.L):
            
            M = self.MPS.M[i]
            shpM = M.shape
            Mgs = self.MPS_gs.M[i]
            psi, e = local_minimization(M, self.LT[i-1], self.MPO.W[i], self.RT[i+1], Mgs, self.LT1[i-1], self.RT1[i+1])
            
            U,S,V = LA.svd(psi.reshape(shpM[0]*shpM[1],shpM[2]),full_matrices=False)
            S /= LA.norm(S)
            A = U.reshape(shpM[0],shpM[1], S.size)
            self.LT[i]  = contract.contract_left(A, self.MPO.W[i], A.conj(), self.LT[i-1])
            self.LT1[i] = ncon([A,self.MPS_gs.M[i].conj(),self.LT1[i-1]],[[1,3,-1],[2,3,-2],[1,2]])

            self.MPS.M[i] = A
                        
            if i != self.L-1:
                SV = (np.diag(S)@V)
                self.MPS.M[i+1] = ncon([SV, self.MPS.M[i+1]],[[-1,1],[1,-2,-3]])
            self.E = e            
            
    def left_sweep(self):
        for i in range(self.L-1,-1,-1):
            M = self.MPS.M[i]
            shpM = M.shape
            Mgs = self.MPS_gs.M[i]
            psi, e = local_minimization(M, self.LT[i-1], self.MPO.W[i], self.RT[i+1], Mgs, self.LT1[i-1], self.RT1[i+1])
            U,S,V = LA.svd(psi.reshape(shpM[0],shpM[1]*shpM[2]),full_matrices=False)
            S /= LA.norm(S)
            B = V.reshape(S.size,shpM[1],shpM[2])
            self.RT[i]  = contract.contract_right(B, self.MPO.W[i], B.conj(), self.RT[i+1])
            self.RT1[i] = ncon([B,self.MPS_gs.M[i].conj(),self.RT1[i+1]],[[-1,3,1],[-2,3,2],[1,2]])

            self.MPS.M[i] = B
            
            self.MPS.Svr[i+1] = S
            if i != 0:
                US = U@np.diag(S)
                self.MPS.M[i-1] = ncon([self.MPS.M[i-1],US],[[-1,-2,1],[1,-3]])
            self.E = e
            
    def dmrg_step(self):
        self.right_sweep()
        self.left_sweep()