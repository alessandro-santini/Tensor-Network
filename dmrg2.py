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

def local_minimization(M,L,H,R,nsteps=10):
    Afunc = lambda x: apply_Heff(L, H, R, x.reshape(M.shape)).ravel()
    return optimize_lanczos(Afunc, M.ravel(), nsteps)

class DMRG2:
    def __init__(self, H):
        chim = 200
        self.MPS = MPS.MPS(H.L, chim, H.d)
        self.MPO  = H
        # check H.L == MPS.L
        self.L = self.MPS.L
        self.E = 0.
        self.chi_MAX = 256
        
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

    def initialize_MPS(self,MPS_handle):
        # Generate a randomMPS and put it in right
        # canonical form
        self.MPS = MPS_handle
        self.MPS.right_normalize()
        L = self.L

        self.RT = [0 for x in range(self.L+1)]
        self.LT = [0 for x in range(self.L+1)]
        
        self.RT[L]  = np.ones((1,1,1))
        self.LT[-1] = np.ones((1,1,1))
        
        # Generates R tensors
        for j in range(L-1,0,-1):
            self.RT[j] = contract.contract_right(self.MPS.M[j], self.MPO.W[j], self.MPS.M[j].conj(), self.RT[j+1])

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
        for i in range(self.L-1):
            shpMi = self.MPS.M[i].shape
            shpMj = self.MPS.M[i+1].shape
            Mij = ncon([self.MPS.M[i],self.MPS.M[i+1]],[[-1,-2,1],[1,-3,-4]])
            Mij = Mij.reshape(shpMi[0],shpMi[1]*shpMj[1],shpMj[2])
            
            shpHi = self.MPO.W[i].shape
            shpHj = self.MPO.W[i+1].shape
            
            Hij = ncon([self.MPO.W[i],self.MPO.W[i+1]],[[-1,1,-3,-5],[1,-2,-4,-6]])
            Hij = Hij.reshape(shpHi[0],shpHj[1],shpHi[2]*shpHj[2],shpHi[3]*shpHj[3])
            
            psi, e = local_minimization(Mij, self.LT[i-1], Hij, self.RT[i+2],nsteps=self.kry_dim)
            
            U,S,V = LA.svd(psi.reshape(shpMi[0]*shpMi[1],shpMj[1]*shpMj[2]),full_matrices=False)
            
            S /= np.linalg.norm(S)
            
            S = S[S>1e-16]
            chi = S.size
            
            indices = np.where( (1-np.cumsum(S**2) < self.etrunc ))[0]
            if len(indices) > 0:
                    chi = indices[0]+1
            else:
                    chi = S.size
            if chi > self.chi_MAX:
                chi = self.chi_MAX
                self.end_max = True
                            
            chi = np.min([chi,self.chi_MAX])
            # Truncation
            U = U[:,:chi]
            S = S[:chi]
            V = V[:chi,:]
            
            S /= LA.norm(S)
            
            A = U.reshape(shpMi[0],shpMi[1],S.size)
            self.LT[i]  = contract.contract_left(A, self.MPO.W[i], A.conj(), self.LT[i-1])
            self.MPS.M[i] = A
            self.MPS.M[i+1] = (np.diag(S)@V).reshape(S.size,shpMj[1],shpMj[2])
            self.E = e            
    
    def left_sweep(self):
        for i in range(self.L-1,1,-1):
            shpMi = self.MPS.M[i-1].shape
            shpMj = self.MPS.M[i].shape
            Mij = ncon([self.MPS.M[i-1],self.MPS.M[i]],[[-1,-2,1],[1,-3,-4]])
            Mij = Mij.reshape(shpMi[0],shpMi[1]*shpMj[1],shpMj[2])
            
            shpHi = self.MPO.W[i-1].shape
            shpHj = self.MPO.W[i].shape
            
            Hij = ncon([self.MPO.W[i-1],self.MPO.W[i]],[[-1,1,-3,-5],[1,-2,-4,-6]])
            Hij = Hij.reshape(shpHi[0],shpHj[1],shpHi[2]*shpHj[2],shpHi[3]*shpHj[3])
            
            psi, e = local_minimization(Mij, self.LT[i-2], Hij, self.RT[i+1],nsteps=self.kry_dim)
            
            U,S,V = LA.svd(psi.reshape(shpMi[0]*shpMi[1],shpMj[1]*shpMj[2]),full_matrices=False)
            
            S /= np.linalg.norm(S)
            
            S = S[S>1e-16]
            chi = S.size
            
            indices = np.where( (1-np.cumsum(S**2) < self.etrunc ))[0]
            if len(indices) > 0:
                    chi = indices[0]+1
            else:
                    chi = S.size
            if chi > self.chi_MAX:
                chi = self.chi_MAX
                self.end_max = True

            chi = np.min([chi,self.chi_MAX])
            # Truncation
            U = U[:,:chi]
            S = S[:chi]
            V = V[:chi,:]
            
            S /= LA.norm(S)

            self.MPS.Svr[i+1] = S
            B = V.reshape(S.size,shpMj[1],shpMj[2])
            self.RT[i]  = contract.contract_right(B, self.MPO.W[i], B.conj(), self.RT[i+1])
            self.MPS.M[i] = B
            self.MPS.M[i-1] = (U@np.diag(S)).reshape(shpMi[0],shpMi[1],S.size)
            self.E = e
            
    def dmrg_step(self,etrunc):
        self.etrunc = etrunc
        self.kry_dim = 10
        self.right_sweep()
        self.left_sweep()