import numpy as np
import numpy.linalg as LA
from ncon import ncon
import contraction_utilities as contract
from scipy.linalg import expm
import MPS_class as MPS_

class MPO:
    def __init__(self,L,d):
        #   Index order
        #       2
        #       |
        #    0--W--1
        #       |
        #       3
        #
        # L: length of the tensor train
        # d: local Hilbert Space dimension
        self.L = L
        self.d = d
        self.W = [0 for x in range(L)]
    
    # M-M-M-M-M
    # | | | | |
    # W-W-W-W-W
    # | | | | |
    # M-M-M-M-M    
    def contractMPOMPS(self, MPS):
       if(MPS.L != self.L): raise Exception('MPS MPO length are different')
       Rtemp = np.ones((1,1,1),dtype=np.complex128)
       for i in range(self.L-1,0,-1):
           Rtemp = contract.contract_right(MPS.M[i], self.W[i], MPS.M[i].conj(), Rtemp)
       return contract.contract_right(MPS.M[0], self.W[0], MPS.M[0].conj(), Rtemp)[0][0][0]
   
    def compressMPO(self, err=0.):
        L = self.L
        for i in range(L-1):
            W = self.W[i].transpose(0, 2, 3, 1)
            shpW = W.shape
            q, r = LA.qr(W.reshape(shpW[0]*shpW[1]*shpW[2],shpW[3]))
            q = q.reshape(shpW[0],shpW[1],shpW[2],q.shape[1]).transpose(0, 3, 1, 2)
            self.W[i] = q
            self.W[i+1] = ncon([r,self.W[i+1]],[[-1,1],[1,-2,-3,-4]])
        
        if False:
            for i in range(L-1,0,-1):
                W = self.W[i].transpose(0,2,3,1)
                shpW = W.shape
                U,S,V = LA.svd(W.reshape(shpW[0],shpW[1]*shpW[2]*shpW[3]),full_matrices=False)
                #S /= np.linalg.norm(S)
                #indices = np.where( (1-np.cumsum(S**2) < err ))[0]
                #if len(indices)>0:
                #    chi = indices[0]+1
                #else:
                #    chi = S.size
                #if S.size > chi:
                #    U = U[:,:chi]
                #    S = S[:chi]
                #    V =  V[:chi,:]
                #S /= np.linalg.norm(S)
                self.W[i] = V.reshape(S.size,shpW[1],shpW[2],shpW[3]).transpose(0,3,1,2)
                self.W[i-1] = ncon([self.W[i-1],(U@np.diag(S))],[[-1,1,-3,-4],[1,-2]])
        
def getMzMPO(L):
    MzMPO = MPO(L,2)
    mz  = np.zeros((2,2,2,2)); mzl = np.zeros((1,2,2,2)); mzr = np.zeros((2,1,2,2))
    sigma_z  = np.array([[1, 0], [0,-1]]);  Id  = np.array([[1, 0], [0, 1]])
    # Bulk
    mz[0,0,:,:] = Id; mz[1,0,:,:] = sigma_z
    mz[1,1,:,:] = Id
    # Boundary
    mzr[0,0,:,:] = Id; mzr[1,0,:,:] = sigma_z
    mzl[0,0,:,:] = sigma_z; mzl[0,1,:,:] = Id    
    # Set the MPO
    MzMPO.W[0] = mzl; MzMPO.W[L-1] = mzr
    for i in range(1,L-1):
        MzMPO.W[i] = mz
    return MzMPO
    
def getMxMPO(L):
    MxMPO = MPO(L,2)
    mx  = np.zeros((2,2,2,2)); mxl = np.zeros((1,2,2,2)); mxr = np.zeros((2,1,2,2))
    sigma_x  = np.array([[0, 1], [1, 0]]);  Id  = np.array([[1, 0], [0, 1]])
    # Bulk
    mx[0,0,:,:] = Id; mx[1,0,:,:] = sigma_x
    mx[1,1,:,:] = Id
    # Boundary
    mxr[0,0,:,:] = Id; mxr[1,0,:,:] = sigma_x
    mxl[0,0,:,:] = sigma_x; mxl[0,1,:,:] = Id    
    # Set the MPO
    MxMPO.W[0] = mxl; MxMPO.W[L-1] = mxr
    for i in range(1,L-1):
        MxMPO.W[i] = mx
    return MxMPO

def getStagMzMPO(L):
    sMzMPO = MPO(L,2)
    mzl = np.zeros((1,2,2,2)); mzr = np.zeros((2,1,2,2))
    sigma_z  = np.array([[1, 0], [0,-1]]);  Id  = np.array([[1, 0], [0, 1]])
    # Boundary
    mzr[0,0,:,:] = Id; 
    mzr[1,0,:,:] = sigma_z*(-1)**(L-1)
    mzl[0,0,:,:] = sigma_z; mzl[0,1,:,:] = Id    
    # Set the MPO
    sMzMPO.W[0] = mzl; 
    sMzMPO.W[L-1] = mzr
    for i in range(1,L-1):
        # Bulk
        mz  = np.zeros((2,2,2,2));
        mz[0,0,:,:] = Id; 
        mz[1,0,:,:] = sigma_z*(-1)**i
        mz[1,1,:,:] = Id
        sMzMPO.W[i] = mz
    return sMzMPO

def return_LocalMz(MPS):
    L = MPS.L
    tempMPS = MPS_.MPS(L,2,2)
    tempMPS.M = MPS.M.copy()
    sigma_z  = np.array([[1, 0], [0,-1]])
    mz = np.zeros(L,complex)
    
    mz[0] = ncon([tempMPS.M[0],sigma_z,tempMPS.M[0].conj()],[[1,2,3],[2,4],[1,4,3]])       
    for i in range(L-1):
            M = tempMPS.M[i]
            shpM = M.shape
            U, S, V = LA.svd(M.reshape(shpM[0]*shpM[1], shpM[2]), full_matrices=False)
            S /= LA.norm(S)
            tempMPS.M[i] =  U.reshape(shpM[0], shpM[1], S.size)
            tempMPS.M[i+1] = ncon([np.diag(S)@V, tempMPS.M[i+1]],[[-1,1],[1,-2,-3]])            
            mz[i+1] = ncon([tempMPS.M[i+1],sigma_z,tempMPS.M[i+1].conj()],[[1,2,3],[2,4],[1,4,3]])       
    return mz

def getLocalMzMPO(L, i):
    LMz = MPO(L, 2)
    sigma_z  = np.array([[1, 0], [0,-1]]);  Id  = np.array([[1, 0], [0, 1]])
    mzl = np.zeros((1,2,2,2)); mzr = np.zeros((2,1,2,2))
    mzr[0,0,:,:] = Id; 
    mzr[1,0,:,:] = Id;
    mzl[0,0,:,:] = Id; 
    mzl[0,1,:,:] = Id    
    
    LMz.W[0] = mzl; 
    LMz.W[L-1] = mzr
    for j in range(1,L-1):
        mz  = np.zeros((2,2,2,2));
        mz[0,0,:,:] = Id; 
        if j == i:
            mz[1,0,:,:] = sigma_z
        else:
            mz[1,0,:,:] = Id
        mz[1,1,:,:] = Id
        LMz.W[j] = mz
    return LMz

def ComputeCorrFunction(MPS,i,j,Opi,Opj):   
    shpMi = MPS.M[i].shape
    shpMj = MPS.M[j].shape
    L = np.zeros((shpMi[0],1,shpMi[0]))
    Rtemp = np.zeros((shpMj[2],1,shpMj[2]))
    
    L[:,0,:] = np.eye(shpMi[0])
    Rtemp[:,0,:] = np.eye(shpMj[2])
    
    Rtemp = contract.contract_right(MPS.M[j],Opj,MPS.M[j].conj(),Rtemp)
    for k in range(j-1, i, -1):
           Rtemp = contract.contract_right(MPS.M[k], np.eye(MPS.d).reshape(1,1,2,2), MPS.M[k].conj(), Rtemp)
    R = contract.contract_right(MPS.M[i], Opi, MPS.M[i].conj(), Rtemp)
    return ncon([L,R],[[1,2,3],[1,2,3]])


        