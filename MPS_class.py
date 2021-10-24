import numpy as np
import numpy.linalg as LA
from ncon import ncon

class MPS:
    def __init__(self,L,chim,d):
        # L: length of the tensor train
        # chim: maximum bond dimension
        # d: local Hilbert Space dimension
        
        #   Index order
        #   0--M--2
        #      |
        #      1
        self.L = L
        self.chim = chim
        self.d = d
        self.M = [0 for x in range(L)]
        # Singular-Values
        self.Svr = [0 for x in range(L+1)]
        self.Svr[0] = np.array([1])
        
    def initializeMPS(self, chi):
        """Initialize a random MPS with bond dimension chi
        local hilbert space dim d and length L"""
        d = self.d
        L = self.L
        self.M[0]   = np.random.rand(1,   d, chi)
        self.M[L-1] = np.random.rand(chi, d, 1)
        for i in range(1,L-1):
            self.M[i] = np.random.rand(chi,d,chi)   
    
    def right_normalize(self):
        for i in range(self.L-1,-1,-1):
            M = self.M[i]
            shpM = M.shape
            U, S, V = LA.svd(M.reshape(shpM[0], shpM[1]*shpM[2]), full_matrices=False)
            S /= LA.norm(S)
            self.M[i] = V.reshape(S.size, shpM[1], shpM[2])
            if i != 0:
                self.M[i-1] = ncon([self.M[i-1],U*S],[[-1,-2,1],[1,-3]])
            self.Svr[i+1] = np.array(S)
            
    def right_normalize_and_truncate(self,chi):
        for i in range(self.L-1,0,-1):
            M = self.M[i]
            shpM = M.shape
            U, S, V = LA.svd(M.reshape(shpM[0], shpM[1]*shpM[2]), full_matrices=False)
            if S.size>chi:
                U = U[:, :chi]
                V = V[:chi, :]
                S = S[:chi]
            S /= LA.norm(S)
            self.M[i] = V.reshape(S.size, shpM[1], shpM[2])
            self.M[i-1] = ncon([self.M[i-1],U*S],[[-1,-2,1],[1,-3]])
    
    def left_normalize(self):
        for i in range(0, self.L-1):
            M = self.M[i]
            shpM = M.shape
            U, S, V = LA.svd(M.reshape(shpM[0]*shpM[1], shpM[2]), full_matrices=False)
            S /= LA.norm(S)
            self.M[i] =  U.reshape(shpM[0], shpM[1], S.size)
            self.M[i+1] = ncon([np.diag(S)@V, self.M[i+1]],[[-1,1],[1,-2,-3]])
    
    def swap(self, i, chiMAX=512, etrunc=1e-10,center='i+1',info=True):
        # CENTER HAS TO BE IN I1 OR I2
        i1 = i; i2=i+1;
        M1, M2 = self.M[i1], self.M[i2]
        shpM1, shpM2 = M1.shape, M2.shape
        Swapped_M1M2 = ncon([M1, M2], [[-1,-3,1],[1,-2,-4]]).reshape(shpM1[0]*shpM2[1],shpM1[1]*shpM2[2])
        U,S,V = svdtruncate(Swapped_M1M2, etrunc, chiMAX,info)
        if center == 'i+1':
            M1 = (U).reshape(shpM1[0],shpM2[1],S.size)
            M2 = (np.diag(S)@V).reshape(S.size,shpM1[1],shpM2[2])
        elif center == 'i':
            M1 = (U@np.diag(S)).reshape(shpM1[0],shpM2[1],S.size)
            M2 = (V).reshape(S.size,shpM1[1],shpM2[2])
        self.M[i1] = M1
        self.M[i2] = M2        
    
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
                X = [self.M[i][:,j,:]@self.M[i][:,j,:].T.conj() for j in range(self.d)]
                print('site',i,np.allclose(sum(X),np.eye(self.M[i].shape[0]))) 
        if which == 'L':
            for i in range(self.L):
                X = [self.M[i][:,j,:].T.conj()@self.M[i][:,j,:] for j in range(self.d)]
                print('site',i,np.allclose(sum(X),np.eye(self.M[i].shape[2])))
    
    def compute_EntEntropy(self):
        Sent = np.zeros(self.L-1)
        Mlist = self.M.copy()
        
        for i in range(0, self.L-1):
            M = Mlist[i]
            shpM = M.shape
            U, S, V = LA.svd(M.reshape(shpM[0]*shpM[1], shpM[2]), full_matrices=False)
            S /= LA.norm(S)
            Mlist[i] =  U.reshape(shpM[0], shpM[1], S.size)
            if i!= self.L-1:
                Mlist[i+1] = ncon([np.diag(S)@V, Mlist[i+1]],[[-1,1],[1,-2,-3]])
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
        
def getAllUp(L):
    res = MPS(L,2,2)
    for x in range(L):
        res.M[x] = np.zeros((1,2,1))
        res.M[x][0,0,0] = 1
    return res