import MPS_class as MPS
import MPO_class as MPO
from ncon import ncon
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

def TEBD_evo(MPS_,Lx,Ly,J=1,epsilon=0.1,etrunc=0,chiMAX=256,chiMAXswap=256,info=True):
    L = Lx*Ly
    config = np.arange(0,L).reshape(Lx,Ly)
    
    theta = np.pi+2*epsilon
    flip_op = np.eye(2)*np.cos(theta/2) - 1j*np.sin(theta/2)*np.array([[0,1],[1,0]])    
    
    sigma_z = np.array([[1,0],[0,-1]])
    Uprop = expm(-1j*np.kron(sigma_z,-J*sigma_z)).reshape(2,2,2,2)
    nn_list_forward = [[] for x in range(L)]
    for x in range(L):
        i,j = np.where(config == x)
        if j != Ly-1: nn_list_forward[x].append( config[i,j+1])
        if i != Lx-1: nn_list_forward[x].append( config[i+1,j])
        nn_list_forward[x] = np.array(nn_list_forward[x]).ravel()
    nn_list_backward = [[] for x in range(L)]    
    for x in reversed(range(L)):
        i,j = np.where(config == x)
        if j != 0: nn_list_backward[x].append( config[i,j-1])
        if i != 0: nn_list_backward[x].append( config[i-1,j])
        nn_list_backward[x] = np.array(nn_list_backward[x]).ravel()
    
    for x in range(L):
        for nn in nn_list_forward[x]:
            # If they are nearest neighbours
            if nn == x+1:
               shpM1,shpM2 = MPS_.M[x].shape, MPS_.M[nn].shape
               Theta = ncon([MPS_.M[x],MPS_.M[nn],Uprop],[[-1,2,1],[1,3,-4],[2,3,-2,-3]]) 
               Theta = Theta.reshape(shpM1[0]*shpM1[1], shpM2[1]*shpM2[2])
               U,S,V = MPS.svdtruncate(Theta, etrunc, chiMAX,info=info)
               MPS_.M[x]  = U.reshape(shpM1[0],shpM1[1],S.size)
               MPS_.M[nn] = (np.diag(S)@V).reshape(S.size,shpM2[1],shpM2[2])
            else:
                for index in range(x,nn-1):
                    MPS_.swap(index,chiMAX=chiMAXswap,info=info)
                shpM1,shpM2 = MPS_.M[nn-1].shape, MPS_.M[nn].shape
                Theta = ncon([MPS_.M[nn-1],MPS_.M[nn],Uprop],[[-1,2,1],[1,3,-4],[2,3,-2,-3]]) 
                Theta = Theta.reshape(shpM1[0]*shpM1[1], shpM2[1]*shpM2[2])
                U,S,V = MPS.svdtruncate(Theta, etrunc, chiMAX,info=info)
                MPS_.M[nn-1]  = (U@np.diag(S)).reshape(shpM1[0],shpM1[1],S.size)
                MPS_.M[nn] = V.reshape(S.size,shpM2[1],shpM2[2]) 
                for index in reversed(range(x,nn-1)):
                    MPS_.swap(index,chiMAX=chiMAXswap,info=info)
        MPS_.M[x] = ncon([MPS_.M[x],flip_op],[[-1,1,-3],[1,-2]])
    
    for x in reversed(range(L)):
        for nn in nn_list_backward[x]:
            # If they are nearest neighbours
            if nn == x-1:
               shpM1,shpM2 = MPS_.M[nn].shape, MPS_.M[x].shape
               Theta = ncon([MPS_.M[nn],MPS_.M[x],Uprop],[[-1,2,1],[1,3,-4],[2,3,-2,-3]]) 
               Theta = Theta.reshape(shpM1[0]*shpM1[1], shpM2[1]*shpM2[2])
               U,S,V = MPS.svdtruncate(Theta, etrunc, chiMAX,info=info)
               MPS_.M[nn]  = (U@np.diag(S)).reshape(shpM1[0],shpM1[1],S.size)
               MPS_.M[x] = (V).reshape(S.size,shpM2[1],shpM2[2])
            else:
               for index in range(x-1,nn,-1):
                   MPS_.swap(index,chiMAX=chiMAXswap,center='i',info=info)
               shpM1,shpM2 = MPS_.M[nn].shape, MPS_.M[nn+1].shape
               Theta = ncon([MPS_.M[nn],MPS_.M[nn+1],Uprop],[[-1,2,1],[1,3,-4],[2,3,-2,-3]]) 
               Theta = Theta.reshape(shpM1[0]*shpM1[1], shpM2[1]*shpM2[2])
               U,S,V = MPS.svdtruncate(Theta, etrunc, chiMAX,info=info)
               MPS_.M[nn]  = U.reshape(shpM1[0],shpM1[1],S.size)
               MPS_.M[nn+1] = (np.diag(S)@V).reshape(S.size,shpM2[1],shpM2[2]) 
               for index in reversed(range(x-1,nn,-1)):
                   MPS_.swap(index,chiMAX=chiMAXswap,center='i',info=info)           
        MPS_.M[x] = ncon([MPS_.M[x],flip_op],[[-1,1,-3],[1,-2]])

Lx = 14
Ly = Lx

L  = Lx*Ly
psi_state = MPS.getAllUp(L)

mag = []
err = 0.
mag.append(MPO.return_LocalMz(psi_state).real.reshape(Lx,Ly))
for k in range(20):
    print('k',k,np.max(mag[k]-mag[k].T))
    for x in psi_state.M:
        print(x.shape)
    TEBD_evo(psi_state, Lx, Ly,chiMAX=2048,chiMAXswap=2048,etrunc=1e-4,info=info)
    mag.append(MPO.return_LocalMz(psi_state).real.reshape(Lx,Ly))