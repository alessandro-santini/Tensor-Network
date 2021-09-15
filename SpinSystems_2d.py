import MPO_class as MPO
import numpy as np

def IsingMPO_2D(Lx, Ly, h=0., J=1):
    
    L = Lx*Ly
    Ham = MPO.MPO(L, 2)
        
    # Pauli Matrices
    Sx = np.array([[0, 1], [1, 0]])
    Sz = np.array([[1, 0], [0,-1]])
    
    for i in range(1, L-1):
        H = np.zeros([2+Ly, 2+Ly, 2, 2])
        H[0, 0, :, :] = np.eye(2)
        H[1, 0, :, :] = Sz
        for j in range(1, Ly):
                H[1+j,j,:,:] = np.eye(2)
        if i<(Ly*(Lx-1)):        
            H[-1,-2,:,:]  = -J*Sz
        else:
            H[-1,-2,:,:]  = np.eye(2)
        if (i+1)%(Ly) != 0:
            H[-1,1,:,:]   = -J*Sz
        H[-1,0,:,:]  = -h*Sx
        H[-1,-1,:,:] = np.eye(2)
        Ham.W[i] = H
    
    HL = np.zeros([1,2+Ly,2,2])
    HR = np.zeros([2+Ly,1,2,2])
    
    HL[0,0, :,:] = -h*Sx
    HL[0,1, :,:] = -J*Sz
    HL[0,-2,:,:] = -J*Sz
    HL[0,-1,:,:] = np.eye(2)
    
    HR[0, 0,:,:] = np.eye(2)
    HR[1, 0,:,:] = Sz
    HR[-1,0,:,:] = -h*Sx
     
    Ham.W[0]  = HL
    Ham.W[-1] = HR
    return Ham
