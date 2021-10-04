import MPO_class as MPO
import numpy as np

def IsingMPO(L, h, J=1):
    Ham = MPO.MPO(L,2)
        
    # Pauli Matrices
    Sx = np.array([[0, 1], [1, 0]])
    Sz = np.array([[1, 0], [0,-1]])
    Id = np.array([[1, 0], [0, 1]])
    
    # Building the local bulk MPO
    H = np.zeros([3,3,2,2])
    H[0,0,:,:] = Id; H[2,2,:,:] = Id; H[2,0,:,:] = -h*Sx
    H[1,0,:,:] = Sz; H[2,1,:,:]= -J*Sz

    # Building the boundary MPOs
    HL = np.zeros((1,3,2,2))
    HL[0,:,:,:] = H[2,:,:,:]
    HR = np.zeros((3,1,2,2))
    HR[:,0,:,:] = H[:,0,:,:]

    Ham.W[0] = HL
    Ham.W[L-1] = HR
    for i in range(1,L-1):
        Ham.W[i] = H
    return Ham

def CustomIsingMPO(L, hSh, J=1):
    Ham = MPO.MPO(L,2)
        
    # Pauli Matrices
    Sz = np.array([[1, 0], [0,-1]])
    Id = np.array([[1, 0], [0, 1]])
    
    # Building the local bulk MPO
    H = np.zeros([3,3,2,2])
    H[0,0,:,:] = Id; H[2,2,:,:] = Id; H[2,0,:,:] = hSh
    H[1,0,:,:] = Sz; H[2,1,:,:]= -J*Sz

    # Building the boundary MPOs
    HL = np.zeros((1,3,2,2))
    HL[0,:,:,:] = H[2,:,:,:]
    HR = np.zeros((3,1,2,2))
    HR[:,0,:,:] = H[:,0,:,:]

    Ham.W[0] = HL
    Ham.W[L-1] = HR
    for i in range(1,L-1):
        Ham.W[i] = H
    return Ham
    

def XXZMPO(L,delta, h, J=1):
    Ham = MPO.MPO(L,2)
    # Pauli Matrices
    Sp = np.array([[0,1],[0,0]])
    Sm = np.array([[0,0],[1,0]])
    Sz = 0.5*np.array([[1, 0], [0,-1]])
    Id = np.array([[1, 0], [0, 1]])
    
    # Building the local bulk MPO
    H = np.zeros([5,5,2,2])
    H[0,0,:,:] =       Id
    H[1,0,:,:] =       Sp
    H[2,0,:,:] =       Sm
    H[3,0,:,:] =       Sz
    H[4,0,:,:] =    -h*Sz
    H[4,1,:,:] =    .5*J*Sm
    H[4,2,:,:] =    .5*J*Sp
    H[4,3,:,:] = delta*Sz
    H[4,4,:,:] =       Id
    
    # Building the boundary MPOs
    HL = np.zeros((1,5,2,2))
    HL[0,:,:,:] = H[4,:,:,:]
    HR = np.zeros((5,1,2,2))
    HR[:,0,:,:] = H[:,0,:,:]
    
    Ham.W[0] = HL
    Ham.W[L-1] = HR
    for i in range(1,L-1):
        Ham.W[i] = H
    return Ham

def HDM(N,sigma,h=1):
    
    L = 2**N
    sigma = 1

    real_states = [np.vectorize(np.binary_repr)(np.arange(2**N),N)][0]
    t = np.array([2.**( - (1 + sigma)*r ) for r in np.arange(-1,N+1) ])
    A = np.zeros((L,L))

    for i, state_a in enumerate(real_states):
        for j, state_b in enumerate(real_states):
            k = N
            while(state_a[:k] != state_b[:k] or k < 0 ):
                k = k-1
            else:
                A[i,j] = t[N-k-1]
    # H = -J\sum_{ij} A_{ij}\sigma^z_i\sigma^z_j - h \sum_i sigma^x_i
    Ham = MPO.MPO(L, 2)
    # Pauli Matrices
    Sx = np.array([[0,1],[1,0]])
    Sz = np.array([[1, 0], [0,-1]])
    Id = np.array([[1, 0], [0, 1]])
    
    # Building the border MPO
    HL = np.zeros([1,L+1,2,2])
    HL[0,0,:,:]  = - h*Sx
    HL[0,-1,:,:] = Id 
    for i in range(1,L):
        HL[0,i,:,:] = -A[0,i]*Sz
    HR = np.zeros([L+1,1,2,2])
    HR[0,0,:,:] = Id
    HR[1,0,:,:] = Sz
    HR[-1,0,:,:] = -h*Sx
    
    Ham.W[0]  = HL
    Ham.W[-1] = HR
    
    # Building the local bulk MPO
    for k in range(1,L-1):
        H = np.zeros([L+1,L+1,2,2])
        H[0,0,:,:] = Id
        H[1,0,:,:] = Sz
        H[-1,0,:,:] = -h*Sx
        for i in range(1,L):
            if i != (L-1):
                H[i+1,i] = Id
            if k+i<L:
                H[-1,i,:,:] = -A[k,k+i]*Sz
        H[-1,-1,:,:] = Id
        Ham.W[k] = H.copy()
    return Ham,A    