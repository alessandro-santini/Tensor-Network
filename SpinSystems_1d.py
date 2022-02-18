import MPO_class as MPO
import numpy as np
from ncon import ncon

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

def IsingMPO_XX_coupling(L, hz, J = 1, hx = 0):
    Ham = MPO.MPO(L,2)
        
    # Pauli Matrices
    Sx = np.array([[0, 1], [1, 0]])
    Sz = np.array([[1, 0], [0,-1]])
    Id = np.array([[1, 0], [0, 1]])
    
    # Building the local bulk MPO
    H = np.zeros([3,3,2,2])
    H[0,0,:,:] = Id; H[2,2,:,:] = Id; H[2,0,:,:] = -hz*Sz-hx*Sx
    H[1,0,:,:] = Sx; H[2,1,:,:]= -J*Sx

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

def HDM(N,sigma,J=1,h=1,gs_notdeg=True):
    L = 2**N

    real_states = [np.vectorize(np.binary_repr)(np.arange(2**N),N)][0]
    #t = np.array([2.**( - (1 + sigma)*r ) for r in np.arange(-1,N+1) ])
    t = np.array([2.**( - (1 + sigma)*k ) for k in np.arange(0, N) ])
    A = np.zeros((L,L))
    
    for i, state_a in enumerate(real_states):
        for j, state_b in enumerate(real_states):
            if i != j :
                k = N
                while( state_a[:k] != state_b[:k] or k < 0 ):
                    k = k-1
                else:
                    # A[i,j] = t[N-k-1]
                    # A[i,j] = np.sum(t[(N-k-1):])
                    A[i,j] = t[(N-k-1)]
    A = J*A
    # H = -J\sum_{ij} A_{ij}\sigma^z_i\sigma^z_j - h \sum_i sigma^x_i
    Ham = MPO.MPO(L, 2)
    # Pauli Matrices
    Sx = np.array([[0,1],[1,0]])
    Sz = np.array([[1, 0], [0,-1]])
    Id = np.array([[1, 0], [0, 1]])
    
    # Building the border MPO
    HL = np.zeros([1,L+1,2,2])
    HL[0,0,:,:]  = -h*Sz
    HL[0,-1,:,:] = Id 
    for i in range(1,L):
        HL[0,i,:,:] = -A[0, i]*Sx

    HR = np.zeros([3,1,2,2])
    HR[0,0,:,:] = Id
    HR[1,0,:,:] = Sx
    HR[2,0,:,:] = -h*Sz
    
    Ham.W[0]  = HL
    Ham.W[-1] = HR
    
    # Building the local bulk MPO
    for k in range(1,L-1):
        H = np.zeros([L+2-k,L+1-k,2,2])
        H[0,0,:,:] = Id
        H[1,0,:,:] = Sx
        H[-1,0,:,:] = -h*Sz
        for i in range(1,L-k):
            #if i != (L-k-1):
            H[i+1,i] = Id
            if k+i<L:
                H[-1,i,:,:] = -A[k,k+i]*Sx
        H[-1,-1,:,:] = Id
        Ham.W[k] = H.copy()
    return Ham,A    


def PBCIsingMPO(L,hx,hz,J=1):
    Ham = MPO.MPO(L,2)
        
    # Pauli Matrices
    Sx = np.array([[0, 1], [1, 0]])
    Sz = np.array([[1, 0], [0,-1]])
    Id = np.array([[1, 0], [0, 1]])
    
    # Building the local bulk MPO
    H = np.zeros([6,6,2,2])
    H[0,0,:,:] = Id; H[2,2,:,:] = Id; H[2,0,:,:] = -hx*Sx-hz*Sz
    H[1,0,:,:] = Sz; H[2,1,:,:]= -J*Sz
    H[0+3,0+3,:,:] = Id; H[2+3,2+3,:,:] = Id; H[2+3,0+3,:,:] = -hx*Sx-hz*Sz
    H[1+3,0+3,:,:] = Sz; H[2+3,1+3,:,:]= -J*Sz
    
    H1 = np.zeros([6,6,2,2])
    H1[0,0,:,:] = -hx*Sx-hz*Sz; H1[0,1,:,:] = -J*Sz; H1[0,2,:,:] = Id
    H1[1,2,:,:] = Sz;
    H1[0+3,0+3,:,:] = -hx*Sx-hz*Sz; H1[0+3,1+3,:,:] = -J*Sz; H1[0+3,2+3,:,:] = Id
    H1[1+3,2+3,:,:] = Sz;
    
    v = np.array([1,0,0,0,1,0]).reshape(6,1)
    # Building the boundary MPOs
    #HL = np.zeros((1,6,2,2))
    HL = ncon([v.T,H1],[[-1,1],[1,-2,-3,-4]])
    #HR = np.zeros((6,1,2,2))
    HR = ncon([H,v],[[-1,1,-3,-4],[1,-2]])

    Ham.W[0] = HL
    Ham.W[L-1] = HR
    for i in range(1,L-1):
        Ham.W[i] = H
    return Ham

def Long_Range_Ising(L,alpha,hz,k=5,J=1):
    from scipy.optimize import curve_fit
    def exp_fit(x, *p):
        res = 0.
        for i in range(len(p)//2):
            res += p[2*i]*p[2*i+1]**x
        return res
    r = np.linspace(1,L,2048); y = r**(-alpha)    
    popt,pcov = curve_fit(exp_fit,r,y,p0=np.ones(k*2),bounds=[2*k*[0.],2*k*[np.inf]],maxfev =1e6)
    print('Err approx pow_law',np.trapz(np.abs(y-exp_fit(r,*popt)),r))
    c = popt[::2]; lam = popt[1::2]

    Ham = MPO.MPO(L,2)
    # Pauli Matrices
    Sx = np.array([[0, 1], [1, 0]])
    Sz = np.array([[1, 0], [0,-1]])
        
    # Kac-Scaling
    Nalpha = 0
    for i in range(1,L+1):
        for j in range(i+1,L+1):
            Nalpha += 1/(j-i)**alpha
    J = J/(L-1)*Nalpha
    print(J)
    H_bulk = np.zeros([2+k,2+k,2,2])
    H_bulk[0, 0, :, :] = np.eye(2)
    for i in range(k):
        H_bulk[1+i, 0, :, :] = Sx
        H_bulk[1+i,1+i,:, :] = lam[i]*np.eye(2)
        H_bulk[-1, 1+i,:, :] = -J*lam[i]*c[i]*Sx
    H_bulk[-1, 0, :, :] = -hz*Sz
    H_bulk[-1,-1,:,:] = np.eye(2)
    for i in range(1,L-1):
        Ham.W[i] = H_bulk
    Ham.W[0]  = H_bulk[-1,:,:,:].reshape(1,2+k,2,2)
    Ham.W[-1] = H_bulk[:,0,:,:].reshape(2+k,1,2,2)
    return Ham
    
    