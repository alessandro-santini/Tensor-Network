import MPO_class as MPO
import numpy as np
from ncon import ncon
import MPS_class as MPS
import tdvp2 

def TightBinding(L, t):
    Ham = MPO.MPO(L,2)
    # Operator in the local basis
    cdag = np.array([[0,0],[1,0]])
    c    = np.array([[0,1],[0,0]])
    n    = np.array([[0,0],[0,1]])
    Id   = np.array([[1,0],[0,1]])
    
    # Building the local bulk MPO
    H = np.zeros([4,4,2,2])
    H[0,0,:,:] = Id; H[3,3,:,:] = Id; 
    H[1,0,:,:] = cdag; H[2,0,:,:] = c;
    H[3,1,:,:] = t*c; H[3,2,:,:] = -t*cdag

    # Building the boundary MPOs
    HL = np.zeros((1,4,2,2))
    HL[0,:,:,:] = H[3,:,:,:]
    HR = np.zeros((4,1,2,2))
    HR[:,0,:,:] = H[:,0,:,:]

    Ham.W[0] = HL
    Ham.W[L-1] = HR
    for i in range(1,L-1):
        Ham.W[i] = H
    return Ham

L = 32
t = 1
H = TightBinding(L, t)
psi = MPS.getAllUp(L)

empty = np.zeros((1,2,1)); empty[0,0,0] = 1
full  = np.zeros((1,2,1)); full[0,1,0]  = 1

for i in range(L//2):
    psi.M[i] = empty
    psi.M[i+L//2] = full
E0 = H.contractMPOMPS(psi)
engine = tdvp2.TDVP2(psi,H)
engine.initialize()
mz  = []
Sent =[]

mz.append(MPO.return_LocalMz(engine.MPS).real)
Sent.append(engine.MPS.compute_EntEntropy().real)
for i in range(1,100):
    engine.time_step(0.1, 1e-9)
    if i % 5 == 0:
        
    mz.append(MPO.return_LocalMz(engine.MPS).real)
    Sent.append(engine.MPS.compute_EntEntropy().real)
mz = np.array(mz)
nloc = 0.5*(1-mz)
Sent = np.array(Sent)
#%%