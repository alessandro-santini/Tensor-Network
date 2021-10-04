import numpy as np

sx = np.array([[0,1],[1,0]])
id_ = np.eye(2)

def K(x,y,z):
    return np.kron(np.kron(x,y),z)

g  =  0.3
c  = np.cos
s  = np.sin

H  =  K(id_,id_,id_) * c(g)**3 + K(sx,sx,sx) * s(g)**3+\
     (K(sx,id_,id_) + K(id_,sx,id_) +K(id_,id_,sx)) * c(g)**2*s(g) +\
     (K(sx,id_,sx) + K(sx,sx,id_) +K(id_,sx,sx)) * c(g)*s(g)**2
