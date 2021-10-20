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

def IsingMPO_2D_gen_field(Lx, Ly, h=(0.,0.,0.), J=1):

    L = Lx*Ly
    Ham = MPO.MPO(L, 2)

    # Pauli Matrices
    Sx = np.array([[0, 1], [1, 0]])
    Sy = np.array([[0,-1j],[1j,0]])
    Sz = np.array([[1, 0], [0,-1]])

    hS = -h[0]*Sx-h[1]*Sy-h[2]*Sz
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

        H[-1,0,:,:]  = hS
        H[-1,-1,:,:] = np.eye(2)
        Ham.W[i] = H

    HL = np.zeros([1,2+Ly,2,2])
    HR = np.zeros([2+Ly,1,2,2])

    HL[0,0, :,:] = hS
    HL[0,1, :,:] = -J*Sz
    HL[0,-2,:,:] = -J*Sz
    HL[0,-1,:,:] = np.eye(2)

    HR[0, 0,:,:] = np.eye(2)
    HR[1, 0,:,:] = Sz
    HR[-1,0,:,:] = hS

    Ham.W[0]  = HL
    Ham.W[-1] = HR
    return Ham


def IsingMPO_2D_Spiral_MPS(L, h=0., J=1):
    def spiral(rows, columns):
        matrix = np.zeros((rows,columns))
        row, column, value = 0, 0, rows*columns-1
        
        while row < rows and column < columns:
            for i in range(column, columns):
                matrix[row,i] = value
                value -= 1
            row += 1
            for i in range(row, rows):
                matrix[i,columns - 1] = value
                value -= 1
            columns -= 1
            if row < rows:
                for i in range(columns - 1, column - 1, -1):
                    matrix[rows - 1,i] = value
                    value -= 1
                rows -= 1
            if column < columns:
                for i in range(rows - 1, row - 1, -1):
                    matrix[i,column] = value
                    value -= 1
                column += 1
        return matrix    
    def compute_dim_MPO(L):
        config = spiral(L,L)
        dim = []
        nearest = []
        for x in range(L*L):
            i,j = np.where(config == x)
            i = i[0]; j = j[0]
            nn = []
            if i !=0: nn.append(config[i-1,j])
            if j !=0: nn.append(config[i,j-1])
            if i !=L-1: nn.append(config[i+1,j])
            if j !=L-1: nn.append(config[i,j+1])
            nn = np.array(nn)
            nearest.append( np.array(nn[nn>config[i,j]]-config[i,j],int))
            new_dim = int(np.max(nn-config[i,j]) + 2)
            if x > 0:
                if new_dim < dim[x-1][1]:
                   dim.append((dim[x-1][1],dim[x-1][1]-1,2,2))
                else:
                    dim.append((dim[x-1][1],new_dim,2,2))    
            else:
                dim.append((1,new_dim,2,2))
        dim[-1] = (dim[-2][1],1,2,2)
        return dim,nearest
    
    Ham = MPO.MPO(L*L, 2)
    Ham.config = np.array(spiral(L,L),int)
    # Pauli Matrices
    Sx = np.array([[0, 1], [1, 0]])
    Sz = np.array([[1, 0], [0,-1]])

    dimW,nn = compute_dim_MPO(L)

    for i,dim in enumerate(dimW):
        Ham.W[i] = np.zeros(dim)
    
    # Left Bond
    Ham.W[0][0,0,:,:] = - h*Sx
    for k in nn[0]:
        Ham.W[0][0,k,:,:] = -J*Sz
    Ham.W[0][0,-1,:,:] = np.eye(2)
    # Right Bond
    Ham.W[-1][0,0,:,:] = np.eye(2)
    Ham.W[-1][1,0,:,:] = Sz
    Ham.W[-1][2,0,:,:] = -h*Sx    
    # Bulk Bonds
    for i in range(1,L**2-1):
        Ham.W[i][0,0,:,:]  = np.eye(2)
        Ham.W[i][-1,0,:,:] = -h*Sx
        Ham.W[i][1,0,:,:] = Sz
        for k in nn[i]:
            Ham.W[i][-1,k,:,:] = -J*Sz
        for k in range(2,Ham.W[i].shape[0]-1):
            Ham.W[i][k,k-1,:,:] = np.eye(2)
        Ham.W[i][-1,-1,:,:] = np.eye(2)
    return Ham

def IsingMPO_2D_diagonal_MPS(L, h=0., J=1):
    def diagonal_matrix(L):
        sites = np.arange(L*L)
        k = 1
        pos = 0
        dist = L-1
        mat = np.zeros((L,L))
        while(dist>-L):
            mat += np.diag(sites[pos:(pos+k)],dist)
            pos += k
            if dist > 0:
                k+=1
            else:
                k-= 1
            dist -= 1
        return np.fliplr(mat)
    
    def compute_dim_MPO(L):
        config = diagonal_matrix(L)
        dim = []
        nearest = []
        for x in range(L*L):
            i,j = np.where(config == x)
            i = i[0]; j = j[0]
            nn = []
            if i !=0: nn.append(config[i-1,j])
            if j !=0: nn.append(config[i,j-1])
            if i !=L-1: nn.append(config[i+1,j])
            if j !=L-1: nn.append(config[i,j+1])
            nn = np.array(nn)
            nearest.append( np.array(nn[nn>config[i,j]]-config[i,j],int))
            new_dim = int(np.max(nn-config[i,j]) + 2)
            if x > 0:
                if new_dim < dim[x-1][1]:
                   dim.append((dim[x-1][1],dim[x-1][1]-1,2,2))
                else:
                    dim.append((dim[x-1][1],new_dim,2,2))    
            else:
                dim.append((1,new_dim,2,2))
        dim[-1] = (dim[-2][1],1,2,2)
        return dim,nearest
    
    Ham = MPO.MPO(L*L, 2)
    Ham.config = np.array(diagonal_matrix(L),int)
    # Pauli Matrices
    Sx = np.array([[0, 1], [1, 0]])
    Sz = np.array([[1, 0], [0,-1]])

    dimW,nn = compute_dim_MPO(L)

    for i,dim in enumerate(dimW):
        Ham.W[i] = np.zeros(dim)
    
    # Left Bond
    Ham.W[0][0,0,:,:] = - h*Sx
    for k in nn[0]:
        Ham.W[0][0,k,:,:] = -J*Sz
    Ham.W[0][0,-1,:,:] = np.eye(2)
    # Right Bond
    Ham.W[-1][0,0,:,:] = np.eye(2)
    Ham.W[-1][1,0,:,:] = Sz
    Ham.W[-1][2,0,:,:] = -h*Sx    
    # Bulk Bonds
    for i in range(1,L**2-1):
        Ham.W[i][0,0,:,:]  = np.eye(2)
        Ham.W[i][-1,0,:,:] = -h*Sx
        Ham.W[i][1,0,:,:] = Sz
        for k in nn[i]:
            Ham.W[i][-1,k,:,:] = -J*Sz
        for k in range(2,Ham.W[i].shape[0]-1):
            Ham.W[i][k,k-1,:,:] = np.eye(2)
        Ham.W[i][-1,-1,:,:] = np.eye(2)
    return Ham