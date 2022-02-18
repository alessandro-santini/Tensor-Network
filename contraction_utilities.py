from ncon import ncon

## tensor contraction from the right hand side
##  0-+     -A--+
##    |      |  |
##  1-R' =  -W--R
##    |      |  |
##  2-+     -B--+    
def contract_right(A,W,B,R):
    return ncon([A,W,B,R],[[-1,3,1],[-2,2,3,4],[-3,4,5],[1,2,5]])

## tensor contraction from the left hand side
## +--   +--A-0
## |     |  |
## L'- = L--W-1
## |     |  |
## +--   +--B-2  
def contract_left(A,W,B,L):
    return ncon([A,W,B,L],[[1,3,-1],[2,-2,3,4],[5,4,-3],[1,2,5]])   

def mix_contract_right(A,W,B,R):
    return ncon([A,W,B,R],[[-1,3,7,1],[-2,2,3,4],[-3,4,7,5],[1,2,5]])
def mix_contract_left(A,W,B,L):
    return ncon([A,W,B,L],[[1,3,7,-1],[2,-2,3,4],[5,4,7,-3],[1,2,5]])   