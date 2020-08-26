import numpy as np

def biprojection(A,gA,B,gB) :
# A et B sont deja les bases reduites donc carrees et orthogonales, gA et gB les coordonn�es r�duites
    qA =  np.dot(A,gA)
    qB = np.dot(B,gB)
    EA = 2*np.linalg.norm( qA - np.dot(B, np.dot(B.T, qA) ) )/(np.linalg.norm( qA )+np.linalg.norm( qB ))
    EB = 2*np.linalg.norm( qB - np.dot(A, np.dot(A.T, qB) ) )/(np.linalg.norm( qA )+np.linalg.norm( qB ))
    return min(EA,EB)

def grassmann(A,B) :
# A et B sont deja les bases reduites donc carrees et orthogonales
    Q=np.dot(A.T,B)
    svd=np.linalg.svd(Q, compute_uv=False)
    sigma=[min(1,i) for i in svd]
    return np.sqrt(np.sum(np.power(np.arccos(sigma),2)))


def schubert(A,B) :
# A et B sont deja les bases reduites donc orthogonales
    r=max(A.shape[1],B.shape[1])
    sigma=[0 for i in range(r)]
    Q=np.dot(A.T,B)
    svd=np.linalg.svd(Q, compute_uv=False)

    for i in range(len(svd)):
        sigma[i]=min((svd[i],1))
    return np.sqrt(np.sum(np.power(np.arccos(sigma),2)))


def euclide(A,B):
    if A.shape[1]>B.shape[1]:
        A=A[:,:B.shape[1]]
        return np.linalg.norm(B-A)
    if B.shape[1]>A.shape[1]:
         B=B[:,:A.shape[1]]
         return np.linalg.norm(B-A)

    return np.linalg.norm(B-A)




def asimov(A,B):
    Q=np.dot(A.T,B)
    svd=np.linalg.svd(Q, compute_uv=False)
    sigma=min((svd[0],1))
    return np.arccos(sigma)

def binet_cauchy(A,B):
    Q=np.dot(A.T,B)
    svd=np.linalg.svd(Q, compute_uv=False)
    return np.sqrt(np.abs(1-np.prod(np.power(svd,2))))

def chordal(A,B):
    Q=np.dot(A.T,B)
    svd=np.linalg.svd(Q, compute_uv=False)
    sigma=[min(1,i) for i in svd]
    return np.sqrt(np.abs(np.sum(np.power(np.sin(np.arccos(sigma)),2))))


def fubini_study(A,B):
    Q=np.dot(A.T,B)
    svd=np.linalg.svd(Q, compute_uv=False)
    sigma=[min(1,i) for i in svd]
    return np.arccos(np.prod(sigma))

def martin(A,B):
    Q=np.dot(A.T,B)
    svd=np.linalg.svd(Q, compute_uv=False)
    sigma=[min(1,i) for i in svd]
    return np.sqrt(np.abs(np.log10(np.abs(np.prod(1/np.power(sigma,2))))))

def procrustes(A,B):
    Q=np.dot(A.T,B)
    svd=np.linalg.svd(Q, compute_uv=False)
    sigma=[min(1,i) for i in svd]
    return 2*np.sqrt(np.abs(np.sum(np.power(np.sin(np.arccos(sigma)/2),2))))

def projection(A,B):
    return np.sin(asimov(A,B))

def spectral(A,B):
    return 2*np.sin(asimov(A,B)/2)





def MatDistance(L,distance, coord = None):
    Distance=np.zeros((len(L),len(L)))
    if coord is None:
        for i in range(len(L)):
            for j in range(i):
                d=distance(L[i],L[j])
                Distance[i][j] = d
                Distance[j][i] = d
    else:
        for i in range(len(L)):
            for j in range(i):
                d=biprojection(L[i],coord[i],L[j],coord[j])
                Distance[i][j] = d
                Distance[j][i] = d
    return Distance