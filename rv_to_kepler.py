import numpy as np

mu = 3.986004418e14

def rv_to_kepler(r,v):
    
    L = np.cross(r, v)
    #Нормировали вектор
    n_orb = L / np.linalg.norm(L)
    
    i = np.arccos(n_orb[2])
    
    
    z = np.array([0,0,1])
    N = np.cross(z, n_orb) if i!=0 else np.array([1, 0, 0])
    
    Omega = np.arctan2(N[1],N[0]) if i!=0 else 0
    
    
    r_norm = np.linalg.norm(r)
    v2 = np.dot(v,v)
    
    e_vek = ((v2 - mu/r_norm)*r - np.dot(r,v)*v)/mu
    e = np.linalg.norm(e_vek)
    e1 = e_vek/e
    
    d1 = N / np.linalg.norm(N) #Нормированный N
    d2 = np.cross(n_orb, d1) #Перпенд n_orb и N(d1)
    
    cos_w = np.dot(e1, d1)
    sin_w = np.dot(e1, d2)
    
    w = np.arctan2(sin_w, cos_w)
    
    
    e2 = np.cross(n_orb, e1)
    
    cos_nu = np.dot(r, e1) / r_norm
    sin_nu = np.dot(r, e2) / r_norm
    
    nu = np.arctan2(sin_nu,cos_nu)
    
    
    a = -mu / 2 / (v2 / 2 - mu / r_norm)
    
    
    return(np.array([i, Omega, e, w, nu, a]))
    
    
# #Test
# r = np.array([
#     6524834, 6862875, 6448296
# ])

# v = np.array([
#     4901.327, 5533.756, -1976.341
# ])

# print(rv_to_kepler(r,v))