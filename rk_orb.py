import numpy as np
import matplotlib.pyplot as plt
from rv_to_kepler import rv_to_kepler

def step_rk(f, S, t, d):
    k1 = f(S, t)
    k2 = f(S + k1*d/2, t + d/2)
    k3 = f(S + k2*d/2, t + d/2)
    k4 = f(S + k3*d, t + d)
    
    return S + (k1 + 2*k2 + 2*k3 + k4)*d/6

def integrate_rk(f, S0, t0, t_f, d):
    n = int((t_f-t0)//d + 1)

    res_S = np.zeros((n, np.size(S0)))
    res_t = np.zeros(n)
    
    res_S[0] = S0
    res_t[0] = t0
    
    for i in range(1, n-1):
        res_S[i] = step_rk(f, res_S[i-1], res_t[i-1], d)
        res_t[i] = res_t[i-1] + d
        
    res_S[n-1] = step_rk(f, res_S[n-2], res_t[n-2], t_f-res_t[n-2])
    res_t[n-1] = t_f
    
    return res_S, res_t

def grav(s, t):
    r = s[:3]
    r_norm = np.linalg.norm(r)
    
        
    MU = 3.986004418e14
    R_earth = 6378137
    
    acc = - MU / r_norm**3 * r
    
    #Возмущения от нецентр. Зеимли
    J2 = 0.0010826269
    j2_const = - 3 * J2 * MU * R_earth ** 2 / (2 * r_norm ** 5)
    j2_grav = j2_const * r * (1 - 5 * (r[2] / r_norm) ** 2)
    j2_grav[2] += 2 * j2_const * r[2]

    
    res = np.array([s[3], s[4], s[5], acc[0] + j2_grav[0], acc[1] + j2_grav[1], acc[2] + j2_grav[2]])
    
    return res

s_0 = np.array([-11957371.5217699557542801,
                       385936.2176575958728790, - 1058529.5938599361106753, - 548.6566157170911993,
                       - 3515.9242334314676555,
                       4576.9840499053161693
                       ])

t_0 = 0
t_1 = 108000
d = 10

orbit, t_orb = integrate_rk(grav, s_0, t_0, t_1, d)

#Rv to kepler

n = len(orbit)
r = np.zeros((n, 3))
v = np.zeros((n, 3))

for i in range(n):
    for j in range(3):
        r[i][j] = orbit[i][j]
        v[i][j] = orbit[i][j+3]
        
kepl = np.zeros((n, 6))

for i in range(n):
    kepl[i] = rv_to_kepler(r[i], v[i])
    
fig, ax = plt.subplots()

#ax.plot(t_orb, kepl[:, 0], label='i')
#ax.plot(t_orb, kepl[:, 1], label='omega')
#ax.plot(t_orb, kepl[:, 2], label='e')
ax.plot(t_orb, kepl[:, 3], label='w')
#ax.plot(t_orb, kepl[:, 4], label='nu')
#ax.plot(t_orb, kepl[:, 5], label='a')

plt.legend()
plt.show()