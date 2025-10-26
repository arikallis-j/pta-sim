import numpy as np

from .consts import PI

def zeros(shape):
    return np.zeros(shape)

def sqrt(x):
    return np.sqrt(x)

def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def arccos(x):
    return np.arccos(x)

def exp(x):
    return np.exp(x)

def ln(x):
    return np.log(x)

def t_sum(indexing, a, b):
    return np.einsum(indexing,a,b)

def mu_0(gamma):
    cos_gamma = cos(gamma)
    mu = 1/3 - 1/6 * (1 - cos_gamma)/2 + (1 - cos_gamma)/2 * ln((1 - cos_gamma)/2)
    return mu

def mu_u(gamma, N = 100):
    N_ph, N_th = N, N

    p1 = np.array([0, 0, 1])
    p2 = np.array([sin(gamma), 0, cos(gamma)])
    alpha_1, alpha_2 = 100, 100

    cos_gamma = t_sum('i,i->',p1,p2)/(t_sum('i,i->',p1,p1)*t_sum('i,i->',p2,p2))
    gamma = arccos(cos_gamma)

    dphi, dtheta = 2*PI/(N_ph -1), PI/(N_th -1)
    phi = np.linspace(dphi/2, 2*PI - dphi/2, N_ph)
    theta = np.linspace(dtheta/2, PI - dtheta/2, N_th)

    phi, theta = np.meshgrid(phi, theta, indexing='xy')
    dphi, dtheta = np.full(phi.shape, dphi), np.full(theta.shape, dtheta)


    Omega = np.array([-sin(theta)*cos(phi), -sin(theta)*sin(phi), -cos(theta)])
    dOmega = dphi * np.sin(theta) * dtheta

    m = np.array([sin(phi), - cos(phi), np.zeros(phi.shape)])
    n = np.array([-cos(theta)*cos(phi), -cos(theta)*sin(phi), sin(theta)])

    e_p = t_sum('ijk,ljk->iljk', m, m) - t_sum('ijk,ljk->iljk', n, n) 
    e_c = t_sum('ijk,ljk->iljk', m, n) + t_sum('ijk,ljk->iljk', n, m) 
    e = e_p + 1j * e_c
    F_1 = 1/2 * t_sum('il,iljk->jk',t_sum('i,l->il', p1, p1), e) / (1 + t_sum('ijk,i->jk', Omega, p1)) 
    F_2 = 1/2 * t_sum('il,iljk->jk',t_sum('i,l->il', p2, p2), e) / (1 + t_sum('ijk,i->jk', Omega, p2)) 

    T_1 =  1 - np.exp(-1j  * 2 * PI * alpha_1 * (1 + t_sum('ijk,i->jk', Omega, p1)))
    T_2 =  1 - np.exp(-1j  * 2 * PI * alpha_2 * (1 + t_sum('ijk,i->jk', Omega, p2)))

    R_1 = F_1 * T_1
    R_2 = F_2 * T_2

    mu = 1/(4*PI) * np.real(np.sum(R_1 * np.conjugate(R_2) * dOmega))
    
    return mu

