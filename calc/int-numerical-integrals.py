import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from tqdm import tqdm
PI = np.pi

class Calculator:
    def __init__(self, N=10):
        self.setup(N)

    def setup(self,N):
        self.N = N
        self.phi = np.linspace(0, 2*PI, N, endpoint=False)
        self.dphi = np.ones(N) * 2*PI / N 
    
    def true_int(self, integral, **params):
        return integral(**params)
    def est_int(self, func, **params):
        return np.sum(self.dphi * func(self.phi, **params))
    def check(self, func, integral, **params):
        Int_true = self.true_int(integral, **params)
        Int_est = self.est_int(func, **params)
        return Int_true, Int_est
    def accuracy(self, func, integral, **params):
        Int_true, Int_est = self.check(func, integral, **params)
        return np.abs(Int_true - Int_est)
    def test_acc(self, func, integral, n_iter=10, log_eps=-14, **Params):
        is_accur = True
        Int_true, Int_est = [], []
        eps = 10.0**(log_eps)
        for i in tqdm(range(n_iter)):
            params = {}
            for k, v in Params.items():
                params[k] = v[i]
            int_true, int_est = self.check(func, integral, **params)
            Int_true.append(int_true)
            Int_est.append(int_est)
            delta = self.accuracy(func, integral, **params)
            sigma = np.abs(delta/(int_est + 10.0**(log_eps-3)))
            is_accur = is_accur and delta<eps
            # print(self.true_int(integral, **params), self.est_int(func, **params))
            if delta>=eps:
                print(f"log-delta = {np.log10(delta):.2f} > {log_eps} | {int_true:.2e} {int_est:.2e}")
        if is_accur:
            print("Test passed")
        else:
            print("Test not passed")

        return np.array(Int_true), np.array(Int_est)

def V(a, k):
    r = (-1 + np.sqrt(1 - a**2))/a
    return 2*PI/(np.sqrt(1 - a**2)) * r**np.abs(k)

def W(a, n, m):
    w = 0
    if n%2 == 1:
        return w
    
    for i in range(n+1):
        for j in range(m+1):
            w+= (-1)**(n//2) / 2**(n+m) * (-1)**i * comb(n, i) * comb(m, j) * V(a, (n+m) - 2*(i+j))
    return w

def g(x, a, c, gamma):
    return  1/((1 + a*np.cos(x))*(1 + c*np.cos(x - gamma)))

def f(x, a, c, gamma):
    A = np.sin(x) * np.sin(x - gamma)
    B = ((1 + a*np.cos(x))*(1 + c*np.cos(x - gamma)))
    return A/B

def f_red(x, a, c, gamma):
    La = (a*np.cos(x) + 1)
    Lc = (c*np.cos(x - gamma) + 1)
    Norm = (A**2 * C**2 * np.sin(Gamma)**2)

    P2 = ((a**2 + c**2) * np.cos(gamma) - a*c*(1+np.cos(gamma)**2))
    P1a = (a*c*(1+np.cos(gamma)**2) - a**2*np.cos(gamma)) + \
        a*c*np.cos(gamma) * a * (np.cos(x-gamma)) 
    P1c = (a*c*(1+np.cos(gamma)**2) - C**2*np.cos(gamma)) + \
        a*c*np.cos(gamma) * c * (np.cos(x))
    P0 = - a*c*(1+np.cos(gamma)**2)

    return 1/Norm * (P2/(La*Lc) + P1a/La + P1c/Lc + P0)

def h1(x, a, b, c, d, gamma):
    A = b*d*np.sin(x) * np.sin(x - gamma)
    B = ((a*np.cos(x) + (b+1))*(c*np.cos(x - gamma) + (d+1)))
    return A/B


def z0(x, a, c, gamma):
    return  1/((1 + a*np.cos(x))*(1 + c*np.cos(x - gamma)))

def Z0(a, c, gamma):
    b = np.sqrt(1 - a**2)
    d = np.sqrt(1 - c**2)
    s = np.sqrt(1 - a * c * np.cos(gamma))
    return (2*PI) * (b+d)/(b*d*(s**2 + b*d))

# def Z1(a, c, gamma):
#     NL = ((a**2 + c**2) * np.cos(gamma) - a*c*(1+np.cos(gamma)**2)) * Z0(a, c, gamma)
#     Ca = (a*c*(1+np.cos(gamma)**2) - a**2*np.cos(gamma)) * W(a, 0, 0)
#     Cb = (a*c*(1+np.cos(gamma)**2) - c**2*np.cos(gamma)) * W(c, 0, 0)
#     La = a*c*np.cos(gamma)**2 * (a * W(a, 0, 1))
#     Lc = a*c*np.cos(gamma)**2 * (c * W(c, 0, 1))
#     C = 2*PI * (-a*c*(1+np.cos(gamma)**2))
#     return (NL + Ca + Cb + La + Lc + C)/(a**2 * c**2 * np.sin(gamma)**2)

def Z1(a, c, gamma):
    cos_g, sin_g = np.cos(gamma), np.sin(gamma)
    b, d = np.sqrt(1 - a**2), np.sqrt(1 - c**2)
    return 2*PI/(b*d - a*c*cos_g + 1) * (cos_g - (b-1)/a * (d-1)/c)

# def func_h1(x, a, b, c, d, gamma):
#     A = np.sin(x) * np.sin(x - gamma)
#     B = ((a*np.cos(x) + (b+1))*(c*np.cos(x - gamma) + (d+1)))
#     return A/B

# def H1(a, b, c, d, gamma):
#     s = np.sqrt((b+1)**2 - a**2)
#     q = np.sqrt((d+1)**2 - c**2) 
#     return 2*PI/(a*c) * (
#         (q*(b+1) + s*(d+1))/(s*q - a*c*np.cos(gamma) + (b+1)*(d+1)) - 1
#     )

def func_j(x, a, b, c, d, gamma):
    return np.sin(x)**2 * (c*np.cos(x - gamma) + (d - 1))/ (a*np.cos(x) + (b+1))

def J(a, b, c, d, gamma):
    x0 = ((b+1) - np.sqrt((b+1)**2 - a**2))/a**2

    return 2*PI * (x0 * (d - 1) - x0**2 * a*c*np.cos(gamma)/2)

"""
J0 = - a^3*c^3 * f*g * (2**p^5 + 2*p^3*q^2 + p*q^4)
L0 = + a^3*c^3 * (-4*f*g*p^3*q^2 - 2*f*g*p*q^4 + h*k*p*q^4)
P0 = a^2*c^3*k*p^3*q^2*s + a^2*c^3*k*p*q^4*s - 2*a^3*c^3*p^2*q^2*s*t - a^3*c^3*q^4*s*t - 4*a*c^3*f*g*p^5 - 6*a*c^3*f*g*p^3*q^2 + a*c^3*h*k*p^3*q^2 - 2*a*c^3*f*g*p*q^4 + a*c^3*h*k*p*q^4 - 2*a^3*c^2*k*p^2*q^2*s - a^3*c^2*k*q^4*s - 2*a^2*c^3*h*p^2*q^2*t - a^2*c^3*h*q^4*t + 6*a^2*c^2*f*g*p^4 + 6*a^2*c^2*f*g*p^2*q^2 - 2*a^2*c^2*h*k*p^2*q^2 + a^2*c^2*f*g*q^4 - a^2*c^2*h*k*q^4 + a^3*c^2*h*p*q^2*t - 4*a^3*c*f*g*p^3 - 2*a^3*c*f*g*p*q^2 + a^3*c*h*k*p*q^2

K3u = a^4*c^3*f*g*(p^5 - 3*p^3*q^2)
K2u = a^2*c^2 * (-a^2*c^1*k*p^3*q^2*s + a^2*c^1*k*p*q^4*s + 4*a^1*c^1*f*g*p^5 - 2*a^1*c^1*f*g*p^3*q^2 - a^1*c^1*h*k*p^3*q^2 - 2*a^1*c^1*f*g*p*q^4 + a^1*c^1*h*k*p*q^4 - a^2*f*g*p^4 + a^2*f*g*p^2*q^2)
K1u = 3*a^4*c^3*f*g*p^3*q^2 - 2*a^3*c^3*k*p^3*q^2*s - a^3*c^3*k*p*q^4*s + a^4*c^3*p^2*q^2*s*t + 6*a^2*c^3*f*g*p^5 + 6*a^2*c^3*f*g*p^3*q^2 - 2*a^2*c^3*h*k*p^3*q^2 + a^2*c^3*f*g*p*q^4 - a^2*c^3*h*k*p*q^4 + a^4*c^2*k*p^2*q^2*s + a^3*c^3*h*p^2*q^2*t - 4*a^3*c^2*f*g*p^4 - 2*a^3*c^2*f*g*p^2*q^2 + a^3*c^2*h*k*p^2*q^2 + a^4*c*f*g*p^3
K0u = -a^4*c^3*k*p*q^4*s + 4*a^3*c^3*f*g*p^3*q^2 + 2*a^3*c^3*f*g*p*q^4 - a^3*c^3*h*k*p*q^4 - a^4*c^2*f*g*p^2*q^2 - a^2*c^3*k*p^3*q^2*s - a^2*c^3*k*p*q^4*s + 2*a^3*c^3*p^2*q^2*s*t + a^3*c^3*q^4*s*t + 4*a*c^3*f*g*p^5 + 6*a*c^3*f*g*p^3*q^2 - a*c^3*h*k*p^3*q^2 + 2*a*c^3*f*g*p*q^4 - a*c^3*h*k*p*q^4 + 2*a^3*c^2*k*p^2*q^2*s + a^3*c^2*k*q^4*s + 2*a^2*c^3*h*p^2*q^2*t + a^2*c^3*h*q^4*t - a^4*c^2*p*q^2*s*t - 6*a^2*c^2*f*g*p^4 - 6*a^2*c^2*f*g*p^2*q^2 + 2*a^2*c^2*h*k*p^2*q^2 - a^2*c^2*f*g*q^4 + a^2*c^2*h*k*q^4 - a^4*c*k*p*q^2*s - a^3*c^2*h*p*q^2*t + 4*a^3*c*f*g*p^3 + 2*a^3*c*f*g*p*q^2 - a^3*c*h*k*p*q^2 - a^4*f*g*p^2

K3w = a^3*c^4*f*g*(p^9 - p^7*q^2 - 5*p^5*q^4 - 3*p^3*q^6)
K2w = -a^2*c^4*f*g*p^8 - a^2*c^4*f*g*p^6*q^2 + a^2*c^4*f*g*p^4*q^4 + a^2*c^4*f*g*p^2*q^6 - a^3*c^4*h*p^5*q^2*t + a^3*c^4*h*p*q^6*t + 4*a^3*c^3*f*g*p^7 + 2*a^3*c^3*f*g*p^5*q^2 - a^3*c^3*h*k*p^5*q^2 - 4*a^3*c^3*f*g*p^3*q^4 - 2*a^3*c^3*f*g*p*q^6 + a^3*c^3*h*k*p*q^6
K1w = 3*a^3*c^4*f*g*p^7*q^2 + 6*a^3*c^4*f*g*p^5*q^4 + 3*a^3*c^4*f*g*p^3*q^6 + a^3*c^4*p^4*q^2*s*t + a^3*c^4*p^2*q^4*s*t + a*c^4*f*g*p^7 + 2*a*c^4*f*g*p^5*q^2 + a*c^4*f*g*p^3*q^4 + a^3*c^3*k*p^4*q^2*s + a^3*c^3*k*p^2*q^4*s + a^2*c^4*h*p^4*q^2*t + a^2*c^4*h*p^2*q^4*t - 4*a^2*c^3*f*g*p^6 - 6*a^2*c^3*f*g*p^4*q^2 + a^2*c^3*h*k*p^4*q^2 - 2*a^2*c^3*f*g*p^2*q^4 + a^2*c^3*h*k*p^2*q^4 - 2*a^3*c^3*h*p^3*q^2*t - a^3*c^3*h*p*q^4*t + 6*a^3*c^2*f*g*p^5 + 6*a^3*c^2*f*g*p^3*q^2 - 2*a^3*c^2*h*k*p^3*q^2 + a^3*c^2*f*g*p*q^4 - a^3*c^2*h*k*p*q^4
K0w = -a^2*c^4*f*g*p^6*q^2 - 2*a^2*c^4*f*g*p^4*q^4 - a^2*c^4*f*g*p^2*q^6 - a^3*c^4*h*p^3*q^4*t - a^3*c^4*h*p*q^6*t + 4*a^3*c^3*f*g*p^5*q^2 + 6*a^3*c^3*f*g*p^3*q^4 - a^3*c^3*h*k*p^3*q^4 + 2*a^3*c^3*f*g*p*q^6 - a^3*c^3*h*k*p*q^6 - a^2*c^4*p^3*q^2*s*t - a^2*c^4*p*q^4*s*t - c^4*f*g*p^6 - 2*c^4*f*g*p^4*q^2 - c^4*f*g*p^2*q^4 - a^2*c^3*k*p^3*q^2*s - a^2*c^3*k*p*q^4*s - a*c^4*h*p^3*q^2*t - a*c^4*h*p*q^4*t + 2*a^3*c^3*p^2*q^2*s*t + a^3*c^3*q^4*s*t + 4*a*c^3*f*g*p^5 + 6*a*c^3*f*g*p^3*q^2 - a*c^3*h*k*p^3*q^2 + 2*a*c^3*f*g*p*q^4 - a*c^3*h*k*p*q^4 + 2*a^3*c^2*k*p^2*q^2*s + a^3*c^2*k*q^4*s + 2*a^2*c^3*h*p^2*q^2*t + a^2*c^3*h*q^4*t - 6*a^2*c^2*f*g*p^4 - 6*a^2*c^2*f*g*p^2*q^2 + 2*a^2*c^2*h*k*p^2*q^2 - a^2*c^2*f*g*q^4 + a^2*c^2*h*k*q^4 - a^3*c^2*h*p*q^2*t + 4*a^3*c*f*g*p^3 + 2*a^3*c*f*g*p*q^2 - a^3*c*h*k*p*q^2

Puw = a^2*c^4*p^3*q^2*s*t + a^2*c^4*p*q^4*s*t + c^4*f*g*p^6 + 2*c^4*f*g*p^4*q^2 + c^4*f*g*p^2*q^4 + a^2*c^3*k*p^3*q^2*s + a^2*c^3*k*p*q^4*s + a*c^4*h*p^3*q^2*t + a*c^4*h*p*q^4*t - 2*a^3*c^3*p^2*q^2*s*t - a^3*c^3*q^4*s*t - 4*a*c^3*f*g*p^5 - 6*a*c^3*f*g*p^3*q^2 + a*c^3*h*k*p^3*q^2 - 2*a*c^3*f*g*p*q^4 + a*c^3*h*k*p*q^4 - 2*a^3*c^2*k*p^2*q^2*s - a^3*c^2*k*q^4*s - 2*a^2*c^3*h*p^2*q^2*t - a^2*c^3*h*q^4*t + a^4*c^2*p*q^2*s*t + 6*a^2*c^2*f*g*p^4 + 6*a^2*c^2*f*g*p^2*q^2 - 2*a^2*c^2*h*k*p^2*q^2 + a^2*c^2*f*g*q^4 - a^2*c^2*h*k*q^4 + a^4*c*k*p*q^2*s + a^3*c^2*h*p*q^2*t - 4*a^3*c*f*g*p^3 - 2*a^3*c*f*g*p*q^2 + a^3*c*h*k*p*q^2 + a^4*f*g*p^2
"""

def k_h(x, a, b, c, d, gamma, s, t, k, h, f, g):
    P_o = a * np.cos(x) + b 
    Q_o = c * np.cos(x - gamma) + d
    P_n = h * np.cos(x) - s 
    Q_n = k * np.cos(x - gamma) - t
    P_m = f * np.sin(x)
    Q_m = g * np.sin(x - gamma)
    return 4*P_m*Q_m*(P_n*Q_n + P_m*Q_m)/((1+P_o)*(1+Q_o))
    

def H0(a, c, gamma, s, t, k, h, f, g):
    p, q = np.cos(gamma), np.sin(gamma)
    nu = a**4 * c**4 * q**4

    Kc =  (a**3 * c**3) * (q**4) * (h*k*p)/2 \
        + (a**3 * c**3) * (q**4 - 2*q**2) * (s*t) \
        + (a**3 * c**3) * (q**4 - 2*q**2 - 2) * (f*g*p)/2 \
        + (a**2 * c**2) * (q**4 - 2*q**2) * h*k \
        + (a**2 * c**2) * (q**4 - 6*q**2 + 6) * f*g \
        + (a**2 * c**2) * (a*p**3 + c) * (k*s*p - h*t) \
        + (a**2 * c**2) * (c*p**3 + a) * (h*t*p - k*s) \
        + (a**2 + c**2) * (a*c) * (q**2) * h*k*p \
        + (a**2 + c**2) * (a*c) * (q**2 - 2) * 2*f*g*p
        
    K3 = + a**3*c**3 * p**3 * (1 - 4*q**2)
    K2 = - a**2*c**2 * p**1 * (1 - 2*q**2) 
    K1 = + a**1*c**1 * p**1 

    K0u = -a**4*c**3*k*p*q**4*s + 4*a**3*c**3*f*g*p**3*q**2 + 2*a**3*c**3*f*g*p*q**4 - a**3*c**3*h*k*p*q**4 - a**4*c**2*f*g*p**2*q**2 - a**2*c**3*k*p**3*q**2*s - a**2*c**3*k*p*q**4*s + 2*a**3*c**3*p**2*q**2*s*t + a**3*c**3*q**4*s*t + 4*a*c**3*f*g*p**5 + 6*a*c**3*f*g*p**3*q**2 - a*c**3*h*k*p**3*q**2 + 2*a*c**3*f*g*p*q**4 - a*c**3*h*k*p*q**4 + 2*a**3*c**2*k*p**2*q**2*s + a**3*c**2*k*q**4*s + 2*a**2*c**3*h*p**2*q**2*t + a**2*c**3*h*q**4*t - a**4*c**2*p*q**2*s*t - 6*a**2*c**2*f*g*p**4 - 6*a**2*c**2*f*g*p**2*q**2 + 2*a**2*c**2*h*k*p**2*q**2 - a**2*c**2*f*g*q**4 + a**2*c**2*h*k*q**4 - a**4*c*k*p*q**2*s - a**3*c**2*h*p*q**2*t + 4*a**3*c*f*g*p**3 + 2*a**3*c*f*g*p*q**2 - a**3*c*h*k*p*q**2 - a**4*f*g*p**2
    K0w = -a**2*c**4*f*g*p**6*q**2 - 2*a**2*c**4*f*g*p**4*q**4 - a**2*c**4*f*g*p**2*q**6 - a**3*c**4*h*p**3*q**4*t - a**3*c**4*h*p*q**6*t + 4*a**3*c**3*f*g*p**5*q**2 + 6*a**3*c**3*f*g*p**3*q**4 - a**3*c**3*h*k*p**3*q**4 + 2*a**3*c**3*f*g*p*q**6 - a**3*c**3*h*k*p*q**6 - a**2*c**4*p**3*q**2*s*t - a**2*c**4*p*q**4*s*t - c**4*f*g*p**6 - 2*c**4*f*g*p**4*q**2 - c**4*f*g*p**2*q**4 - a**2*c**3*k*p**3*q**2*s - a**2*c**3*k*p*q**4*s - a*c**4*h*p**3*q**2*t - a*c**4*h*p*q**4*t + 2*a**3*c**3*p**2*q**2*s*t + a**3*c**3*q**4*s*t + 4*a*c**3*f*g*p**5 + 6*a*c**3*f*g*p**3*q**2 - a*c**3*h*k*p**3*q**2 + 2*a*c**3*f*g*p*q**4 - a*c**3*h*k*p*q**4 + 2*a**3*c**2*k*p**2*q**2*s + a**3*c**2*k*q**4*s + 2*a**2*c**3*h*p**2*q**2*t + a**2*c**3*h*q**4*t - 6*a**2*c**2*f*g*p**4 - 6*a**2*c**2*f*g*p**2*q**2 + 2*a**2*c**2*h*k*p**2*q**2 - a**2*c**2*f*g*q**4 + a**2*c**2*h*k*q**4 - a**3*c**2*h*p*q**2*t + 4*a**3*c*f*g*p**3 + 2*a**3*c*f*g*p*q**2 - a**3*c*h*k*p*q**2
    
    Kuw = a**2*c**4*p**3*q**2*s*t + a**2*c**4*p*q**4*s*t + c**4*f*g*p**6 + 2*c**4*f*g*p**4*q**2 + c**4*f*g*p**2*q**4 + a**2*c**3*k*p**3*q**2*s + a**2*c**3*k*p*q**4*s + a*c**4*h*p**3*q**2*t + a*c**4*h*p*q**4*t - 2*a**3*c**3*p**2*q**2*s*t - a**3*c**3*q**4*s*t - 4*a*c**3*f*g*p**5 - 6*a*c**3*f*g*p**3*q**2 + a*c**3*h*k*p**3*q**2 - 2*a*c**3*f*g*p*q**4 + a*c**3*h*k*p*q**4 - 2*a**3*c**2*k*p**2*q**2*s - a**3*c**2*k*q**4*s - 2*a**2*c**3*h*p**2*q**2*t - a**2*c**3*h*q**4*t + a**4*c**2*p*q**2*s*t + 6*a**2*c**2*f*g*p**4 + 6*a**2*c**2*f*g*p**2*q**2 - 2*a**2*c**2*h*k*p**2*q**2 + a**2*c**2*f*g*q**4 - a**2*c**2*h*k*q**4 + a**4*c*k*p*q**2*s + a**3*c**2*h*p*q**2*t - 4*a**3*c*f*g*p**3 - 2*a**3*c*f*g*p*q**2 + a**3*c*h*k*p*q**2 + a**4*f*g*p**2
    
    H_c = 1/nu * Kc * (2*PI)

    H_3 = 1/nu * K3 * (
        (f*g) * (a*W(a, 0, 3) + c*W(c, 0, 3))
    )

    H_2 = 1/nu * K2 * (
        + ((a**2) * W(a, 0, 2) + (c**2) * W(c, 0, 2)) * (f*g)*p \
        + (a*c) * ((k*s)*a*W(a, 0, 2) + (h*t)*c*W(c, 0, 2)) * q**2\
        + (a*c) * (W(a, 0, 2) + W(c, 0, 2)) * ((h*k)*q**2 + (2*f*g)*(q**2 - 2))\
    )

    H_1 = 1/nu * K1 * (
        + (a**3*W(a, 0, 1) + c**3*W(c, 0, 1)) * p**2 * (f*g)
        + (a*c) * ((k*s)*a**2*W(a, 0, 1) + (h*t)*c**2*W(c, 0, 1)) * p*q**2 
        + (a**2*c**2) * (a*W(a, 0, 1) + c*W(c, 0, 1)) * p*q**2 * (3*f*g*p + s*t)
        + (a*c) * (a*W(a, 0, 1) + c*W(c, 0, 1)) * p * ((h*k)*q**2 + (2*f*g)*(q**2 - 2))
        + (a*c) * (c*W(a, 0, 1) + a*W(c, 0, 1)) * ((q**4 - 6*q**2 + 6)*(f*g) + (q**4 - 2*q**2)*(h*k)) 
        + (a**2*c**2) * (((h*t)*p + (k*s)*(q**2 - 2))*W(a, 0, 1) + ((k*s)*p + (h*t)*(q**2 - 2))*W(c, 0, 1)) * q**2
    )

    H_0 = 1/nu * (
        K0u * W(a, 0, 0) + K0w * W(c, 0, 0)
    )

    H_uw = 1/nu * Kuw * Z0(a, c, gamma)
    
    return  (H_c + H_3 + H_2 + H_1 + H_0 + H_uw)

def H3(a, b, c, d, gamma, s, t, k, h, f, g):
    return 4*f*g*H0(a/(b+1), c/(d+1), gamma, s, t, k, h, f, g)/((b+1)*(d+1))
    


N = 1000000
calc = Calculator(N)

M = calc.N
Gamma = 2*PI * np.random.random(M)
Theta = PI * np.random.random(M)
Alpha = PI * np.random.random(M)
Beta = PI * np.random.random(M)
A = np.sin(Alpha) * np.sin(Theta)
B = np.cos(Alpha) * np.cos(Theta)
C = np.sin(Beta) * np.sin(Theta)
D = np.cos(Beta) * np.cos(Theta)
H = np.sin(Alpha) * np.cos(Theta)
S = np.cos(Alpha) * np.sin(Theta)
K = np.sin(Beta) * np.cos(Theta)
T = np.cos(Beta) * np.sin(Theta)
F = np.sin(Alpha)
G = np.sin(Beta)

# A = np.random.random(M)
# C = np.random.random(M)
# B = 2*np.random.random(M) - 1
# D = 2*np.random.random(M) - 1
# S = 2*np.random.random(M) - 1
# T = 2*np.random.random(M) - 1
# K = 2*np.random.random(M) - 1
# H = 2*np.random.random(M) - 1
# F = np.random.random(M)
# G = np.random.random(M)


phi = calc.phi

# delta_expr = (f(phi, A, C, Gamma) - f_red(phi, A, C, Gamma))
# print(delta_expr[delta_expr>10.0**(-8)])

Int_true, Int_est = calc.test_acc(k_h, H3, log_eps=-8, a=A, b=B, c=C, d=D, gamma=Gamma, s=S, t=T, k=K, h=H, f=F, g=G)


# plt.scatter(Gamma[:10], Int_true)
# plt.scatter(Gamma[:10], Int_est)
# plt.show()