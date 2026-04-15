import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
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
    def test_acc(self, func, integral, log_eps=-14, **Params):
        is_accur = True
        Int_true, Int_est = [], []
        eps = 10.0**(log_eps)
        for i in range(M):
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
                print(f"log-delta = {np.log10(delta):.2f} > {log_eps}")
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

def H1(a, b, c, d, gamma):
    s = np.sqrt((b+1)**2 - a**2)
    q = np.sqrt((d+1)**2 - c**2) 
    return 2*PI*b*d/(s*q - a*c*np.cos(gamma) + (b+1)*(d+1)) * (np.cos(gamma) - (s - (b+1))/a * (q - (d+1))/c)

N = 10000
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
# A = np.random.random(M)
# C = np.random.random(M)
# B = 2*np.random.random(M) - 1
# D = 2*np.random.random(M) - 1
K = np.random.randint(0, 4, M) #*np.ones(M)
L = np.random.randint(0, 4, M)

phi = calc.phi

delta_expr = (f(phi, A, C, Gamma) - f_red(phi, A, C, Gamma))
# print(delta_expr[delta_expr>10.0**(-8)])

Int_true, Int_est = calc.test_acc(h1, H1, log_eps=-8, a=A, b=B, c=C, d=D, gamma=Gamma)


plt.scatter(Gamma, Int_true)
plt.scatter(Gamma, Int_est)
plt.show()