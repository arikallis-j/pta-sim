import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from tqdm import tqdm
PI = np.pi

class Calculator:
    def __init__(self, N=10, rng =(0, 2*PI)):
        self.setup(N, rng)

    def setup(self,N, rng = (0, 2*PI)):
        self.a, self.b = rng
        self.N = N
        self.x = np.linspace(self.a, self.b, N, endpoint=False)
        self.dx = np.ones(N) * (self.b - self.a) / N 
    
    def true_int(self, integral, **params):
        return integral(**params)
    def est_int(self, func, **params):
        return np.sum(self.dx * func(self.x, **params))
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


def h51(x, a, c, gamma):
    A = np.cos(x) * np.cos(x - gamma) * np.cos(x) * np.cos(x - gamma)
    B = ((a*np.cos(x) + 1)*(c*np.cos(x - gamma) + 1))
    return A/B

def H51(a, c, gamma):
    p, q = np.cos(gamma), np.sin(gamma)
    nu = a**2 * c**2
    s = np.sqrt(1 - a**2)
    q = np.sqrt(1 - c**2)

    H_0 = 2*PI/(a**2 * c**2) * (
        + 1 - (s + q)/(s*q - a*c*np.cos(gamma) + 1)
        + (a*c*np.cos(gamma)) * (1/2 + 1/s * (s - 1)/a**2 + 1/q * (q - 1)/c**2)
        + (a*c*np.cos(gamma)) * (1/s + 1/q)/(1 - a*c*p + s*q)  
    )
    return H_0

def h52(x, a, c, gamma):
    A = np.cos(x) * np.cos(x)
    B = ((a*np.cos(x) + 1)*(c*np.cos(x - gamma) + 1))
    return A/B
    
def H52(a, c, gamma):
    p, q = np.cos(gamma), np.sin(gamma)
    return 1/a**2 * (V(c, 1) * a*p - V(c, 0) + Z(a,c,gamma))

def h53(x, a, c, gamma):
    A = np.cos(x-gamma) * np.cos(x-gamma)
    B = ((a*np.cos(x) + 1)*(c*np.cos(x - gamma) + 1))
    return A/B
    
def H53(a, c, gamma):
    p, q = np.cos(gamma), np.sin(gamma)
    return 1/c**2 * (V(a, 1) * c*p - V(a, 0) + Z(a,c,gamma))


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

def z(x, a, c, gamma):
    return  1/((1 + a*np.cos(x))*(1 + c*np.cos(x - gamma)))

def Z(a, c, gamma):
    b = np.sqrt(1 - a**2)
    d = np.sqrt(1 - c**2)
    s = np.sqrt(1 - a * c * np.cos(gamma))
    return (2*PI) * (b+d)/(b*d*(s**2 + b*d))

def func_j(x, a, b, c, d, gamma):
    return np.sin(x)**2 * (c*np.cos(x - gamma) + (d - 1))/ (a*np.cos(x) + (b+1))

def J(a, b, c, d, gamma):
    x0 = ((b+1) - np.sqrt((b+1)**2 - a**2))/a**2
    return 2*PI * (x0 * (d - 1) - x0**2 * a*c*np.cos(gamma)/2)

def h1(x, a, c, gamma):
    A = np.sin(x) * np.sin(x - gamma)
    B = ((a*np.cos(x) + 1)*(c*np.cos(x - gamma) + 1))
    return A/B

def H1(a, c, gamma):
    p = np.cos(gamma)
    s = np.sqrt(1 - a**2)
    q = np.sqrt(1 - c**2) 
    return 2*PI/(a**2*c**2) * 1/(s*q - a*c*p + 1) * (
        (a*c) * (a*c*p - (s-1)*(q-1))
    )

def h2(x, a, c, gamma):
    A = np.sin(x) * np.sin(x - gamma) * np.cos(x)
    B = ((a*np.cos(x) + 1)*(c*np.cos(x - gamma) + 1))
    return A/B

def H2(a, c, gamma):
    p = np.cos(gamma)
    s = np.sqrt(1 - a**2)
    q = np.sqrt(1 - c**2)
    return 2*PI/(a**2*c**2) *  1/(s*q - a*c*p+ 1) * (
        - (c+a*p) * ((a*c*p) - (s-1)*(q-1))
        + (c*s + a*p*q) * (a*c*p)
    )

def h3(x, a, c, gamma):
    A = np.sin(x) * np.sin(x - gamma) * np.cos(x - gamma)
    B = ((a*np.cos(x) + 1)*(c*np.cos(x - gamma) + 1))
    return A/B

def H3(a, c, gamma):
    p = np.cos(gamma)
    s = np.sqrt(1 - a**2)
    q = np.sqrt(1 - c**2)
    return 2*PI/(a**2*c**2) *  1/(s*q - a*c*p+ 1) * (
        - (a+c*p) * ((a*c*p) - (s-1)*(q-1))
        + (a*q + c*p*s) * (a*c*p)
    )

def h4(x, a, c, gamma):
    A = np.sin(x) * np.sin(x - gamma) * np.cos(x) * np.cos(x - gamma)
    B = ((a*np.cos(x) + 1)*(c*np.cos(x - gamma) + 1))
    return A/B

def H4(a, c, gamma):
    p = np.cos(gamma)
    s = np.sqrt(1 - a**2)
    q = np.sqrt(1 - c**2)
    return 2*PI/(a**2*c**2) * 1/(s*q - a*c*p + 1) * (
        + (1 - p/(2*a*c) * (s-1)*(q-1) * ((s-1)*(q-1)-4)) * (a*c*p - (s-1)*(q-1)) 
        + (s+q) * (p/(2*a*c) * (s-1)*(q-1) * ((s-1)*(q-1)-4))
    )

def h5(x, a, c, gamma):
    A = np.sin(x) * np.sin(x - gamma) * np.sin(x) * np.sin(x - gamma)
    B = ((a*np.cos(x) + 1)*(c*np.cos(x - gamma) + 1))
    return A/B

def H5(a, c, gamma):
    p = np.cos(gamma)
    s = np.sqrt(1 - a**2)
    q = np.sqrt(1 - c**2)
    return 2*PI/(a**2*c**2) * 1/(s*q - a*c*p + 1) * (
        + (- 1 + (s+q) - p/(2*a*c) * (s-1)*(q-1) * ((s-1)*(q-1) - 4*s*q)) * (a*c*p - (s-1)*(q-1)) 
        + (s+q) * ((s-1)*(q-1) + p/(2*a*c) * (s-1)*(q-1) * ((s-1)*(q-1) - 4*s*q))
    )


def h0(x, a, c, gamma, s, t, k, h, f, g):
    P_o = a * np.cos(phi) + 1
    Q_o = c * np.cos(phi - gamma) + 1
    P_n = h * np.cos(phi) - s
    Q_n = k * np.cos(phi - gamma) - t
    P_m = f * np.sin(phi)
    Q_m = g * np.sin(phi - gamma)
    return 4 * P_m*Q_m * (P_m*Q_m + P_n*Q_n) / (P_o * Q_o)


def H0(a, c, gamma, s, t, k, h, f, g):
    p = np.cos(gamma)
    a1 = np.sqrt(1 - a**2)
    c1 = np.sqrt(1 - c**2)
    H_0 = f*g*4*PI/(a**3*c**3) * (
        + p * (a1-1)*(c1-1) * ((h*k + f*g) * (a1-1)*(c1-1) - 4*(h*k + f*g*a1*c1))
        + 2 * (a*c)/(a1*c1 - a*c*p + 1) * (
            + (a*c*p - (a1-1)*(c1-1)) * ((s*t*a*c) + (h*k - f*g) + t*h*(c+a*p) + s*k*(a + c*p))
            - (a*c*p) * (t*h*(c*a1 + a*c1*p) + s*k*(a*c1 + c*a1*p) - f*g*(a1 + c1))
        )
    )
    return H_0

def h_func(phi, theta, alpha, beta, gamma):
    a = np.sin(alpha) * np.sin(theta)
    b = np.cos(alpha) * np.cos(theta)
    c = np.sin(beta) * np.sin(theta)
    d = np.cos(beta) * np.cos(theta)
    h = np.sin(alpha) * np.cos(theta)
    s = np.cos(alpha) * np.sin(theta)
    k = np.sin(beta) * np.cos(theta)
    t = np.cos(beta) * np.sin(theta)
    f = np.sin(alpha)
    g = np.sin(beta)

    P_o = a * np.cos(phi) + b
    Q_o = c * np.cos(phi - gamma) + d
    P_n = h * np.cos(phi) - s
    Q_n = k * np.cos(phi - gamma) - t
    P_m = f * np.sin(phi)
    Q_m = g * np.sin(phi - gamma)

    return 4 * P_m*Q_m * (P_m*Q_m + P_n*Q_n)/((1 + P_o) * (1 + Q_o))


def h_func_int(theta, alpha, beta, gamma, N=1000000):
    phi = np.linspace(0, 2*PI, N, endpoint=False)
    dphi = np.ones(N) * 2*PI/ N 
    return np.sum(dphi * h_func(phi, theta, alpha, beta, gamma))

def h_func_int_int(alpha, beta, gamma, kappa, N=1000000):
    theta = np.linspace(0, PI, N, endpoint=False)
    dtheta = np.ones(N) * PI/ N 
    kernel = np.sin(theta) * np.exp(kappa*np.cos(theta)) * h_func_int(theta, alpha, beta, gamma)
    return np.sum(dtheta * kernel)


def H_int(theta, alpha, beta, gamma):
    a = np.cos(alpha)
    b = np.cos(beta)
    c = np.sin(alpha)*np.sin(beta)*np.cos(gamma) + np.cos(alpha)*np.cos(beta)
    x = np.cos(theta)

    H_0 = + 8*PI * (a*b - c) * (
        +  1/2
        +  ((a*x+1)-np.abs(a+x))/(x**2-1) * 1/(1-a**2)
        +  ((b*x+1)-np.abs(b+x))/(x**2-1) * 1/(1-b**2)
        + (1-c) * (
            1 + (np.abs(a+x) - (a*x+1))*(np.abs(b+x) - (b*x+1))/((x**2-1)*(c - a*b))
        ) / (np.abs(a+x)*np.abs(b+x) + (a*x+1)*(b*x+1) + (x**2-1)*(c - a*b))
    )   

    return  H_0

from scipy.special import expi 

def hh_func(x, alpha, beta, gamma, kappa, eps=1e-15):
    a, b = np.cos(alpha), np.cos(beta)
    c = np.cos(alpha) * np.cos(beta) + np.sin(alpha) * np.sin(beta) * np.cos(gamma)
    k = kappa

    x = np.where(np.abs(x) < eps, eps, x)
    x = np.where(np.abs(x+1) < eps, -1 + eps, x)
    x = np.where(np.abs(x-1) < eps, +1 - eps, x)

    H_0 = + 8*PI * (a*b - c) * (
        +  1/2
        +  ((a*x+1)-np.abs(a+x))/(x**2-1) * 1/(1-a**2)
        +  ((b*x+1)-np.abs(b+x))/(x**2-1) * 1/(1-b**2)
        + (1-c) * (
            1 + (np.abs(a+x) - (a*x+1))*(np.abs(b+x) - (b*x+1))/((x**2-1)*(c - a*b))
        ) / (np.abs(a+x)*np.abs(b+x) + (a*x+1)*(b*x+1) + (x**2-1)*(c - a*b))
    )   

    return (H_0) * np.exp(kappa*x)


def Hh_int(alpha, beta, gamma, kappa):
    a, b = np.cos(alpha), np.cos(beta)
    c = np.cos(alpha) * np.cos(beta) + np.sin(alpha) * np.sin(beta) * np.cos(gamma)
    k = kappa

    t = (a + b)/(1+c)
    s = np.sqrt(1 + 2*a*b*c - c**2 - a**2 - b**2)/(1+c)
    r = (t + 1j*s)
    H_0 = (
        + 2*(a*b - c)*np.sinh(k)/k
        + 2*(a*b - c)/(1+a) * np.exp(-k) * (expi(+k*(1-a)) - expi(+2*k)) 
        + 2*(a*b - c)/(1+b) * np.exp(-k) * (expi(+k*(1-b)) - expi(+2*k)) 
        + 2*(a*b - c)/(1-b) * np.exp(+k) * (expi(-k*(1+b)) - expi(-2*k))
        + 2*(a*b - c)/(1-a) * np.exp(+k) * (expi(-k*(1+a)) - expi(-2*k))  
        - (1-c) * np.exp(-k) * (expi(+k*(1-a)) - expi(+2*k)) 
        - (1-c) * np.exp(-k) * (expi(+k*(1-b)) - expi(+2*k)) 
        - (1-c) * np.exp(+k) * (expi(-k*(1+a)) - expi(-2*k))
        - (1-c) * np.exp(+k) * (expi(-k*(1+b)) - expi(-2*k))
        - (a-b) * np.exp(+k) * (expi(-k*(1+a)) - expi(-k*(1+b)))
        - (a-b) * np.exp(-k) * (expi(+k*(1-b)) - expi(+k*(1-a)))
        - (1-c) * 2 * np.real(np.exp(-k*r) * (expi(+k*(r-1)) + expi(+k*(r+1))))
        + (1-c) * 2 * np.real(np.exp(-k*r) * (expi(+k*(r-a)) + expi(+k*(r-b))))
    )

    return  4*PI  * H_0

def Int_est(alpha, beta, gamma, kappa, nside = 8):
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    dOmega = hp.nside2pixarea(nside)

    a = np.sin(alpha) * np.sin(theta)
    b = np.cos(alpha) * np.cos(theta)
    c = np.sin(beta) * np.sin(theta)
    d = np.cos(beta) * np.cos(theta)
    h = np.sin(alpha) * np.cos(theta)
    s = np.cos(alpha) * np.sin(theta)
    k = np.sin(beta) * np.cos(theta)
    t = np.cos(beta) * np.sin(theta)
    f = np.sin(alpha)
    g = np.sin(beta)

    P_o = a * np.cos(phi) + b
    Q_o = c * np.cos(phi - gamma) + d
    P_n = h * np.cos(phi) - s
    Q_n = k * np.cos(phi - gamma) - t
    P_m = f * np.sin(phi)
    Q_m = g * np.sin(phi - gamma)


    K_I = (P_o - 1)*(Q_o - 1)
    K_J = 2 * P_m**2 * (Q_o - 1)/(P_o + 1)
    K_K = 2 * Q_m**2 * (P_o - 1)/(Q_o + 1)
    K_H = 4 * P_m*Q_m * (P_m*Q_m + P_n*Q_n)/((1 + P_o) * (1 + Q_o))
    K = K_I + K_J + K_K + K_H
    F = K * np.exp(kappa * np.cos(theta))
    return 1/(4*np.pi) * np.sum(dOmega * F)

def Int_theory(alpha, beta, gamma, kappa):
    a = np.cos(alpha) # (p·Ω)
    b = np.cos(beta) # (q·Ω)
    c = np.cos(alpha) * np.cos(beta) + np.sin(alpha) * np.sin(beta) * np.cos(gamma)  # (p·q)
    V = np.sqrt(1 + 2*a*b*c - c**2 - a**2 - b**2) # |(p×q)·Ω|
    k = kappa
    t = (a + b)/(1 + c)
    s = V/(1 + c) 

    def expi2(x, s):
        z = x + 1j*s
        return 2 * np.real(np.exp(-z) * expi(z))

    shc = np.sinh(k)/k
    chc_2 = (k*np.cosh(k) - np.sinh(k))/k**2
    chc_3 = 3*(k*np.cosh(k) - np.sinh(k))/k**3
    chc_a = (np.cosh(k) - np.exp(-k*a))/(a*k)
    chc_b = (np.cosh(k) - np.exp(-k*b))/(b*k)
        
    return (
        + 1/3 * (3*shc + chc_3)
        + 1 * (shc - chc_3) * (a*b)
        - (a+b) * (chc_2)

        + 2 * (chc_a - shc) * (a*b - a**2)/(1 - a**2)
        + 2 * (chc_b - shc) * (a*b - b**2)/(1 - b**2)
    
        + (1-c) * (
            + np.exp(-k*a) * expi2(k*(t-a),k*s) 
            + np.exp(-k*b) * expi2(k*(t-b),k*s)
            - np.exp(+k) * expi2(k*(t+1),k*s) 
            - np.exp(-k) * expi2(k*(t-1),k*s) 

            - 1/3 * (chc_3) 
            + 2 * (chc_a - shc) * a**2/(1 - a**2)
            + 2 * (chc_b - shc) * b**2/(1 - b**2)
        )
        
    )
# s_a = np.sqrt(1 - a**2)
# s_c = np.sqrt(1 - c**2)
# s_g = np.sqrt(1 - a*c*p)
# r_a = (s_a-1)/a
# r_c = (s_c-1)/c
# V0_a = 2*PI/s_a
# V0_c = 2*PI/s_c
# V1_a = 2*PI/s_a * r_a
# V1_c = 2*PI/s_c * r_c
# V2_a = 2*PI/s_a * r_a**2
# V2_c = 2*PI/s_c * r_c**2
# Z_ac = (2*PI) * (s_a+s_c)/(s_a*s_c*(s_g**2 + s_a*s_c))


N = 1000000
calc = Calculator(N)

M = calc.N
Theta = PI * np.random.random(M)
Alpha = PI * np.random.random(M)
Beta = PI * np.random.random(M)
Gamma = 2 * PI * np.random.random(M)
# Xi = PI * np.random.random(M)
Xi = np.arccos(np.cos(Alpha)*np.cos(Beta) + np.sin(Alpha)*np.sin(Beta)*np.cos(Gamma))
Kappa = 2*np.random.random(M)
A = np.cos(Alpha)
B = np.cos(Beta)
C = np.cos(Xi)
K = Kappa


# import healpy as hp

# NSIDE = 32
# print(
#     "Approximate resolution at NSIDE {} is {:.2} deg".format(
#         NSIDE, hp.nside2resol(NSIDE, arcmin=True) / 60
#     )
# )

# A = np.sin(Alpha) * np.sin(Theta)
# B = np.cos(Alpha) * np.cos(Theta)
# C = np.sin(Beta) * np.sin(Theta)
# D = np.cos(Beta) * np.cos(Theta)
# H = np.sin(Alpha) * np.cos(Theta)
# S = np.cos(Alpha) * np.sin(Theta)
# K = np.sin(Beta) * np.cos(Theta)
# T = np.cos(Beta) * np.sin(Theta)
# F = np.sin(Alpha)
# G = np.sin(Beta)

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


# phi = calc.x

# delta_expr = (f(phi, A, C, Gamma) - f_red(phi, A, C, Gamma))
# print(delta_expr[delta_expr>10.0**(-8)])

# Int_true, Int_est = calc.test_acc(h_func, h_func_int, log_eps=-8, alpha=Alpha, beta=Beta, gamma=Gamma, theta=Theta) 

calc.setup(N, rng=(-1,1))
Int_true, Int_est = calc.test_acc(hh_func, Hh_int, log_eps=-8, alpha=Alpha, beta=Beta, gamma=Gamma, kappa=Kappa) 

# plt.scatter(Gamma[:10], Int_true)
# plt.scatter(Gamma[:10], Int_est)
# plt.show()