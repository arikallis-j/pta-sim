import numpy as np
import matplotlib.pyplot as plt

PI = np.pi
N = 100
phi = np.linspace(0, 2*PI, N, endpoint=False)
dphi = np.ones(N) * 2*PI / N 


def f_int_1(phi, a):
    return np.sin(phi)**2/(1 + a*np.cos(phi))

def int_true_1(a):
    return 2*PI/a**2 * (1 - np.sqrt(1 - a**2))

def f_int_3(phi, a):
    return np.sin(phi)**2 * np.cos(phi)/(1 + a*np.cos(phi))

def int_true_3(a):
    return 2*PI/a**3 * (a**2/2 - (1 - np.sqrt(1 - a**2)) )

def f_int_2(phi, a):
    return np.sin(phi)**2 * np.sin(phi)/(1 + a*np.cos(phi))

def f_int_h2(phi, a):
    return np.sin(phi) * np.sin(phi)/(1 + a*np.cos(phi))**2


def f_int_j(phi, theta, alpha, beta, gamma):
    a = np.sin(alpha) * np.sin(theta)
    b = np.cos(alpha) * np.cos(theta)
    c = np.sin(beta) * np.sin(theta)
    d = np.cos(beta) * np.cos(theta)
    A, B, C, D = a, b+1, c, d-1
    return np.sin(phi)**2 * (C * np.cos(phi - gamma) + D) / (A*np.cos(phi) + B)

def f_int_k(phi, theta, alpha, beta, gamma):
    a = np.sin(alpha) * np.sin(theta)
    b = np.cos(alpha) * np.cos(theta)
    c = np.sin(beta) * np.sin(theta)
    d = np.cos(beta) * np.cos(theta)
    A, B, C, D = a, b-1, c, d+1
    return np.sin(phi-gamma)**2 * (A * np.cos(phi) + B) / (C*np.cos(phi - gamma) + D)

def int_true_j(theta, alpha, beta, gamma):
    a = np.sin(alpha) * np.sin(theta)
    b = np.cos(alpha) * np.cos(theta)
    c = np.sin(beta) * np.sin(theta)
    d = np.cos(beta) * np.cos(theta)
    A, B, C, D = a, b+1, c, d-1
    S = np.sqrt(B**2 - A**2)
    return 2*PI * (D * (B-S)/A**2 - C*np.cos(gamma)/2 * (B-S)**2/A**3)

def int_true_k(theta, alpha, beta, gamma):
    return int_true_j(theta, beta, alpha, gamma)


def f_int_h1a1(phi, theta, alpha, beta, gamma, x):
    a = np.sin(alpha) * np.sin(theta)
    b = np.cos(alpha) * np.cos(theta)
    c = np.sin(beta) * np.sin(theta)
    d = np.cos(beta) * np.cos(theta)
    A, B, C, D, X = a, b+1, c, d+1, x
    K = X*B + (1-X)*D
    R = np.sqrt((X*A)**2 + ((1-X)*C)**2 + 2*(A*X)*((1-X)*C)*np.cos(gamma))
    return np.sin(phi)**2 / (R*np.cos(phi) + K)**2

def f_int_h1a2(phi, theta, alpha, beta, gamma, x):
    a = np.sin(alpha) * np.sin(theta)
    b = np.cos(alpha) * np.cos(theta)
    c = np.sin(beta) * np.sin(theta)
    d = np.cos(beta) * np.cos(theta)
    A, B, C, D, X = a, b+1, c, d+1, x
    K = X*B + (1-X)*D
    R = np.sqrt((X*A)**2 + ((1-X)*C)**2 + 2*(A*X)*((1-X)*C)*np.cos(gamma))
    return np.cos(phi)**2 / (R*np.cos(phi) + K)**2

def f_int_h1a3(phi, theta, alpha, beta, gamma, x):
    a = np.sin(alpha) * np.sin(theta)
    b = np.cos(alpha) * np.cos(theta)
    c = np.sin(beta) * np.sin(theta)
    d = np.cos(beta) * np.cos(theta)
    A, B, C, D, X = a, b+1, c, d+1, x
    K = X*B + (1-X)*D
    R = np.sqrt((X*A)**2 + ((1-X)*C)**2 + 2*(A*X)*((1-X)*C)*np.cos(gamma))
    return np.sin(phi)*np.cos(phi) / (R*np.cos(phi) + K)**2

def int_true_h1a1(theta, alpha, beta, gamma, x):
    a = np.sin(alpha) * np.sin(theta)
    b = np.cos(alpha) * np.cos(theta)
    c = np.sin(beta) * np.sin(theta)
    d = np.cos(beta) * np.cos(theta)
    A, B, C, D, X = a, b+1, c, d+1, x
    K = X*B + (1-X)*D
    R = np.sqrt((X*A)**2 + ((1-X)*C)**2 + 2*(A*X)*((1-X)*C)*np.cos(gamma))
    T = np.sqrt(K**2 - R**2)
    return 2*PI/(R**2) * (K/T - 1)

def int_true_h1a2(theta, alpha, beta, gamma, x):
    a = np.sin(alpha) * np.sin(theta)
    b = np.cos(alpha) * np.cos(theta)
    c = np.sin(beta) * np.sin(theta)
    d = np.cos(beta) * np.cos(theta)
    A, B, C, D, X = a, b+1, c, d+1, x
    K = X*B + (1-X)*D
    R = np.sqrt((X*A)**2 + ((1-X)*C)**2 + 2*(A*X)*((1-X)*C)*np.cos(gamma))
    T = np.sqrt(K**2 - R**2)
    return 2*PI/(R**2) * (K/T * R**2/T**2 - K/T + 1)


M = 100
Theta = PI * (0.25*np.random.random(M) + 0.5)
Alpha = PI * (0.25*np.random.random(M) + 0.5)
Beta = PI * (0.25*np.random.random(M) + 0.5)
Gamma = 2*PI * (0.25*np.random.random(M) + 0.5)
X = (0.25*np.random.random(M) + 0.5)

Int_est = []
Int_true = []
for k in range(M):
    theta, alpha, beta, gamma, x = Theta[k], Alpha[k], Beta[k], Gamma[k], X[k]
    int_est = np.sum(dphi * f_int_h1a2(phi, theta, alpha, beta, gamma, x))
    Int_est.append(int_est)
    int_true = int_true_h1a2(theta, alpha, beta, gamma, x)
    Int_true.append(int_true)

Int_est = np.array(Int_est)
Int_true = np.array(Int_true)
plt.scatter(Theta, Int_est - Int_true)
# plt.scatter(Theta, )
plt.show()


