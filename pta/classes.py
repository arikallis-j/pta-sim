import numpy as np
import matplotlib.pyplot as plt

from .consts import PI, EPS
from .funcs import t_sum, sin, cos, exp, sqrt, arccos, zeros, mu_0
from tqdm import tqdm

year = 3.1536e7 
pc = 3.26*year
kpc = 1e3 * pc


class GravitationalWave:
    def __init__(self, key='ipoint', param=(1,0,1e-9,0)):
        self.key = key
        self.param = param
        self._generate_wave()

    def _generate_wave(self):
        if self.key == 'ipoint':
            A, phi_0, f_0, psi_0 = self.param
            self.f_0 = f_0
            self.h = lambda t: A * exp(2j * (PI * f_0 * t + phi_0)) * exp(2j*psi_0)

class PulsarArray:
    def __init__(self, N=10, key='angle'):
        self.key = key
        self.N = N
        self._create_pulsar_array()

    def _create_pulsar_array(self):
        if self.key == 'angle':
            pulsar_array = []
            pulsar_dist = []
            pulsar_coord = []
            gamma = np.linspace(0, PI, self.N)
            
            for k in range(self.N):
                g = gamma[k]
                p = np.array([np.sin(g), 0, np.cos(g)])
                lp = 10 * kpc
                vp = lp * p
                pulsar_array.append(p)
                pulsar_dist.append(lp)
                pulsar_coord.append(vp)
        if self.key == 'cos':
            pulsar_array = []
            pulsar_dist = []
            pulsar_coord = []
            cos_gamma = np.linspace(-1, 1, self.N)
            gamma = arccos(cos_gamma)
            
            for k in range(self.N):
                g = gamma[k]
                p = np.array([np.sin(g), 0, np.cos(g)])
                lp = 10 * kpc
                vp = lp * p
                pulsar_array.append(p)
                pulsar_dist.append(lp)
                pulsar_coord.append(vp)
            
        if self.key == 'sphere':
            pulsar_array = []
            pulsar_dist = []
            pulsar_coord = []
            gamma = np.linspace(0, PI, self.N)
            phi = np.linspace(0, 2*PI, self.N, endpoint=False)
            for i in range(self.N):
                ph = phi[i]
                for k in range(self.N):
                    g = gamma[k]
                    p = np.array([cos(ph)*sin(g), sin(ph)*sin(g), cos(g)])
                    lp = 10 * kpc
                    vp = lp * p
                    pulsar_array.append(p)
                    pulsar_dist.append(lp)
                    pulsar_coord.append(vp)
        
        if self.key == 'ball':
            pulsar_array = []
            pulsar_dist = []
            pulsar_coord = []
            gamma = np.linspace(0, PI, self.N)
            phi = np.linspace(0, 2*PI, self.N, endpoint=False)
            Lp = np.linspace(0, 30, self.N)
            for i in range(self.N):
                lp = Lp[i] * kpc
                for j in range(self.N):
                    ph = phi[j]
                    for k in range(self.N):
                        g = gamma[k]
                        p = np.array([cos(ph)*sin(g), sin(ph)*sin(g), cos(g)])
                        vp = lp * p
                        pulsar_array.append(p)
                        pulsar_dist.append(lp)
                        pulsar_coord.append(vp)

        self.pulsar_array = np.array(pulsar_array)
        self.pulsar_dist = np.array(pulsar_dist)
        self.pulsar_coord = np.array(pulsar_coord)

class Grid:
    def __init__(self, N_ph=10, N_th=10, N_t=10):
        self.N_ph = N_ph
        self.N_th = N_th
        self.N_t = N_t

        self._create_angles()
        self._create_vectors()
        self._create_times()

    def redshift(self, pa, gw):
        L, P, h = pa.pulsar_dist, pa.pulsar_array, gw.h
        N_p = P.shape[0]
        Z = np.zeros((N_p,self.N_t))
        t = np.zeros((N_p,self.N_t))

        for k in tqdm(range(N_p)):
            p, lp = P[k], L[k]
            alpha = lp * (1 + t_sum('lij,l->ij', self.Omega, p))
            alpha = np.tile(alpha[None,:,:], (self.N_t, 1, 1))

            tau = np.tile(self.t[:, None, None], (1, self.N_th, self.N_ph))
            delta_h = h(tau) - h(tau - alpha)
            F = 1/2 * t_sum('lm,lmij->ij',t_sum('l,m->lm', p, p), self.e) / (1 + t_sum('lij,l->ij', self.Omega, p))
            F = np.tile(F[None,:,:], (self.N_t, 1, 1))
            dOmega = np.tile(self.dOmega[None,:,:], (self.N_t, 1, 1))
            Z[k,:] = np.sum(np.real(delta_h * np.conjugate(F)) * dOmega, axis=(1,2))
            t[k,:] = self.t
        
        return t, Z

    def HD_exp(self, pa, gw):
        p = pa.pulsar_array
        N_p = pa.pulsar_array.shape[0]
        t, Z = self.redshift(pa, gw)

        i, j = np.arange(0,N_p), np.arange(0,N_p)
        i, j = np.meshgrid(i, j, indexing='xy')
        mask = i>j

        Z1, Z2 = Z[i][mask], Z[j][mask]
        p1, p2 = p[i][mask], p[j][mask]
        tau = t[j][mask]
        delta_12 = np.ones(N_p)[i][mask]
        
        rho_12 = 1/self.T * np.sum(Z1 * Z2, axis=1) * self.dt
        h2 = np.mean(np.abs(gw.h(tau))**2)/2
        mu_12 = rho_12/(h2 * (1 + delta_12))
        gamma = arccos(t_sum('kl,kl->k',p1,p2))

        gamma, mu_12 = np.sort(gamma), mu_12[np.argsort(gamma)]
        
        gamma_u = np.linspace(0+EPS, PI-EPS, 100)
        indices = np.digitize(gamma, gamma_u)
        mu_u = np.array([mu_12[indices == i].mean() for i in range(0, len(gamma_u))])
        
        mu_12 = mu_12 / mu_u[1] * 1/3
        mu_u = mu_u / mu_u[1] * 1/3

        return gamma_u, mu_u, gamma, mu_12

    def HD_curve(self, pa):
        p = pa.pulsar_array
        Np = p.shape[0]
        mu = []
        gamma = []

        i, j = np.arange(0,Np),  np.arange(0,Np)
        i, j = np.meshgrid(i, j, indexing='xy')
        mask = i>j

        p1, p2 = p[i][mask], p[j][mask]

        F_1 = 1/2 * t_sum('lmk,lmij->kij',t_sum('kl,km->lmk', p1, p1), self.e) / (1 + t_sum('lij,kl->kij', self.Omega, p1))
        F_2 = 1/2 * t_sum('lmk,lmij->kij',t_sum('kl,km->lmk', p2, p2), self.e) / (1 + t_sum('lij,kl->kij', self.Omega, p2))

        gamma = arccos(t_sum('kl,kl->k',p1,p2))
        
        mu = 1/(4*PI) * np.real(np.sum(F_1 * np.conjugate(F_2) * self.dOmega, axis=(1,2)))

        return gamma, mu

    def plot_HD_curve(self, pa=None, gw=None, key='theory'):
        gamma, Gamma = None, None
        gamma_12, Gamma_12  = None, None
        if key=='theory':
            gamma, Gamma = self.HD_curve(pa)

        elif key=='exp':
            gamma, Gamma, gamma_12, Gamma_12 = self.HD_exp(pa, gw)

        gamma_0 = np.linspace(0 + EPS, PI, 1000)
        Gamma_0 = mu_0(gamma_0)

        plt.grid(True)
        plt.scatter(gamma_12 * 180/PI, Gamma_12, c='blue', linewidths=1, label='PTA')
        plt.plot(gamma * 180/PI, Gamma, color='red', label='exp HD')
        plt.plot(gamma_0 * 180/PI, Gamma_0, color='black', label='theory HD')

        plt.title("HD curve")
        plt.xlabel("$\gamma$, deg")
        plt.ylabel("$\Gamma(\gamma)$")
        plt.legend()
        plt.show()

    def _create_angles(self):
        dphi, dtheta = 2*PI/(self.N_ph -1), PI/(self.N_th - 1)
        phi = np.linspace(dphi/2, 2*PI - dphi/2, self.N_ph)
        theta = np.linspace(dtheta/2, PI - dtheta/2, self.N_th)

        self.phi, self.theta = np.meshgrid(phi, theta, indexing='xy')
        self.dphi = np.full(self.phi.shape, dphi)
        self.dtheta = np.full(self.theta.shape, dtheta)

    def _create_vectors(self):
        self.Omega = np.array([-sin(self.theta)*cos(self.phi), 
                               -sin(self.theta)*sin(self.phi), 
                               -cos(self.theta)])

        self.dOmega = self.dphi * np.sin(self.theta) * self.dtheta

        self.m = np.array([sin(self.phi), 
                          -cos(self.phi),
                          zeros(self.phi.shape)])

        self.n = np.array([-cos(self.theta)*cos(self.phi), 
                           -cos(self.theta)*sin(self.phi), 
                           sin(self.theta)])

        self.e_p = t_sum('ijk,ljk->iljk', self.m, self.m) - t_sum('ijk,ljk->iljk', self.n, self.n) 
        self.e_c = t_sum('ijk,ljk->iljk', self.m, self.n) + t_sum('ijk,ljk->iljk', self.n, self.m) 
        
        self.e = self.e_p + 1j * self.e_c
    
    def _create_times(self, T=10*year):
        dt = T/(self.N_t - 1)
        t = np.linspace(dt/2, 1 - dt/2, self.N_t)
        
        # self.t = np.tile(t[:, None, None], (1, *self.phi.shape))
        self.T = T
        self.t = t
        self.dt = dt