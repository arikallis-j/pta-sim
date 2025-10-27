import numpy as np
import matplotlib.pyplot as plt

from .consts import PI, EPS
from .funcs import t_sum, sin, cos, exp, sqrt, arccos, zeros, mu_0
from tqdm import tqdm

year = 3.1536e7 
f_yr = 1/year
pc = 3.26*year
kpc = 1e3 * pc

class GravitationalWave:
    def __init__(self, key='ipoint', param=(1, 100e-9)):
        self.key = key
        self.param = param

    def generate_wave(self, f, df):
        if self.key == 'ipoint':
            A, f_0 = self.param
            H = A * self._delta_func(f_0, f, df)
        if self.key == 'ipow':
            A, alpha = self.param
            H = A * self._pow(alpha, f, df)
        return H

    def _delta_func(self, f_0, f, df):
        d = np.zeros(f.shape)
        for k in range(f.shape[0]-1):
            if f[k] <= f_0 and f[k+1] > f_0:
                d[k] = 1/df[k]
        return d

    def _pow(self, alpha, f, df):
        d = (f/f_yr)**(alpha)
        norm = 1 / np.sum(d * df)
        return norm * d

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
                lp = (k+1) * kpc
                vp = lp * p
                pulsar_array.append(p)
                pulsar_dist.append(lp)
                pulsar_coord.append(vp)

        self.pulsar_array = np.array(pulsar_array)
        self.pulsar_dist = np.array(pulsar_dist)
        self.pulsar_coord = np.array(pulsar_coord)

class Grid:
    def __init__(self, pa, gw, N_ph=10, N_th=10, N_t=10, N_f=10):
        self.N_ph = N_ph
        self.N_th = N_th
        self.N_t = N_t
        self.N_f = N_f

        self._create_angles()
        self._create_vectors()
        self._create_times()

    def _generate_redshift(self, pa, gw):
        P, L = pa.pulsar_array, pa.pulsar_dist
        self.N_p = P.shape[0]
        self.H = gw.generate_wave(self.f, self.df)
        self.h_tilda = sqrt(self.H)

        h_tilda = np.tile(self.h_tilda[None, :], (self.N_t, 1))
        f = np.tile(self.f[None, :], (self.N_t, 1))
        df = np.tile(self.df[None, :], (self.N_t, 1))
        t = np.tile(self.t[:, None],(1, self.N_f))
        h = np.sum(exp(1j * 2 * PI * f * t) * h_tilda * df, axis=1)
        
        Z = np.zeros((self.N_p, self.N_t))

        for k in tqdm(range(self.N_p)):
            p, lp = P[k], L[k]
            beta = lp * (1 + t_sum('lij,l->ij', self.Omega, p))
            
            beta = np.tile(beta[None, None, :, :], (self.N_t, self.N_f, 1, 1))
            t = np.tile(self.t[:, None, None, None],(1, self.N_f, self.N_th, self.N_ph))
            t_p = t - beta

            f = np.tile(self.f[None, :, None, None], (self.N_t, 1, self.N_th, self.N_ph))
            df = np.tile(self.df[None, :, None, None], (self.N_t, 1, self.N_th, self.N_ph))
            h_tilda = np.tile(self.h_tilda[None, :, None, None], (self.N_t, 1, self.N_th, self.N_ph))
            h_p = np.sum(exp(1j * 2 * PI * f * t_p) * h_tilda * df, axis=1)
            
            h_e = np.tile(h[:, None, None], (1, self.N_th, self.N_ph))
            delta_h = h_e - h_p
            
            F = 1/2 * t_sum('lm,lmij->ij',t_sum('l,m->lm', p, p), self.e) / (1 + t_sum('lij,l->ij', self.Omega, p))
            F = np.tile(F[None, :, :], (self.N_t, 1, 1))
            dOmega = np.tile(self.dOmega[None, :, :], (self.N_t, 1, 1))
            Z[k] = np.real(np.sum(delta_h * F * dOmega, axis=(1,2)))
        
        self.Z = Z

    def HD_obs(self, pa, gw):
        self._generate_redshift(pa, gw)
        p, z = pa.pulsar_array, self.Z
        h2 = np.sum(self.H * self.df)

        i, j = np.arange(0, self.N_p),  np.arange(0, self.N_p)
        i, j = np.meshgrid(i, j, indexing='xy')
        mask = i>j
        i, j = i[mask], j[mask]

        p1, p2 = p[i], p[j]
        z1, z2 = z[i], z[j]

        r12 = np.sum(z1 * z2 * self.dt, axis=1)
        mu = r12 / (4*PI*h2)

        gamma = arccos(t_sum('kl,kl->k',p1,p2))

        gamma_m = np.linspace(0, PI, 100)
        mu_m = np.zeros(gamma_m.shape)
        N_m = np.zeros(gamma_m.shape)

        for k in range(gamma.shape[0]):
            mu_k, gamma_k = mu[k], gamma[k]
            for i in range(gamma_m.shape[0]-1):
                if gamma_m[i] <=  gamma_k and gamma_k < gamma_m[i+1]:
                    mu_m[i] += mu_k
                    N_m[i] += 1
        mu_m[N_m!=0] = mu_m[N_m!=0] / N_m[N_m!=0]
        mu_m[N_m==0] = None
        print(mu_m)


        return gamma, mu, gamma_m, mu_m

    def HD_curve(self, pa, gw):
        p = pa.pulsar_array
        Np = p.shape[0]

        i, j = np.arange(0,Np),  np.arange(0,Np)
        i, j = np.meshgrid(i, j, indexing='xy')
        mask = i>j
        i, j = i[mask], j[mask]

        p1, p2 = p[i], p[j]

        F_1 = 1/2 * t_sum('lmk,lmij->kij',t_sum('kl,km->lmk', p1, p1), self.e) / (1 + t_sum('lij,kl->kij', self.Omega, p1))
        F_2 = 1/2 * t_sum('lmk,lmij->kij',t_sum('kl,km->lmk', p2, p2), self.e) / (1 + t_sum('lij,kl->kij', self.Omega, p2))

        gamma = arccos(t_sum('kl,kl->k',p1,p2))
        
        mu = 1/(4*PI) * np.real(np.sum(F_1 * np.conjugate(F_2) * self.dOmega, axis=(1,2)))

        return gamma, mu

    def plot_HD_curve(self, pa=None, gw=None, key='theory'):
        gamma, Gamma = np.array([]), np.array([])
        gamma_12, Gamma_12  = np.array([]), np.array([])

        if key=='theory':
            gamma_12, Gamma_12 = self.HD_curve(pa, gw)

        if key=='obs':
            gamma_12, Gamma_12, gamma, Gamma = self.HD_obs(pa, gw)

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
    
    def _create_times(self, T=10*year, rng_f=(1e-9,1e-1)):
        dt = T/(self.N_t - 1)
        t = np.linspace(dt/2, T - dt/2, self.N_t)
        
        f1, f2 = rng_f
        log_df = (np.log(f2) - np.log(f1))/(self.N_f - 1)
        log_f = np.linspace(np.log(f1), np.log(f2), self.N_f)
        f, df = exp(log_f), np.zeros(log_f.shape)
        df[:-1:] = f[1::] - f[:-1:]
        df[-1] = exp(np.log(f2) + log_df) - f2


        self.T, self.f1, self.f2 = T, f1, f2
        self.t, self.dt = t, dt
        self.f, self.df = f, df
        
