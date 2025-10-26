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
            Lp = np.linspace(0.001, 30, 10)
            for i in range(10):
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
    def __init__(self, N_ph=10, N_th=10, N_t=10, N_f=10):
        self.N_ph = N_ph
        self.N_th = N_th
        self.N_t = N_t
        self.N_f = N_f

        self._create_angles()
        self._create_vectors()
        self._create_times()

    def redshift(self, pa, gw):
        L, P = pa.pulsar_dist, pa.pulsar_array
        N_p = P.shape[0]
        Z = np.zeros((N_p,self.N_t))
        t = np.zeros((N_p,self.N_t))
        
        self.H = gw.generate_wave(self.f, self.df)
        h_tilde = sqrt(self.H)
        h_tilde = np.tile(h_tilde[:,None, None, None],(1, self.N_t, self.N_th, self.N_ph))
        f = np.tile(self.f[:,None, None, None],(1, self.N_t, self.N_th, self.N_ph))
        df = np.tile(self.df[:,None, None, None],(1, self.N_t, self.N_th, self.N_ph))

        for k in tqdm(range(N_p)):
            p, lp = P[k], L[k]
            alpha = lp * (1 + t_sum('lij,l->ij', self.Omega, p))
            alpha = np.tile(alpha[None, None, :, :], (self.N_f, self.N_t, 1, 1))
            tau = np.tile(self.t[None, :, None, None], (self.N_f, 1, self.N_th, self.N_ph))
            tau_e, tau_p = tau, tau - alpha 
            h_e = np.sum(exp(2j * PI * f * tau_e) * h_tilde * df, axis=0)
            h_p = np.sum(exp(2j * PI * f * tau_p) * h_tilde * df, axis=0)
            delta_h = h_e - h_p
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
        h2 = np.sum(self.H *self.df)

        i, j = np.arange(0,N_p), np.arange(0,N_p)
        i, j = np.meshgrid(i, j, indexing='xy')
        mask = i>j
        i, j = i[mask], j[mask]

        mu_12 = np.zeros(i.shape)
        gamma = np.zeros(i.shape)
        for k in tqdm(range(i.shape[0])):
            ii, jj = i[k], j[k]
            
            Z1, Z2 = Z[ii], Z[jj]
            p1, p2 = p[ii], p[jj]
            
            rho_12 = np.sum(Z1 * Z2) * self.dt
            mu_12[k] = rho_12/(h2 * 4 * PI) 
            gamma[k] = arccos(t_sum('l,l->',p1,p2))

        gamma, mu_12 = np.sort(gamma), mu_12[np.argsort(gamma)]
        
        gamma_u = np.linspace(0+EPS, PI-EPS, 100)
        indices = np.digitize(gamma, gamma_u)
        mu_u = np.array([mu_12[indices == i].mean() for i in range(0, len(gamma_u))])

        return gamma_u, mu_u, gamma, mu_12
    
    def HD_curve(self, pa, gw):
        L, p = pa.pulsar_dist,  pa.pulsar_array
        Np = p.shape[0]
        mu = []
        gamma = []

        i, j = np.arange(0,Np),  np.arange(0,Np)
        i, j = np.meshgrid(i, j, indexing='xy')
        mask = i>j

        p1, p2 = p[i][mask], p[j][mask]
        L1, L2 = L[i][mask], L[j][mask]

        F_1 = 1/2 * t_sum('lmk,lmij->kij',t_sum('kl,km->lmk', p1, p1), self.e) / (1 + t_sum('lij,kl->kij', self.Omega, p1))
        F_2 = 1/2 * t_sum('lmk,lmij->kij',t_sum('kl,km->lmk', p2, p2), self.e) / (1 + t_sum('lij,kl->kij', self.Omega, p2))
        
        # f = np.tile(self.f[:,None,],(1, L1.shape[0]))
        # L1 = np.tile(L1[None,:], (self.N_f, 1))
        # L2 = np.tile(L2[None,:], (self.N_f, 1))

        # alpha_1 = L1*f
        # alpha_2 = L2*f
        # alpha_1 = np.tile(alpha_1[:, :, None, None], (1, 1, self.N_th, self.N_ph))
        # alpha_2 = np.tile(alpha_2[:, :, None, None], (1, 1, self.N_th, self.N_ph))

        # beta_1 = (1 + t_sum('lij,kl->kij', self.Omega, p1))
        # beta_2 = (1 + t_sum('lij,kl->kij', self.Omega, p2))
        # beta_1 = np.tile(beta_1[None, :, :, :], (self.N_f, 1, 1, 1))
        # beta_2 = np.tile(beta_2[None, :, :, :], (self.N_f, 1, 1, 1))

        # df = np.tile(self.df[:, None, None, None],(1, L1.shape[1], self.N_th, self.N_ph))

        # self.H = gw.generate_wave(self.f, self.df)

        # H = np.tile(self.H[:, None, None, None],(1, L1.shape[1], self.N_th, self.N_ph))

        # T_1 =  np.sum(1 - np.exp(-1j  * 2 * PI * alpha_1 * beta_1) * H * df, axis=0)
        # T_2 =  np.sum(1 - np.exp(-1j  * 2 * PI * alpha_2 * beta_2) * H * df, axis=0)

        R_1 = F_1 #* T_1
        R_2 = F_2 #* T_2

        gamma = arccos(t_sum('kl,kl->k',p1,p2))
        
        mu = 1/(4*PI) * np.real(np.sum(R_1 * np.conjugate(R_2) * self.dOmega, axis=(1,2)))

        return gamma, mu

    def plot_HD_curve(self, pa=None, gw=None, key='theory'):
        gamma, Gamma = np.array([]), np.array([])
        gamma_12, Gamma_12  = np.array([]), np.array([])

        if key=='theory':
            gamma_12, Gamma_12 = self.HD_curve(pa, gw)

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
        
