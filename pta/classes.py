import numpy as np
import matplotlib.pyplot as plt

from .consts import PI, EPS
from .funcs import t_sum, sin, cos, arccos, zeros, mu_0

class PulsarArray:
    def __init__(self, N=10, key='angle'):
        self.key = key
        self.N = N

        self._create_pulsar_array()

    def _create_pulsar_array(self):
        if self.key == 'angle':
            pulsar_array = []
            Lp = []
            gamma = np.linspace(0, PI, self.N)
            
            for g in gamma:
                p = np.array([np.sin(g), 0, np.cos(g)])
                lp = 10 # kpc
                pulsar_array.append(p)
                Lp.append(lp)
            
            self.pulsar_array = np.array(pulsar_array)
            self.Lp = np.array(Lp)

class Grid:
    def __init__(self, N_ph=10, N_th=10):
        self.N_ph = N_ph
        self.N_th = N_th

        self._create_angles()
        self._create_vectors()

    def HD_curve(self, p_array):
        Np = p_array.shape[0]
        mu = []
        gamma = []
        for i in range(Np):
            for j in range(Np):
                if i!=j and i<j:
                    p1, p2 = p_array[i], p_array[j]
                    F_1 = 1/2 * t_sum('il,iljk->jk',t_sum('i,l->il', p1, p1), self.e) / (1 + t_sum('ijk,i->jk', self.Omega, p1)) 
                    F_2 = 1/2 * t_sum('il,iljk->jk',t_sum('i,l->il', p2, p2), self.e) / (1 + t_sum('ijk,i->jk', self.Omega, p2))
                    
                    gamma_k = arccos(t_sum('i,i->',p1,p2)/(t_sum('i,i->',p1,p1)*t_sum('i,i->',p2,p2)))
                    mu_k = 1/(4*PI) * np.real(np.sum(F_1 * np.conjugate(F_2) * self.dOmega))

                    gamma.append(gamma_k)
                    mu.append(mu_k)

        gamma = np.array(gamma)    
        mu = np.array(mu)           

        return gamma, mu

    def plot_HD_curve(self, p_array):
        gamma, Gamma = self.HD_curve(p_array)
        gamma_0 = np.linspace(0 + EPS, PI, 1000)
        Gamma_0 = mu_0(gamma_0)
        plt.grid(True)
        plt.plot(gamma_0 * 180/PI, Gamma_0, color='black', label='theory')
        plt.scatter(gamma * 180/PI, Gamma, c='blue', linewidths=1, label='PTA')

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