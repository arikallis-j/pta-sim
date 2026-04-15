import numpy as np
import healpy as hp
from .const import *

# Theory
def mu_0(gamma):
    cos_gamma = np.cos(gamma)
    mu = 1/3 - 1/6 * (1 - cos_gamma)/2 + (1 - cos_gamma)/2 * np.log((1 - cos_gamma)/2)
    return mu

def K_12(Omega, p1, p2):
    return 1/4 * (2 * (np.einsum('l,l->', p1, p2) - np.einsum('l,l->', Omega, p1) * np.einsum('l,l->', Omega, p2))**2 / ((1 + np.einsum('l,l->', Omega, p1)) * (1 + np.einsum('l,l->', Omega, p2))) - ((1 - np.einsum('l,l->', Omega, p1)) * (1 - np.einsum('l,l->', Omega, p2))) ) 

# Distributions
def isotropic(skymap, args=None):
    return np.ones(skymap.phi.shape)

def stochastic(skymap, args=43):
    seed_bg = args
    rng_bg = np.random.default_rng(seed=seed_bg)
    eta = (rng_bg.normal(size=skymap.npix) + 1j * rng_bg.normal(size=skymap.npix)) / np.sqrt(2)
    return np.abs(eta)**2

def delta_2d(skymap, args=(PI/2,0)):
    theta0, phi0 = args
    ipix = hp.ang2pix(skymap.nside, theta0, phi0)
    delta = np.zeros(skymap.npix, dtype=float)
    delta[ipix] = 4*PI / skymap.dOmega
    return delta

def gaussian(skymap, args=(PI/2,0,1)):
    theta0, phi0, sigma = args
    Omega_0 = hp.ang2vec(theta0, phi0)
    delta_Omega = np.einsum('li,l->i', skymap.Omega, Omega_0)
    kappa = 1 / sigma**2
    gauss = kappa/(np.sinh(kappa)) * np.exp(kappa * delta_Omega)
    return gauss

def string_2d(skymap, args=(PI/2, 360)):
    theta0, lenght = args
    phi = np.linspace(-lenght/2, +lenght/2 , 1000) * DEG
    theta = np.full_like(phi, PI/2)
    ipixs = hp.ang2pix(skymap.nside, theta, phi)
    delta = np.zeros(skymap.npix, dtype=float)
    delta[ipixs] = 4*PI / skymap.dOmega / len(ipixs)
    return delta

def dipole(skymap, args=None):
    cond = skymap.phi <= PI
    ipix = hp.ang2pix(skymap.nside, skymap.theta[cond], skymap.phi[cond])
    dipole = np.zeros(skymap.npix)
    dipole[ipix] = 4*PI / skymap.dOmega / len(ipix)
    return dipole

def quadrupole(skymap, args=None):
    cond = np.logical_and(skymap.phi <= PI, skymap.theta <= PI/2)
    ipix = hp.ang2pix(skymap.nside, skymap.theta[cond], skymap.phi[cond])
    dipole = np.zeros(skymap.npix)
    dipole[ipix] = 4*PI / skymap.dOmega / len(ipix)
    return dipole

def point(skymap, args=(PI/2,0,100)):
    theta0, phi0, radius_deg = args
    vec_center = hp.ang2vec(theta0, phi0) 
    ipix_disc = hp.query_disc(skymap.nside, vec=vec_center, radius=np.radians(radius_deg))
    point = np.zeros(skymap.npix, dtype=float)
    if len(ipix_disc) == 0:
        ipix = hp.ang2pix(skymap.nside, theta0, phi0, nest=False)
        point[ipix] = 4*PI / skymap.dOmega
    else:
        point[ipix_disc] = 4*PI / skymap.dOmega / len(ipix_disc)
    
    return point

# Spectra
def delta_1d(timeline, args=(10)):
    f0 = args
    idx = np.argmin(np.abs(timeline.f - f0))
    delta = np.zeros_like(timeline.f)
    delta[idx] = 1.0/timeline.df
    return delta

def power(timeline, args=(-5)):
    alpha = args
    d = np.abs(timeline.f + EPS)**(alpha)
    d[timeline.f==0] = 0
    norm = 1 / np.sum(d * timeline.df)
    return norm * d