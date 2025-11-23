import pta as pta
import jax.numpy as jnp

grid = pta.Grid(N_ph=100, N_th=100)
gw = pta.GravitationalWave(spec='delta', distr='iso')
pa = pta.PulsarArray(10, distr='randball')
telescope = pta.Telescope(N_t=10)
telescope.observate_redshift(grid, pa, gw)
telescope.plot_HD_curve(grid, pa, gw, key='obs')
