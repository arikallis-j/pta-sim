import pta


pa = pta.PulsarArray(100)
gw = pta.GravitationalWave()
grid = pta.Grid(100,100,100)

grid.plot_HD_curve(pa, gw, key='exp')