import pta

pa = pta.PulsarArray(100, key='angle')
gw = pta.GravitationalWave(key='ipoint')#, param=(1, -5))
grid = pta.Grid(100,100,2, 2)

grid.plot_HD_curve(pa, gw, key='theory')