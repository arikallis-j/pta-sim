import pta

pa = pta.PulsarArray(100, key='angle')
gw = pta.GravitationalWave(key='ipow',param=(1, -5))
grid = pta.Grid(190,100,100,100)
grid.plot_HD_curve(pa, gw, key='obs')