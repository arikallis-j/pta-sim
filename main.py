import pta

pa = pta.PulsarArray(40,key='angle')
gw = pta.GravitationalWave(key='ipoint')#,param=(1, -5))
grid = pta.Grid(30,30,1000,1000)
grid.plot_HD_curve(pa, gw, key='obs')