import typer

from .classes import PulsarArray, GravitationalWave, Grid

app = typer.Typer()

@app.command()
def build(kgw: str = 'ipoint', np: int = 20, n_ph: int = 10, n_th: int = 10, nt: int = 10, nf: int = 10, nb: int = 1):
    gw = GravitationalWave(key=kgw)
    pa = PulsarArray(np)
    grid = Grid(n_ph,n_th,nt,nf,nb)
    grid.plot_HD_curve(pa, gw, key='obs', show=False)


@app.callback(invoke_without_command=True)
def context(ctx: typer.Context):
    """
    CLI running
    """
    if ctx.invoked_subcommand is None:
        print("Running a CLI...")