# ================================
# Test Binary
# ================================

from summa.simulate import binary
from summa.utilities import ba
from summa import sml, woc

from summa import plot

M = 15
N = 2500
N1 = int(0.3 * N)

sim = binary(M, N, N1)
sim.sim()

cls = sml()
cls.fit(sim.data)

clw = woc(rv='binary')

plot.performance(sim.get_ba(), cls.get_ba(),
                {'SML':ba(cls.get_inference(sim.data), sim.labels).ba,
                 'WOC':ba(clw.get_inference(sim.data), sim.labels).ba},
                metric='BA')
