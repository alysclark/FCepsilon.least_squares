from collections import OrderedDict
import numpy as np
import math

import OpenCOR as oc


times = np.array([0, 240, 480, 960, 1920, 3840])
pFC = np.array([0.0, 0.0189, 0.0208, 0.0646, 0.0495, 0.0645])
pSyk = np.array([0.0, 0.0143, 0.0255, 0.0303, 0.0242, 0.0202])

class Simulation(object):
    def __init__(self):
        self.simulation = oc.simulation()
        self.simulation.data().setStartingPoint(0)
        self.simulation.data().setEndingPoint(3840)
        self.simulation.data().setPointInterval(1)
        self.constants = self.simulation.data().constants()
        self.model_constants = OrderedDict({k: self.constants[k]
                                            for k in self.constants.keys()})

    def run_once(self, c, v):
        self.simulation.resetParameters()
        self.constants[c] = v
        self.simulation.run()
        return (self.simulation.results().points().values(),
                self.simulation.results().states()['FCepsilonRI/pSyk'].values())

    def run(self, c, scale=2.0):
        self.simulation.clearResults()
        v = self.model_constants[c]
        base = self.run_once(c, v)[1][times]
        divergence = 0.0
        for s in [1.0/scale, scale]:
            trial = self.run_once(c, s*v)[1][times]
            divergence += math.sqrt(np.sum((base - trial)**2))
        return divergence

    def test_run(self):
        variation = OrderedDict()
        for c in self.model_constants.keys():
            variation[c] = self.run(c)
        return variation

    def test(self, c, v):
        trial = self.run_once(c, s*v)[1][times]
        return math.sqrt(np.sum((pSyk - trial)**2))

s = Simulation()

v = s.test_run()
#print(v)
print({ k:d  for k, d in v.items() if d > 0.001 })



