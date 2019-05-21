# ================================
# Test Binary
# ================================

from pySUMMA.simulate import rank
from pySUMMA.utilities import roc
from pySUMMA import summa, woc
from pySUMMA import plot


# Set simulation parameters
nClassifiers = 15
nSamples = 2500
nPositiveSamples = int(0.3 * nSamples)

# simulate data set
sim = rank(nClassifiers, nSamples, nPositiveSamples)
sim.sim()

# apply SUMMA to the simulation data
cls = summa()
# note that sim.data is an nClassifier by nSample ndarray
cls.fit(sim.data)

# wisdom of the crowd classifier
clw = woc()

# compute the AUC for SUMMA and WOC ensembles, and retrieve the AUC of the best individual classifier
clAUC = {"SUMMA" : roc(cls.get_scores(sim.data),
                            sim.labels).auc,
             "WOC" : roc(clw.get_scores(sim.data),
                            sim.labels).auc,
             "Best Ind" : sim.get_auc().max()}

# plot the inferred AUC of base classifiers vs the true AUC
# and plot the AUC of each ensemble classifier
plot.performance(sim.get_auc(),		# True base classifier AUC
                      cls.get_auc(),   	   	# inferred base classifier AUC
                      clAUC) 	# ensemble AUC in python dictionary
