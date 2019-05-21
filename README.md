SUMMA Ensemble Package
===============

This package provides an implementation of aggregation strategies for binary classification:

- Strategy for Multiple Method Aggregation (SUMMA) by [Ahsen et al.](https://arxiv.org/abs/1802.04684),
- Wisdom of the Crowds (WOC) aggregation strategy, and
- Spectral Metalearner (SML) by [Parisi et al](https://www.pnas.org/content/111/4/1253.short).

and plotting tools in Python.

Dependencies
-----------
To use this package you will need:

- Python (=3.7)
- numpy (=1.16.2)
- scipy (=1.2.1)
- matplotlib (=3.0.3)
- tensorly (=0.4.3)

Note that we use numpy for computation with the tensorly package.


Citing
------

Please don't forget to cite our manuscript:

```
  @article{summa_pkg,
    author  = {Mehmet Eren Ahsen and Robert Vogel and Gustavo Stolovitzky},
    title = {SUMMA: An R/Python Package for Unsupervised Ensemble Learning},
  }
```

Installation
------------

### Installation from a local copy

The pySUMMA package can be installed from a local copy of the pySUMMA package source code using Python setup tools.  First, make a directory in which you would like the pySUMMA package to be located, let's call this directory `myProject`.  Then move the unzipped pySUMMA package into the `myProject` directory.  Next, open a terminal window navigate to the `myProject` directory.  We assume that a python virtual environment by the name of `pySummaEnv` has been activated.


First install the required packages using `pip`
```bash
(pySummaEnv) $ pip install -r pySUMMA/requirements.txt
```

followed by installing the `pySUMMA`

```bash
(pySummaEnv) $ pip install --no-index pySUMMA/
```

Note, that the `--no-index` flag ensures that pip installs the local `pySUMMA` package as opposed to searching and installing from the online Python Package Index.

That is it!



### Installation from GitHub

Need to be completed


Examples
--------

A detailed example that can be run on using the command line can be found in the *examples* directory.

Below is a simple example on how to simulate rank predictions by base classifiers data, apply the SUMMA and WOC classifiers, compute the AUC, and generate a summary plot.

```python
# load modules

>>> from pySUMMA.simulate import rank
>>> from pySUMMA.utilities import roc
>>> from pySUMMA import summa, woc
>>> from pySUMMA import plot


# Set simulation parameters
>>> nClassifiers = 15
>>> nSamples = 2500
>>> nPositiveSamples = int(0.3 * nSamples)

# simulate data set
>>> sim = rank(nClassifiers, nSamples, nPositiveSamples)
>>> sim.sim()

# apply SUMMA to the simulation data
>>> cls = summa()
# note that sim.data is an nClassifier by nSample ndarray
>>> cls.fit(sim.data)

# wisdom of the crowd classifier
>>> clw = woc()

# compute the AUC for SUMMA and WOC ensembles, and retrieve the AUC of the best individual classifier
>>> clAUC = {"SUMMA" : roc(cls.get_scores(sim.data),
                            sim.labels).auc,
             "WOC" : roc(clw.get_scores(sim.data),
                            sim.labels).auc,
             "Best Ind" : sim.get_auc().max()}

# plot the inferred AUC of base classifiers vs the true AUC
# and plot the AUC of each ensemble classifier
>>> plot.performance(sim.get_auc(),		# True base classifier AUC
                      cls.get_auc(),   	   	# inferred base classifier AUC
                      clAUC) 	# ensemble AUC in python dictionary

```
