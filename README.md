SUMMA Ensemble Package
===============

This package provides an implementation of aggregation strategies for binary classification:

- Strategy for Multiple Method Aggregation (SUMMA) by [Ahsen et al.](https://arxiv.org/abs/1802.04684) [1],
- Wisdom of the Crowds (WOC) aggregation strategy by [Marbach et al.](https://www.nature.com/articles/nmeth.2016) [2], and
- Spectral Metalearner (SML) by [Parisi et al.](https://www.pnas.org/content/111/4/1253.short) [3].

where our implementations use [Jaffe et al.](http://proceedings.mlr.press/v38/jaffe15.pdf) [4] to infer the singular value associated with elements of the third central moment tensor.

and helpful tools for:

- Computing the empirical ROC and Balanced Accuracy from classifier predictions,
- generating simulation data representative of the binary predictions [-1, 1] by base classifiers,
- generating simulation data representative of the rank predictions by binary classifiers, and
- plotting tools.

Dependencies
-----------
To use this package you will need:

- Python (=3.7)
- numpy (=1.16.2)
- scipy (=1.2.1)
- matplotlib (=3.0.3)


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

If the dependencies are installed and the corresponding virtual environment `pySummaEnv` is active then `pySUMMA` can be installed from the GitHub repository.  One way to install `pySUMMA` from the master branch is to,

```bash
(pySummaEnv) pip install git+https://github.com/ahsen1402/pySUMMA.git@master
```

at the command line.

That is it!

SUMMA
-----

**TODO : Need to add description**

### Example

Testing our SUMMA implementation can be done at the command line or through an interactive Python prompt.  To run at the command line use `examples/summaSimulationExample.py`.  The file `examples/summaSimulationExample.sh` provides serval examples of running at the command line, and itself can be run.

Below you will find an example for running in an interactive Python prompt.
```python
# load modules

from pySUMMA.simulate import Rank
from pySUMMA.utilities import Roc
from pySUMMA import Summa, RankWoc
from pySUMMA import plot


# Set simulation parameters
nClassifiers = 15
nSamples = 2500
nPositiveSamples = int(0.3 * nSamples)

# simulate data set
sim = Rank(nClassifiers, nSamples, nPositiveSamples)
sim.sim()

# apply SUMMA to the simulation data
cls = Summa()
# note that sim.data is an nClassifier by nSample ndarray
cls.fit(sim.data)

# wisdom of the crowd classifier
clw = RankWoc()

# compute the AUC for SUMMA and WOC ensembles, and retrieve the AUC of the best individual classifier
clAUC = {"SUMMA" : Roc(cls.get_scores(sim.data),
                       sim.labels).auc,
         "WOC" : Roc(clw.get_scores(sim.data),
                     sim.labels).auc,
         "Best Ind" : sim.get_empirical_auc().max()}

# plot the inferred AUC of base classifiers vs the true AUC
# and plot the AUC of each ensemble classifier
plot.performance(sim.get_empirical_auc(),		
                 cls.get_auc(),   	   	
                 clAUC)

```



SML
---

Parisi et al. [3] developed the Spectral Meta Learner (SML) as an
unsupervised ensemble method.  Here, binary predictions [-1, 1] of an ensemble of M
base classifiers are aggregated by a weighted sum.  The weight (w_i) of the i th base
classifier is proportional to its performance, which when written in latex notation is

w_i \propto 2\pi_i - 1

where \pi_i is the balanced accuracy of the the i th base classifier.
They found that these weights can be estimated from data, without class labels,
when the predictions of the base classifiers are conditionally independent.  
Under the assumption of conditional independence the authors found that
the off-diagonal elements of covariance matrix of base classifier predictions
are that of a rank one matrix.  The corresponding i th eigenvector
element of this rank one matrix, written in latex notation, is

v_i = (2\pi_i - 1) / ||2\pi -1||

where ||2\pi-1|| is the vector norm.  Consequently, the authors set each weight w_i
equal to v_i for each i = 1, 2, 3, ..., M.

Jaffe et al [4] extended the work of Parisi et al. [3] by studying the properties
of the third central moment tensor (T) of an ensemble 3rd order conditionally
independent base classifier predictions.  Like the covariance matrix, the elements T_{ijk}
such that i \neq j \neq k are those of a rank one tensor.  Using the corresponding
singular value, they shou



### Example

Testing our SML implementation can be done at the command line or through an interactive Python prompt.  To run at the command line use `examples/smlSimulationExample.py`.  The file `examples/smlSimulationExample.sh` provides serval examples of running at the command line, and itself can be run.

Below you will find an example for running in an interactive Python prompt.
```python
# load modules

from pySUMMA.simulate import Binary
from pySUMMA.utilities import Ba
from pySUMMA import Sml, BinaryWoc
from pySUMMA import plot


# Set simulation parameters
nClassifiers = 15
nSamples = 2500
nPositiveSamples = int(0.3 * nSamples)

# simulate data set
sim = Binary(nClassifiers, nSamples, nPositiveSamples)
sim.sim()

# apply SML to the simulation data
cls = Sml()
# note that sim.data is an nClassifier by nSample ndarray
cls.fit(sim.data)

clw = BinaryWoc()

# compute the BA for SML and retrieve the BA of the best individual classifier
clBA = {"SML": Ba(cls.get_inference(sim.data),
                   sim.labels).ba,
        "WOC": Ba(clw.get_inference(sim.data),
                  sim.labels).ba,
        "Best Ind" : sim.get_ba().max()}

# plot the inferred AUC of base classifiers vs the true AUC
# and plot the AUC of each ensemble classifier
plot.performance(sim.get_ba(),
                 cls.get_ba(),
                 clBA,
                 metric="BA")

```




Cite Us
-------

Please don't forget to cite our manuscript:

```
  @article{summa_pkg,
    author  = {Mehmet Eren Ahsen and Robert Vogel and Gustavo Stolovitzky},
    title = {SUMMA: An R/Python Package for Unsupervised Ensemble Learning},
  }
```

References
----------

1. Mehmet Eren Ahsen, Robert Vogel, and Gustavo Stolovitzky. Unsupervised evaluation and weighted aggregation of ranked predictions. *arXiv preprint* arXiv:1802.04684, 2018.
2. Daniel Marbach, James~C Costello, Robert Küffner, Nicole M Vega, Robert J Prill, Diogo M Camacho, Kyle R Allison, Andrej Aderhold, Richard Bonneau, Yukun Chen, et al. Wisdom of crowds for robust gene network inference. *Nature methods*, 9(8):796, 2012.
3. Fabio Parisi, Francesco Strino, Boaz Nadler, and Yuval Kluger. Ranking and combining multiple predictors without labeled data. *Proceedings of the National Academy of Sciences*, 111(4):1253--1258, 2014.
4. Ariel Jaffe, Boaz Nadler, and Yuval Kluger. Estimating the accuracies of multiple classifiers without labeled data. *Artificial Intelligence and Statistics*, pages 407--415, 2015.
