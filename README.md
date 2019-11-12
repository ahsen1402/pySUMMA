SUMMA Ensemble Package
===============

This package provides an implementation of aggregation strategies for binary classification:

- Strategy for Multiple Method Aggregation (SUMMA) by [Ahsen et al.](https://arxiv.org/abs/1802.04684) [1],
- Wisdom of Crowds (WOC) aggregation strategy by [Marbach et al.](https://www.nature.com/articles/nmeth.2016) [2], and
- Spectral Metalearner (SML) by [Parisi et al.](https://www.pnas.org/content/111/4/1253.short) [3].

using the Python programming language.  Our implementations use the method first described in [Jaffe et al.](http://proceedings.mlr.press/v38/jaffe15.pdf) [4] to infer the singular value associated with elements of the third central moment tensor.

In addition, the package includes helpful tools for:

- Computing the empirical ROC and Balanced Accuracy from classifier predictions,
- generating simulation data representative of the binary predictions [-1, 1] by base classifiers,
- generating simulation data representative of the rank predictions by binary classifiers, and
- a plotting tool.

Dependencies
-----------
To use this package you will need:

- Python (>=3.5)
- numpy (>=1.14.0)
- scipy (>=1.0.0)
- matplotlib (>=2.2.0)

Installation
------------

### Installation from a local copy

The pySUMMA package can be installed from a local copy of the pySUMMA package source code using setup tools.  First, make a directory in which you would like the pySUMMA package to be located, let's call this directory `myProject`.  Then move the unzipped pySUMMA package into the `myProject` directory.  Next, open a terminal window navigate to the `myProject` directory.  We assume that a python virtual environment by the name of `SummaEnv` has been activated.

First install the required packages using `pip`
```bash
(SummaEnv) $ pip install -r pySUMMA/requirements.txt
```

followed by installing the `pySUMMA` package

```bash
(SummaEnv) $ pip install --no-index pySUMMA/
```

Note, that the `--no-index` flag ensures that pip installs the local `pySUMMA` package as opposed to searching and installing from the online Python Package Index.

### Installation from GitHub

If the dependencies are installed and the corresponding virtual environment `SummaEnv` is active then `pySUMMA` can be installed from this GitHub repository.  One way to install `pySUMMA` from the master branch is to,

```bash
(SummaEnv) pip install git+https://github.com/ahsen1402/pySUMMA.git@master
```

at the command line.

SUMMA
-----

See Ahsen et al. [1] for details on SUMMA and Jaffe et al. [4] about the decomposition of the third central moment tensor.

### Example

We begin by importing the SUMMA and WOC classifiers; and simulation, plotting, and evaluation tools.

```python
import matplotlib.pyplot as plt
from pySUMMA.simulate import Rank
from pySUMMA.utilities import Roc
from pySUMMA import Summa, RankWoc
from pySUMMA import plot
```
The `Rank` class is a simulation tool that takes the number of classifiers, samples, and positive class samples as input.  Here, base classifier performances are sampled at random between the default AUC values of [0.45, 0.8].  Users can change the AUC limits by using the keyword argument `auc_lims`.  An (nClassifier, nSample) `ndarray` of simulated rank predictions are accessible by using the `data` attribute, while the (nSample) `ndarray` of sample class labels in the `labels` attribute.

```python
nClassifiers = 15
nSamples = 2500
prevalence = 0.3
nPositiveSamples = int(prevalence * nSamples)

sim = Rank(nClassifiers, nSamples, nPositiveSamples)
sim.sim()
```

Next apply the WOC and SUMMA classifiers using the `RankWoc` and `Summa` classes respectively.  To infer SUMMA model parameters, use the `Summa` class method `fit`.

```python
clw = RankWoc()

cls = Summa()
cls.fit(sim.data)
```

To infer the positive class prevalence use the `get_prevalence` method of the `Summa` class.

```python
print(("Positive Class Prevalence\n"
       "Inferred:\t{:0.3f}\n"
       "True:\t\t{:0.3f}").format(cls.get_prevalence(), prevalence))
```

To evaluate the performance of the classifiers use the `Roc` class, and the `get_scores` method for each classifier.

```python
summa_roc = Roc(cls.get_scores(sim.data), sim.labels)
woc_roc = Roc(clw.get_scores(sim.data), sim.labels)
```

The ROC curve can be plotted by accessing the true positive rate `tpr` and false positive rate `fpr` attributes of the `Roc` class.

```python
plt.figure()
plt.plot(summa_roc.fpr, summa_roc.tpr, '-', label="SUMMA")
plt.plot(woc_roc.fpr, woc_roc.tpr, ':', label="WOC")
plt.plot([0, 1], [0, 1], ":", label="Random", color="black", alpha=0.25)
plt.legend(loc=4)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()
```

The AUC of each classifier can be found using the `auc` attribute of the `Roc` class, and plotted using the `plot.performance` function.  The best individual classifier can be selected from the ndarray of base classifier AUC values estimated by the `get_empirical_auc` method of the simulation `Rank` class.  Lastly, the `plot.performance` function displays the correlation between the empirical and SUMMA inferred (`Summa` class `get_auc` method) base classifier AUC values.

```python
clAUC = {"SUMMA" : summa_roc.auc,
         "WOC" : woc_roc.auc,
         "Best Ind" : sim.get_empirical_auc().max()}

plot.performance(sim.get_empirical_auc(),		
                 cls.get_auc(),   	   	
                 clAUC)

```

SML
---

See Parisi et al. [3] for details on SML and Jaffe et al. [4] about the decomposition of the third central moment tensor.

### Example

We begin by importing the SML and WOC classifiers; and simulation, plotting, and evaluation tools.

```python
from pySUMMA.simulate import Binary
from pySUMMA.utilities import Ba
from pySUMMA import Sml, BinaryWoc
from pySUMMA import plot
```

The `Binary` class is a simulation tool that takes the number of classifiers, samples, and positive class samples as input.  Here, base classifier performances are sampled at random between the default Balanced Accuracy (BA) values of [0.35, 0.9].  Users can change the BA limits by using the keyword argument `ba_lims`.  An (nClassifier, nSample) `ndarray` of simulated binary predictions [-1, 1] are accessible by using the `data` attribute, while the (nSample) `ndarray` of sample class labels in the `labels` attribute.

```python
nClassifiers = 15
nSamples = 2500
nPositiveSamples = int(0.3 * nSamples)

sim = Binary(nClassifiers, nSamples, nPositiveSamples)
sim.sim()
```

Next apply the WOC and SML classifiers using the `BinaryWoc` and `Sml` classes, respectively.  To infer SML model parameters, use the `Sml` class method `fit`.

```python
clw = BinaryWoc()

cls = Sml()
cls.fit(sim.data)
```

To evaluate the performance of the classifiers use the `Ba` class, and the `get_inference` method for each classifier.  The BA of each classifier can be found using the `ba` attribute of the `Ba` class, and plotted using the `plot.performance` function while specifying the metric as "BA".  The best individual classifier can be selected from the ndarray of base classifier BA values computed by the `get_ba` method of the simulation `Binary` class.  Lastly, the `plot.performance` function displays the correlation between the empirical and SML inferred (`Sml` class `get_ba` method) base classifier BA values.

```python
sml_ba = Ba(cls.get_inference(sim.data), sim.labels)
woc_ba = Ba(clw.get_inference(sim.data), sim.labels)

clBA = {"SML": sml_ba.ba,
        "WOC": woc_ba.ba,
        "Best Ind" : sim.get_ba().max()}

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
