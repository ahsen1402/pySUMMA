

# ================================
# LOAD EXAMPLE DEPENDENT PACKAGE
# ================================

from argparse import ArgumentParser

# ================================
# LOAD pySUMMA PACKAGES
# ================================

from pySUMMA.simulate import rank
from pySUMMA.utilities import roc
from pySUMMA import summa, woc
from pySUMMA import plot


# ================================
# RUN SIMULATATION AND EXAMPLES
# ================================

def main(N, M, prevalence, inferPrevalence, maxiter, savename):
    # ================================
    # SIMULATE
    print("==========\n{}\n==========".format("Simulate"))
    print("{} Samples".format(N))
    print("{} Methods".format(M))
    print("{} Is the fraction of samples from the positive class".format(prevalence))
    if inferPrevalence:
        print("Infer positive class prevalence")
    else:
        print("Assume prevalence of the positive class samples is known")

    sim = rank(M, N, int(prevalence * N))
    sim.sim()

    # ================================
    # RUN CLASSIFIERS

    # SUMMA classifier

    # Note if prevalence is given as an argument the
    # third central moment tensor will not be analyzed
    # e.g. summa(prevalence = 0.3)

    print("==========\n{}\n==========".format("Train Classifier"))
    if inferPrevalence:
        cls = summa(max_iter=maxiter)
        cls.fit(sim.data)
    else:
        cls = summa(max_iter=maxiter, prevalence=0.3)
        cls.fit(sim.data)

    # Wisdom of Crowd Ensemble
    clw = woc()

    # ================================
    # STATS AND PLOT

    print("{}\n==========".format("Computing AUC"))
    clAUC = {'SUMMA':roc(cls.get_scores(sim.data), sim.labels).auc,
             'WOC':roc(clw.get_scores(sim.data), sim.labels).auc,
             "Best Ind" : sim.get_auc().max()}

    print("{}\n==========".format("Plotting"))
    plot.performance(sim.get_auc(),
                     cls.get_auc(),
                     clAUC,
                     savename=savename)
    return 0


# ===================================
# PARSER TOOLS
# ===================================

class customRange:
    def __init__(self, l, u):
        self.lowerBound = l
        self.upperBound = u

    def __eq__(self, x):
        if (x < self.lowerBound) | (x > self.upperBound):
            return False
        else:
            return True

# ===================================
# PARSE INPUTS AND CALL MAIN
# ===================================

if __name__ == "__main__":
    # ====================
    # Define Arguments, default values, and restrictions
    argparse = ArgumentParser()
    argparse.add_argument("--samples",
                          default=1500,
                          dest = "N",
                          type=int,
                          choices=[customRange(100, 10001)])
    argparse.add_argument("--methods",
                          type=int,
                          default=10,
                          dest = "M",
                          choices=[customRange(3, 101)])
    argparse.add_argument("--prevalence",
                          type=float,
                          default=0.3,
                          choices=[customRange(0.1, 0.9)],
                          dest="prevalence")
    argparse.add_argument("-i", "--inferPrevalence",
                          dest="inferPrevalence",
                          action="store_true")
    argparse.add_argument("--maxiter",
                          type=int,
                          default=500,
                          dest="maxiter",
                          choices=[customRange(100, 10000)])
    argparse.add_argument("--savename", type=str, default="results.pdf",
                          dest="savename")


    # ====================
    # parse the arguments
    args = argparse.parse_args()

    # ====================
    # run simulation and perform analysis
    main(args.N, args.M,
         args.prevalence,
         args.inferPrevalence,
         args.maxiter,
         args.savename)
