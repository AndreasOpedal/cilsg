# Collaborative Filtering - Unstable Geniuses

We explain how to the submission directory is organized, how to run the algorithms (and in which modes), and the role of notebooks.

## Table of Contents

1. [Directory structure](#directory)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Algorithms](#algorithms)
5. [Modes](#modes)
6. [Execution](#execution)
7. [Notebooks](#notebooks)

## Directory structure

This submission directory is structured as follows: the `data/` directory contains the training data and the missing entries for the submission. The `src/` data contains the implementation of most algorithms (apart from the auto-encoder). Last, the `notebook/` directory contains notebooks for each algorithm.

## Requirements

The following packages are needed (Python 3.8.0):

+ numpy=1.17.4
+ pandas=0.25.3
+ scikit-surprise=1.1.0
+ Cython=0.29.16
+ matplotlib=3.1.2
+ tqdm=4.45.0

## Setup

Some algorithms (`.pyx` extension) are implemented using Cython to speed-up the computations. In order to run these algorithms, first execute the following commands:

```
cd src/
python3 setup.py build_ext --inplace
```

## Algorithms

We implemented the following algorithms:

+ *Mean* (src/baseline.py): this simple algorithm predicts the global sample mean for each missing entry
+ *SVD* (src/baseline.py): compute the SVD of the training matrix. The training matrix can be imputed with the sample mean, sample median, or with 0s
+ *ALS* (src/baseline.py): an implementation of alternating least squares
+ *SGDPP2* (src/factorization.pyx): an variation of the SVD++ algorithm
+ *pLSA* (src/plsa.pyx)
+ *SVDthr* (src/thresholding.py): an implementation of SVD thresholding
+ *VAE* (??): an implementation of a variational auto-encoder
+ *Ensemble* (??): an ensemble method where the result of different algorithms is averaged

We organize these algorithms (with the exception of *VAE* and *Ensemble*) in the `src/source.py` file. Two dictionaries are used to organize algorithm classes and instances. The `algo_classes` dictionary maps algorithms' names to algorithms Python classes, e.g. `algo_classes['Mean'] = Mean`. The `instances` dictionary maps algorithms' names to a dictionary of that algorithm's instances. Instances are mapped by a unique number. For example, the default version of the *Mean* algorithm is mapped as `instances['Mean'][1] = Mean()`.
How to add a new instance of an algorithm? Say that we want to run *SVD* with a different number of factors: then we add to the file the line `instances['SVD'][2] = SVD(n_factors=2)`.

## Modes

Algorithms can be executed in the following modes:

+ *cv*: performs cross-validation on the selected algorithm instance and computes the validation RMSE on a 0.25 split
+ *kfold*: performs k-fold cross-validation on the selected algorithm instance. The user can decide on the number of folds (10 by default)
+ *grid*: performs grid search on the selected algorithm class using the parameter grid in the `src/source.py` file
+ *random_search*: performs random search on the selected algorithm class using the parameter distribution grid in the `src/source.py` file
+ *dump*: trains the selected algorithm instance on the whole dataset and writes the predictions in a `.csv` file

## Execution

To execute an algorithm with a given mode, run the following commands:

```
cd src/
python3 main.py mode algorithm
```

The following options are available:

+ -h: help message
+ --model_num (int): the number of the algorithm instance. By default 1 (corresponding to our best model configurations)
+ --k (int): the number of folds for the k-fold cross-validation. By default 10
+ --n_iters (int): the number of iterations for the randomized search. By default 10
+ --verbose (bool): whether the algorithm should be verbose. By default False
+ --seed (int): the random seed. By default 0

## Notebooks

Notebooks are meant to give the user a better understanding of the code implemented in `src/`. Their goal is to provide the mathematical background, clarify the code, and creating plots to better illustrate how the algorithm learns the weights.
Notebooks are not meant however to train the full model or to perform cross-validation, as it would be quite time-consuming. The exception to this rule is the notebook for the auto-encoder.
