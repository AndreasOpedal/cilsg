# Unstable Geniuses

## Table of Contents

1. [Directory structure](#directory)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Algorithms](#algorithms)
5. [Modes](#modes)
6. [Execution](#execution)
7. [Notebooks](#notebooks)

## Directory structure

This submission directory is structured as follows: the `data/` directory contains the training data and the missing entries for the submission. The `src/` data contains the implementation of most algorithms (apart from the autoencoder). Last, the `notebook/` directory contains notebooks for each algorithm.
Notebooks are meant to be an aid the understanding on how each algorithm works, while the files in the `src/` are meant to be used for quick execution.

## Requirements

The following packages are needed (Python 3.8.0):

* numpy=1.17.4
* pandas=0.25.3
* scikit-surprise=1.1.0
* Cython=0.29.16
* matplotlib=3.1.2
* tqdm=4.45.0

## Setup

Some algorithms (`.pyx`) are implemented using Cython. In order to run these algorithms, execute the following command:

```
cd src/
python3 setup.py build_ext --inplace
```

## Algorithms

We implemented the following algorithms:

+ *Mean* (baseline.py): this simple algorithm predicts the global sample mean for each missing entry
+ *SVD* (baseline.py): compute the SVD of the training matrix. The training matrix can be imputed with the sample mean, sample median, or with 0s
+ *ALS* (baseline.py)
+ *SGDPP2* (factorization.pyx): an variation of the SVD++ algorithm
+ *pLSA* (plsa.pyx)
+ *SVDthr* (thresholding.py): an implementation of SVD thresholding
+ *VAE* (??)
+ *Ensemble* (??)

We organize these algorithms (with the exception of *VAE* and *Ensemble*) in the `src/source.py` file.

## Modes

Algorithms can be executed in the following modes:

+ *cv*: performs cross-validation on the selected algorithm instance and computes the validation RMSE on a 0.25 split
+ *target_cv*: similar to *cv*, but the RMSE is computed for each rating (i.e. the RMSE for validation ratings equal to 1, 2, 3, 4, 5). This version can be used to check whether the algorithm is properly balanced
+ *kfold*: performs k-fold cross-validation on the selected algorithm instance. The user can decide on the number of folds (10 by default)
+ *grid*: performs grid search on the selected algorithm class using the parameter grid in the `src/source.py` file
+ *random_search*: performs random search on the selected algorithm class using the parameter distribution grid in the `src/source.py` file
+ *dump*: trains the selected algorithm instance on the whole dataset and writes the predictions in a `.csv` file

## Execution

To execute an algorithm with a given mode, run the following command:

```
python3 src/main.py mode algo_class
```

The following options are available:

+ -h: help message
+ --model_num (int): the number of the algorithm instance. By default 1
+ --k (int): the number of folds for the k-fold cross-validation. By default 10
+ --n_iters (int): the number of iterations for the randomized search. By default 10
+ --verbose (bool): whether the algorithm should be verbose. By default False
+ --seed (int): the random seed. By default 0

## Notebooks

As mentioned before, notebooks are meant to give the user a better understanding of the code. (Maybe say that there are plots, ...)
