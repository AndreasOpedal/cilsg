# Collaborative Filtering - Unstable Geniuses

We explain how to the submission directory is organized, how to run the algorithms (and in which modes), and the role of notebooks.

## Table of Contents

1. [Directory structure](#directory)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Algorithms](#algorithms)
    * [Structure](#structure)
    * [Adding a new instance](#newinstance)
    * [Adding a new algorithm](#newalgo)
5. [Modes](#modes)
6. [Execution](#execution)
    * [Example: reproducing the results](#reproduce)
    * [Example: performing cross-validation](#crossvalidation)
7. [Notebooks](#notebooks)

## Directory structure

This submission directory is structured as follows: the `data/` directory contains the training data and the missing entries for the submission. The `src/` data contains the implementation of most algorithms (apart from the auto-encoder). The `notebook/` directory contains notebooks for each algorithm. Finally, the `predictions/` directory holds the predictions for our best models.

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

+ `Mean (src/baseline.py)`: this simple algorithm predicts the global sample mean for each missing entry
+ `SVD (src/baseline.py)`: compute the SVD of the training matrix. The training matrix can be imputed with the sample mean, sample median, or with zeros
+ `ALS (src/baseline.py)`: an implementation of alternating least squares
+ `SVDPP2 (src/factorization.pyx)`: a variation of the SVD++ algorithm
+ `pLSA (src/plsa.pyx)`: an implementation of pLSA, which uses the EM-algorithm, where we add an additional SVD step to compute the final prediction
+ `SVDthr (src/thresholding.py)`: an implementation of SVD thresholding
+ `VAE (??)`: an implementation of a variational auto-encoder
+ `Ensemble (notebooks/ensemble.ipynb)`: an ensemble method where the results of different algorithms are averaged

Hyper-parameters for each class are well documented in the source code.

### Structure

With the exception of *VAE* and *Ensemble*, algorithms' creation is managed in the `src/source.py` file. Two dictionaries are used to organize algorithm classes and instances. The `algo_classes` dictionary maps algorithms' names to algorithms Python classes, e.g. `algo_classes['SVD'] = SVD`. The `instances` dictionary maps algorithms' names to a dictionary of that algorithm's instances. Instances are mapped by a unique number. For example, the default version of the *SVD* algorithm is mapped as `instances['SVD'][1] = SVD()`.

**Note**: the key for the algorithm class in both dictionaries must be spelled the same way as the name of the actual Python class.

### Adding a new instance

Our structure makes it easy to add a new instance for a given algorithm.
For example, assume we want to change the number of latent factors in the *SVD* from 160 (the default value) to 50. Because we are not creating a new algorithm class, we merely have to add a new entry in the `instances` dictionary under the *SVD* key. We thus add the following line in the `src/source.py` file:

```
instances['SVD'][2] = SVD(n_factors=50)
```

### Adding a new algorithm

To add a new algorithm, we first refer to the following link[https://surprise.readthedocs.io/en/stable/building_custom_algo.html] on how to build a custom class using the Surprise package.
After doing that, we need to add the new algorithm to the `src/source.py` file. Let us assume this new algorithm is called `MyAlgo` and it is written in the file `src/myalgo.py`. We add the following lines in the `src/source.py`

1. `from myalgo.py import MyAlgo`
2. `algo_classes['MyAlgo'] = MyAlgo`
3. `instances['MyAlgo'][1] = MyAlgo()`

## Modes

Each algorithm can be executed in a specific *mode*. The available modes are the following:

+ `cv`: performs cross-validation on the selected algorithm instance and computes the validation RMSE on a 0.25 split
+ `kfold`: performs k-fold cross-validation on the selected algorithm instance. The user can decide on the number of folds (10 by default)
+ `grid`: performs grid search on the selected algorithm class using the parameter grid in the `src/source.py` file
+ `random_search`: performs random search on the selected algorithm class using the parameter distribution grid in the `src/source.py` file
+ `dump`: trains the selected algorithm instance on the whole dataset and writes the predictions in a `.csv` file in the `predictions/` directory

## Execution

To execute our code in the command line, run the following commands:

```
cd src/
python3 main.py mode algorithm
```

where `mode` is listed in the (modes)[#modes] section, and `algorithm` is listed in the (algorithms)[#algorithms] section.

The following options are available:

+ `-h`: help message
+ `--model_num` (int): the number of the algorithm instance. By default 1 (corresponding to our best model configurations)
+ `--k` (int): the number of folds for the k-fold cross-validation. By default 10
+ `--n_iters` (int): the number of iterations for the randomized grid search. By default 10
+ `--verbose` (bool): whether the algorithm should be verbose. By default False
+ `--seed` (int): the random seed. By default 0

**Note**: algorithms `VAE` and `Ensemble` can only be run via notebooks (see `notebooks/` directory).

### Example: reproducing the results

Let us reproduce the results for *SVD*, by training the algorithm on the full trainset and writing its predictions to a `.csv` file:

```
cd src/
python3 main.py dump SVD
```

### Example: performing cross-validation

Let us perform cross-validation on the *SVD* instance we added in the [previous](#newinstance) section:

```
cd src/
python3 main.py cv SVD --model_num=2
```

## Notebooks

Notebooks are meant to give the user a better understanding of the code implemented in `src/`. Their goal is to provide the mathematical background, clarify the code, and creating plots to better illustrate how the algorithm learns the weights.
Notebooks are not meant however to train the full model or to perform cross-validation, as it would be quite time-consuming. The exception to this rule is the notebook for the auto-encoder.
