# FACtor Score IteM reductIon with Lasso Estimator (FACSIMILE)

This package implements the FACSIMILE method for approximating factor scores based on reduced item sets. Given a scenario where a large number of items are available to measure a latent trait, FACSIMILE selects a subset of items that can be used to approximate the factor scores that would be obtained if all items were used. 

The method uses Lasso-regularised regression to select items that are most predictive of the factor scores, and determine coefficients for the selected items that can be used to approximate the factor scores.


## Installation

First, clone or download this repository. The package can then be installed using pip from the root directory:
    
```bash
pip install .
```

## Usage

The package can be used to select items and approximate factor scores for a given dataset. In general, the simplest way to do this is to use the provided optimisation methods, which will evaluate the performance of different levels of regularisation (resulting in different numbers of items being included). 

```python
from facsimile.eval import FACSIMILEOptimiser

# Initialise the optimiser
optimiser = FACSIMILEOptimiser(n_iter=100, n_jobs=10)

# Fit 
optimiser.fit(X_train, y_train, X_val, y_val)

```

The best performing model can then be selected and used to approximate factor scores for a new dataset:

```python
# Get the best classifier
best_clf = optimiser.get_best_classifier()

# Fit
best_clf.fit(X_train, y_train)

# Get predictions
y_pred = best_clf.predict(X_test)
```

Similarly, it is possible to select the best performing model subject to the requirements 

```python
# Get the best classifier
best_clf_70 = optimiser.get_best_classifier_max_items(70)

# Fit
best_clf_70.fit(X_train, y_train)

# Get predictions
y_pred = best_clf_70.predict(X_test)

```