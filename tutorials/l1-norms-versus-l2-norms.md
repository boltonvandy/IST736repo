# L1 Norms versus L2 Norms

Ridge regression and lasso regression are two different techniques for increasing the robustness against colinearity of ordinary least squares regression. Both of these algorithms attempt to minimize a cost function. The cost is a function of two terms: one, the residual sum of squares (RSS), taken from ordinary least squares; the other, an additional regularizer penalty. The second term is an L2 norm in ridge regression, and an L1 norm in lasso regression.

## Overview
Let's look at the equations. In ordinary least squares, we solve to minimize the following cost function:

$$\text{Cost} = (y-X\beta)^T(y-X\beta)$$

This term is the RSS, residual sum of squares. In ridge regression we instead solve:

$$\text{Cost} = (y-X\beta)^T(y-X\beta)+\lambda\beta^T\beta$$

The $\lambda\beta^T\beta$ term is an L2 norm.

In lasso regression we instead solve:

$$\text{Cost} = (y-X\beta)^T(y-X\beta)+\lambda \left|\beta\right|$$

The $\lambda \left| \beta \right|$ term is an L1 norm.

At a higher level, the chief difference between the L1 and the L2 terms is that the L2 term is proportional to the square of the $\beta$ values, while the L1 norm is proportional the absolute value of the values in $\beta$. This fundamental difference accounts for all of the difference between how lasso regression and ridge regression "work". L1-verus-L2 pops up elsewhere in machine learning as well, so it's important to understand what's going on here!

## Definition of a norm

Let's step back for a moment and consider the question: what is a norm?

A norm is a mathematical thing that is applied to a vector (like the vector $\beta$ above). The norm of a vector maps vector values to values in $[0, \infty)$. In machine learning, norms are useful because they are used to express distances: this vector and this vector are so-and-so far apart, according to this-or-that norm.

Going a bit further, we define $|| x ||_p$ as a "p-norm". Given $x$, a vector with $i$ components, a p-norm is defined as:

$$|| x ||_p = \left(\sum_i |x_i|^p\right)^{1/p}$$

The simplest norm conceptually is Euclidean distance. This is what we typically think of as distance between two points in space:

$$|| x ||_2 = \sqrt{\left(\sum_i x_i^2\right)} = \sqrt{x_1^2 + x_2^2 + \ldots + x_i^2}$$

Another common norm is taxicab distance, which is the 1-norm:

$$|| x ||_1 = \sum_i |x_i| = |x_1| + |x_2| + \ldots + |x_i|$$

Taxicab distance is so-called because it emulates moving between two points as though you are moving through the streets of Manhattan in a taxi cab. Instead of measuring the distance "as the crow flies" it measures the right-angle distance between two points:

![](https://upload.wikimedia.org/wikipedia/commons/0/08/Manhattan_distance.svg)

You can read more about taxicab geometry [here](https://en.wikipedia.org/wiki/Taxicab_geometry).

## p-norms and regularization

Taxicab distance is the 1-norm, also known as the L1 norm. The L2 norm is actually the 2-norm, Euclidian distance, squared. Hence, we can rewrite our cost equations as:

$$\text{Ridge Cost} = (y-X\beta)^T(y-X\beta)+||\beta||_2^2$$
$$\text{Lasso Cost} = (y-X\beta)^T(y-X\beta)+||\beta||_1$$

This process of adding a norm to our cost function is known as regularization. We can regularize the data for different underlying reasons and with different effects. In the case of ridge and lasso regression, both of these regularizers are built to problem-solve colinearity and model complexity; but as we saw in earlier notebooks. the way in which they go about doing so is fundamentally different.

The properties of regularizing with L1 and L2 norms is what causes these differences. Since these norms will pop up in other places later, it's a good idea to study what gives them their properties, using ridge regression and lasso regression as guides.

(this notebook is based on [this blog post](http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/))

## L1-L2 norm comparisons

### Robustness: L1 > L2

Robustness is defined as resistance to outliers in a dataset. The more able a model is to ignore extreme values in the data, the more robust it is.

The L1 norm is more robust than the L2 norm, for fairly obvious reasons: the L2 norm squares values, so it increases the cost of outliers exponentially; the L1 norm only takes the absolute value, so it considers them linearly.

### Stability: L2 > L1

Stability is defined as resistance to horizontal adjustments. This is the perpendicular opposite of robustness.

The L2 norm is more stable than the L1 norm. A later notebook will explore why.

### Solution numeracy: L2 one, L1 many

Because L2 is Euclidean distance, there is always one right answer as to how to get between two points fastest. Because L1 is taxicab distance, there are as many solutions to getting between two points as there are ways of driving between two points in Manhattan! This is best illustrated by the same graphic from earlier:

![](https://upload.wikimedia.org/wikipedia/commons/0/08/Manhattan_distance.svg)

## L1-L2 regularizer comparisons

### Computational difficulty: L2 > L1

L2 has a closed form solution because it's a square of a thing. L1 does not have a closed form solution because it is a non-differenciable piecewise function, as it involves an absolute value. For this reason, L1 is computationally more expensive, as we can't solve it in terms of matrix math, and most rely on approximations (in the lasso case, coordinate descent).

### Sparsity: L1 > L2

Sparsity is the property of having coefficients which are highly significant: very near 0 or very not near 0. In theory, the coefficients very near 0 can later be eliminated.

Feature selection is a further-involved form of sparsity: instead of shrinking coefficients near to 0, feature selection is taking them to exactly 0, and hence excluding certain features from the model entirely. Feature selection is a technique moreso than a property: you can do feature selection as an additional step after running a highly sparse model. But lasso regression is interesting in that it features inbuilt feature selection, while ridge regression is just very sparse.

That about covers the high-level properties of L2 and L1 norms and regularizers. Hopefully you can see how these properties are exactly the same ones exposed in ridge and lasso regression!


```python

```
