# Regression analysis

## Introduction

In statistical modeling, regression analysis is a set of statistical processes for estimating the relationships between a dependent variable (often called the outcome or response variable, or a label in machine learning parlance) and one or more error-free independent variables (often called regressors, predictors, covariates, explanatory variables or features).

This page introduces the following methods:

- Gradient Boosting;
- Lasso;
- Random Forest;
- Ridge;
- SVR (Support Vector Regression)
- The file `batch_regression.py` aggregates and compiles different methods.

All the files with prefix `batch_` implement offline (or batch) regression methods, meaning the data are known in advance. Extreme-edge incremental means that data continually arrives, flows in a window/buffer of fixed size, so we need to build the regression only with the available data in the buffer.

## Offline learning and regression



## Extreme-edge incremental learning and regression
