# Regression analysis

## Introduction

Within the framework of statistical modeling, regression analysis comprises a collection of statistical methodologies aimed at quantifying the relationships between a dependent variable—frequently referred to as the outcome, response variable, or, in the context of machine learning, a label—and one or more independent variables devoid of errors. These independent variables are commonly termed as regressors, predictors, covariates, explanatory variables, or features.

This page introduces the following regression methods:

- Gradient Boosting;
- Lasso;
- Random Forest;
- Ridge;
- SVR (Support Vector Regression)
- The file `batch_regression.py` aggregates and compiles different methods.

All the files with prefix `batch_` implement offline (or batch) regression methods, meaning the data are known in advance. Extreme-edge incremental means that data continually arrives, flows in a window/buffer of fixed size, so we need to build the regression only with the available data in the buffer.

## Offline learning and regression

<figure>
  <img src="Images/LASSO.png" alt="My image caption">
  <figcaption><b>Fig. 1:</b> Exploring data with LASSO method</figcaption>
</figure>

  <p>  <br></p>
<figure>
  <img src="Images/RIDGE.png" alt="My image caption">
  <figcaption><b>Fig. 2:</b> Exploring data with RIDGE method</figcaption>
</figure>

  <p>  <br></p>

  The next figures correspond to outcomes of the `batch_SVR.py` Python code, and illustrate different situations and configurations for the parameters of the model. Please, refer to the source code.

 <p>  <br></p>
  <figure>
  <img src="Images/SVR_1.png" alt="My image caption">
  <figcaption><b>Fig. 3:</b> Exploring dataset with SVR (1/6)</figcaption>
</figure>

 <p>  <br></p>
  <figure>
  <img src="Images/SVR_2.png" alt="My image caption">
  <figcaption><b>Fig. 4:</b> Exploring dataset with SVR (2/6)</figcaption>
</figure>

 <p>  <br></p>
  <figure>
  <img src="Images/SVR_3.png" alt="My image caption">
  <figcaption><b>Fig. 5:</b> Exploring dataset with SVR (3/6)</figcaption>
</figure>

 <p>  <br></p>
  <figure>
  <img src="Images/SVR_4.png" alt="My image caption">
  <figcaption><b>Fig. 6:</b> Exploring dataset with SVR (4/6)</figcaption>
</figure>

 <p>  <br></p>
  <figure>
  <img src="Images/SVR_5.png" alt="My image caption">
  <figcaption><b>Fig. 7:</b> Exploring dataset with SVR (5/6)</figcaption>
</figure>

 <p>  <br></p>
  <figure>
  <img src="Images/SVR_6.png" alt="My image caption">
  <figcaption><b>Fig. 8:</b> Exploring dataset with SVR (6/6)</figcaption>
</figure>
  <p>  <br></p>

## Extreme-edge incremental learning and regression
