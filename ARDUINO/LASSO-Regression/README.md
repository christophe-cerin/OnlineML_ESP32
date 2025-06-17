## What is Lasso Regression ?

The acronym Lasso stands for "Least Absolute Shrinkage and Selection Operator." This method is frequently used in machine learning to handle high-dimensional data, as it facilitates automatic feature selection.

Lasso regression is a regularization technique that involves applying a penalty to prevent overfitting and improve the accuracy of statistical models.

Lasso regression, also known as "L1 regularization," is a form of regularization for linear regression models. Regularization is a statistical method that reduces the risk of errors associated with overfitting training data.

When to Use Lasso Regression
Lasso regression is ideal for predictive problems: its ability to perform automatic variable selection can simplify models and improve prediction accuracy. 
That said, Ridge Regression can perform better than Lasso regression due to the bias introduced by the latter as coefficients shrink toward zero. It also has limitations regarding correlated features in the data, as it arbitrarily chooses one feature to include in the model.

## Lasso Regression : How Does It Work ?

### Performing Exploratory Data Analysis

Before applying a linear regression algorithm to your dataset, explore the data to identify potential underlying issues. It is important to determine whether:
• some data are missing
• there are a large number of features
• the distribution of continuous variables is centered around the mean with equivalent standard deviations
• predictors are correlated with each other.
Understanding these elements is important because datasets with high dimensionality and correlated variables can be prone to overfitting. 
Data that is not centered around the mean with a standard deviation of 1 will also need to be rescaled to limit the impact of large scales on the model. 
If features are not rescaled, this can negatively affect the cost function, which in turn will impact the beta coefficients. Simply put, unscaled features can lead to unintended penalties in Lasso regression due to unit differences.

### Split the Data and Rescaling Continuous Predictors

Once we have performed exploratory data analysis, we will split the data into a training set and a test set. After this splitting step, rescaling is applied to the data as needed. 
Z-score scaling is a common approach to feature scaling: it rescales features so that they have a standard deviation of 1 and a mean of 0.


## Usage
### Compilation

```
g++ online_lasso_regression_optimized_en.cpp -I /home/mamadou/src -L /usr/include/python3.10 -lpython3.10 -o online_lasso_regression_optimized_en.out
```
### Results

Iteration over the data in blocks of W rows 4096 :

```
Data loaded : 6635x10
Convergence after 993 iterations
Score MSE: 0.941619

Examples of Predictions vs Real Values :
Prediction : -0.344497 | Actual : 1.45842 | Error : 1.80292
Prediction : -0.106083 | Actual : -0.882726 | Error : 0.776643
Prediction : 0.150393 | Actual : -0.947758 | Error : 1.09815
Prediction : 0.143169 | Actual : -0.557566 | Error : 0.700735
Prediction : 0.150393 | Actual : 2.36887 | Error : 2.21848
Prediction : -0.337272 | Actual : 0.157785 | Error : 0.495057
Prediction : 0.0564726 | Actual : -0.232407 | Error : 0.288879
Prediction : 0.150393 | Actual : 0.417913 | Error : 0.267519
Prediction : 0.150393 | Actual : -0.492534 | Error : 0.642928
Prediction : 0.0239616 | Actual : -0.427502 | Error : 0.451464

Processing Completed Successfully

``` 

### Graphic

<figure>
  <img alt="Regression LASSO-Real Values & Predictions" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/regression_lasso_optimized.png"  title="Regression LASSO-Real Values & Predictions"/>

  <figcaption><b>Figure : </b> Regression LASSO-Real Values & Predictions on the Tour Perret Dataset with  <a href="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/online_lasso_regression_optimized_en.cpp">online_lasso_regression_optimized_en.cpp </a></figcaption>
</figure>

### Advanced Analysis of the LASSO Regression Plot
1. Graph Structure
The resulting graph shows three main elements:
  - Blue points: Standardized true values ​​(Y)
  - Red points: LASSO model predictions
  - Green line: Regression line (Y = wX + b)

2. Interpreting the Results
  a. Alignment of Predictions with True Values
    - A good model would show overlapping red and blue points. If the red points are scattered around the blue points, this indicates:
        - Underestimation (red points systematically below)
        - Overestimation (red points systematically above)
        - High variance (random scatter)
  b. Regression Line Slope (w)
    - Slope close to 1: Strong linear relationship between X and Y.
    - Slope close to 0: Weak relationship, suggesting that X poorly explains Y.
    - Negative slope: Inverse correlation (rare in LASSO unless λ is too low).
  c. MSE Deviation: Mean Squared Error (MSE)
    - A high MSE (> 0.5 on standardized data) indicates:
        - Noisy data
        - Insufficient explanatory variables
        - Incorrectly set hyperparameters (λ, learning rate)
   
3. Diagnosing Potential Problems
  Case 1: Homogeneous Dispersion
    • Problem: Noise in the data or missing features.
    • Solution: Increase the L1 penalty (λ) or add variables.
  Case 2: Visible Nonlinear Pattern
    • Problem: The LASSO is linear, but the data is not.
    • Solution: Add polynomial features or use a nonlinear model.
  Case 3: Outliers
    • Problem: Outliers disrupting the regression.
    • Solution: Preprocess the data (robust normalization, outlier removal).
   
4. Possible Optimizations
  a. Hyperparameter Tuning
  OptimizedLasso model(0.01 /*lr*/, 1000 /*iter*/, 1.0 /*l1*/);
    • λ (l1): Increase for more sparsity (1.0 → 5.0).
    • Learning Rate: Reduce for oscillations (0.01 → 0.001).
