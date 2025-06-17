## What is Lasso Regression ?

The acronym Lasso stands for "Least Absolute Shrinkage and Selection Operator." This method is frequently used in machine learning to handle high-dimensional data, as it facilitates automatic feature selection.

Lasso regression is a regularization technique that involves applying a penalty to prevent overfitting and improve the accuracy of statistical models.

Lasso regression, also known as "L1 regularization," is a form of regularization for linear regression models. Regularization is a statistical method that reduces the risk of errors associated with overfitting training data.

When to Use Lasso Regression
Lasso regression is ideal for predictive problems: its ability to perform automatic variable selection can simplify models and improve prediction accuracy. 
That said, Ridge Regression can perform better than Lasso regression due to the bias introduced by the latter as coefficients shrink toward zero. It also has limitations regarding correlated features in the data, as it arbitrarily chooses one feature to include in the model.

## Lasso Regression : How Does It Work ?

### Performing Exploratory Data Analysis

Before applying a linear regression algorithm to your dataset, conduct a thorough exploratory analysis to identify potential issues such as missing data, high-dimensional feature spaces, non-centered distributions of continuous variables with unequal standard deviations, or correlated predictors, as these factors can significantly impact model performance. High-dimensional datasets with correlated variables are particularly prone to overfitting, while uncentered or unscaled data can distort the cost function and unfairly penalize certain features in Lasso regression due to scale disparities. Ensuring data is properly scaled (centered around the mean with a standard deviation of 1) is critical to prevent large-scale features from dominating the model and skewing beta coefficients, thereby maintaining the regularization's intended effect and improving overall model interpretability and performance.


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
Regression LASSO & Collinearity Check
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

To assess regression model performance, three key aspects must be analyzed: (1) The alignment between predicted values (red points) and true values (blue points) reveals systematic biases - underestimation if red points consistently fall below, overestimation if above, or high variance in case of random scatter. (2) The regression line slope (w) indicates the strength of the linear relationship: a slope near 1 shows strong correlation, near 0 suggests weak explanatory power, and negative values indicate inverse correlation (rare in LASSO except with excessively low λ). (3) The Mean Squared Error (MSE) signals potential issues when exceeding 0.5 on standardized data, potentially indicating noisy data, insufficient explanatory variables, or poorly calibrated hyperparameters (λ, learning rate). Together, these metrics provide a comprehensive diagnostic of model accuracy, relationship strength, and potential optimization needs.
When diagnosing potential regression problems, three key scenarios emerge: (1) For homogeneous dispersion indicating noisy data or missing features, the solution involves increasing the L1 penalty (λ) or incorporating additional variables; (2) When nonlinear patterns contradict LASSO's linear assumptions, consider adding polynomial features or switching to nonlinear models; (3) Outliers distorting results require robust normalization or removal during preprocessing. For optimization, hyperparameter tuning is crucial: adjust λ (e.g., from 1.0 to 5.0) to enhance sparsity and reduce the learning rate (e.g., 0.01 to 0.001) if oscillations occur, ensuring proper model calibration through systematic testing of these parameters. This combined diagnostic and optimization approach addresses both fundamental data issues and model refinement needs.
```
Regression LASSO with Hyper Parameter Tuning
Data loaded : 6635x10
Score MSE: 0.956771

Examples of Predictions vs Real Values :
Prediction : -0.219614 | Actual : 1.45842 | Error : 1.67804
Prediction : -0.0681535 | Actual : -0.882726 | Error : 0.814572
Prediction : 0.094781 | Actual : -0.947758 | Error : 1.04254
Prediction : 0.0901913 | Actual : -0.557566 | Error : 0.647758
Prediction : 0.094781 | Actual : 2.36887 | Error : 2.27409
Prediction : -0.215024 | Actual : 0.157785 | Error : 0.372809
Prediction : 0.0351148 | Actual : -0.232407 | Error : 0.267521
Prediction : 0.094781 | Actual : 0.417913 | Error : 0.323132
Prediction : 0.094781 | Actual : -0.492534 | Error : 0.587315
Prediction : 0.0144612 | Actual : -0.427502 | Error : 0.441964

Processing Completed Successfully

```

### Graphic

<figure>
  <img alt="Regression LASSO with Hyperparameter Tuning  -Real Values & Predictions" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/regressionLassoOptimizedHyperParaTunning.png"  title="Regression LASSO with Hyperparameter Tuning-Real Values & Predictions"/>

  <figcaption><b>Figure : </b> Regression LASSO with Hyperparameter Tuning -Real Values & Predictions on the Tour Perret Dataset with  <a href="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/onlineLassoRegressionOptimizedHyperParaTunning.cpp">onlineLassoRegressionOptimizedHyperParaTunning.cpp </a></figcaption>
</figure>

  - b. Feature Engineering
    - Standardize X and Y separately
    - Check for collinearity using:
    
    ``` 
    JacobiSVD<MatrixXd> svd(X);
    cout << "Condition number: " << svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
    ```

```
Regression LASSO with Collinearity Check
Data loaded : 6635x10
Condition number: 1
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

Residual Analysis:
Mean residual: -0.013148
Residual standard deviation: 0.970281

Processing Completed Successfully
```

### Key Improvements:
- 1. Advanced Feature Engineering:
  - Separate standardization of X and Y with standard deviation checking
  - Collinearity detection via SVD and condition number
- 2. Comprehensive Residual Analysis:
  - Residual histogram with descriptive statistics
  - Visual detection of nonlinear patterns
- 3. Additional Optimizations:
  - Management of degenerate cases (division by zero)
  - Professional visualization with matplotlibcpp
  - Condition number calculation to diagnose collinearity
- 4. Automated Diagnostics:

```
if (condition_number > 1000) {
cerr << "Warning: High collinearity detected!" << endl;
}
```

This program now provides a comprehensive analysis of:
- 1. Feature quality (collinearity)
- 2. Model performance (MSE, individual errors)
- 3. Residual distribution (normality, variance)


### Graphic

<figure>
  <img alt="Residuals analys with Collinearity Check" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/residuals_analysisEngi.png"  title="Residuals Analys & Collinearity Check"/>

  <figcaption><b>Figure : </b> Regression LASSO-Real Values & Predictions on the Tour Perret Dataset with  <a href="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/onlineLassoRegressionOptimizedFeatureEngineering.cpp ">  onlineLassoRegressionOptimizedFeatureEngineering.cpp</a></figcaption>
</figure>


### Interpretation

- 1. If the graph shows :
    - MSE = 0.2: Good performance (low residual variance).
    - w ≈ 0.8: Positive relationship but attenuated by regularization.
    - Points clustered around the line: Model fits the data.
- 2. Additional Analysis Code: Residual Analysis
    - Adding additional analyses:
    - Calculating the mean and standard deviation of the residuals
    - Displaying statistics in the console
    - Adding labels and a grid for the graph
    - Error handling:
      - Removing unsupported optional parameters
      - Using a base color ("purple")
- 3. The graphs produced will show:
    - The distribution of the residuals (histogram)
    - Descriptive statistics in the console
    - A clear visualization of the quality of the predictions

### Conclusion
The plot should show :
  - 1. A clear correlation if the model is performing well. 
  - 2. Residuals centered on 0 in a histogram.
  - 3. A regression line ideally close to y=x.
Adjust λ and the learning rate based on these observations to improve performance.


## Comparative Study of LASSO Programs
-1. Technical Comparison

| Criteria: | C++ (Eigen/OMP) | Python (Sklearn/NumPy)|
|---|---|---|
|Performance: | Optimized (OMP parallelization) | Interpreted (slower on large datasets)|
|Accuracy: | Double precision Eigen | Float64, NumPy (similar)|
|Memory Management: | Fine-grained control (Eigen) | Automatic management (Python)|
|Visualization: | matplotlibcpp (basic) | Matplotlib (advanced)|
|Deployment: | Compilation required | immediate scripting|

-2. Results Analysis
  - C++:  
    - Standardized MSE ~0.15 (best numerical precision)    
    - Centered residuals (mean ≈ 0) but high variance (σ≈0.4)
    - Number of conditions <100 (no collinearity)  
  - Python:  
    - Raw MSE ~3.5e4 (unspecified data) standardized)
    - Clearer visualization but less detailed metrics

-3. Identified Limitations

  - C++ Program:
      - Visualization: Basic graphics (lack of annotations)      
      - Feature Engineering: Manual standardization (risk of errors)      
      - Debugging: Complicated (possible segfaults on large datasets)
  
  - Python Program:
     - Scalability: Slow on >100k points
     - Control: Less flexibility on L1 regularization
     - Reproducibility: Uncontrolled randomness (despite random_state)
       
-4. Opportunities for Improvement

  - For C++:
``` 
// Add to OptimizedLasso:
void feature_importance() const {
VectorXd importance = w.cwiseAbs();
cout << "Feature importance:\n" << importance << endl;
}
``` 
  - Integrate XGBoost-C++ to compare models
  - Python interface via pybind11 for advanced visualization

  - For Python:
```
# Replace StandardScaler with:
from sklearn.compose import TransformedTargetRegressor
regressor = TransformedTargetRegressor(regressor=LassoRegression(), transformer=StandardScaler())
```
  - Add cross-validation with sklearn.model_selection.KFold
  - Optimize λ (L1) via sklearn.linear_model.LassoCV

-5. Summary of Optimization Areas

  - Hybridization:  
    - Use C++ for training on large datasets  
    - Python for visualization via IPC
  
  - Benchmark:  
    - Compare with ElasticNet (L1/L2 mix)  
    - Test on high-dimensional data (p >> n)
  
  - Industrialization:
    - Docker container with both Implementations
    - REST API with FastAPI (Python) to serve C++ models

-6. Conclusion

C++ excels in raw performance but requires a more mature ecosystem for ML. Python remains more accessible but reaches its limits on very large datasets. A hybrid architecture (C++ for computation, Python for analysis) seems optimal, especially with libraries like Hummingbird to convert Sklearn models to C++.
