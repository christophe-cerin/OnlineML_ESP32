## Simple linear regression

The goal of simple linear regression is to predict the value of a dependent variable based on an independent variable. The greater the linear relationship between the independent variable and the dependent variable, the more accurate the prediction. This goes hand in hand with the fact that the greater the proportion of the variance in the dependent variable that can be explained by the independent variable, the more accurate the prediction. Visually, the relationship between variables can be represented by a scatterplot. The greater the linear relationship between the dependent and independent variables, the more the data points lie on a straight line.

This form of analysis estimates the coefficients of a linear equation involving one or more independent variables that best predict the value of the dependent variable. Linear regression fits a straight line or surface that minimizes the differences between predicted and actual output values. Simple linear regression calculators use a least-squares method to find the best-fitting line for a paired data set. You then estimate the value of X (the dependent variable) from Y (the independent variable).
Before performing linear regression, you must ensure that your data can be analyzed using this procedure. Your data must meet certain required assumptions.
Here's how to verify these assumptions:
1. The variables must be measured continuously. Examples of continuous variables include time, sales, weight, and test scores.
2. Use a scatter plot to quickly determine if a linear relationship exists between these two variables. 3. Observations must be independent of each other (i.e., there must be no dependency).
4. Your data must not contain significant outliers.
5. Check for homoscedasticity, a statistical concept that the variances along the best-fitting linear regression line remain equal throughout that line.
6. The residuals (errors) of the best-fitting regression line follow a normal distribution.

## [Program SimpleLinearRegression.cpp](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/SIMPLE-LINEAR-Regression/SimpleLinearRegression.cpp)
## The Result of Program Execution

./SimpleLinearRegression  
coefficents = {9.67564, 0.043611}  
0.910276  
