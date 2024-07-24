## Linear Model for MicroPython on ESP32
The goal of this repository is to provide accessible implementations of common machine learning models that can run in 
resource-limited environments, specifically designed to run on the ESP32. These models are conversions of the main [River](https://riverml.xyz/latest/) algorithms, which can be viewed [here](https://github.com/online-ml/river/tree/main/river/linear_model). Due to the limited functionality of micropython-ulab, which does not support the full River algorithms, we have converted the River code into micropython-ulab compatible versions as described in this repository. 
The tools are designed for use with MicroPython on ESP32 devices, leveraging the micropython-ulab library for numerical operations.

### Pre-requisites
MicroPython-ulab: Ensure you have MicroPython-ulab installed on your ESP32. You can find installation instructions in [MICROPYTHON-ULAB](https://github.com/online-ml/river/tree/main/river/feature_extraction)

### Example Usage

#### Bayesian Linear Regression
A probabilistic approach to linear regression where the weights are treated as random variables with a prior distribution. 
This model updates its beliefs about the weights as more data becomes available.

 ```
model = BayesianLinearRegression(alpha=1, beta=1, smoothing=0.8)
data = [{'feature1': 0.5, 'feature2': 1.5}, {'feature1': 1.0, 'feature2': 2.0}]
targets = [3.0, 4.5]

for x, y in zip(data, targets):
    model.learn_one(x, y)

prediction = model.predict_one({'feature1': 0.5, 'feature2': 1.5})
print(f"Prediction: {prediction}")
 ```
#### Linear Regression
A classic regression model that finds the best-fit line for a set of data. This implementation supports L1 and L2 regularization to prevent overfitting and is trained using gradient descent.

 ```
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([3, 7, 11])

model = LinearRegression(learning_rate=0.01, l2=0.1)
model.fit(X, y, epochs=1000)

prediction = model.predict(np.array([[3, 4]]))
print(f"Prediction: {prediction}")
 ```

#### Logistic Regression
A classification model that predicts probabilities for binary outcomes using the logistic function. It supports regularization and can learn an intercept term.
 ```
model = LogisticRegression(learning_rate=0.01)
X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
y = np.array([0, 0, 1])

model.fit(X, y, epochs=100)
predictions = model.predict(X)
print(f"Predictions: {predictions}")

 ```

#### PARegressor (Passive-Aggressive Regressor)
An online learning algorithm suitable for regression tasks. It adjusts the model's weights in response to each new data point, making it efficient for real-time applications.
 ```
from pa_regressor import PARegressor

model = PARegressor(C=0.01, mode=1, eps=0.1, learn_intercept=False)
X = [{0: 1.0, 1: 2.0}, {0: 2.0, 1: 3.0}, {0: 3.0, 1: 4.0}]
y = [4.0, 5.0, 6.0]

for xi, yi in zip(X, y):
    model.learn_one(xi, yi)

prediction = model.predict_one({0: 4.0, 1: 5.0})
print(f"Prediction: {prediction}")

 ```

#### PARegressor (Passive-Aggressive Regressor)
An online learning algorithm suitable for regression tasks. It adjusts the model's weights in response to each new data point, making it efficient for real-time applications.

 ```
model = PARegressor(C=0.01, mode=1, eps=0.1, learn_intercept=False)
X = [{0: 1.0, 1: 2.0}, {0: 2.0, 1: 3.0}, {0: 3.0, 1: 4.0}]
y = [4.0, 5.0, 6.0]

for xi, yi in zip(X, y):
    model.learn_one(xi, yi)

prediction = model.predict_one({0: 4.0, 1: 5.0})
print(f"Prediction: {prediction}")
 ```

#### PAClassifier (Passive-Aggressive Classifier)
An online learning algorithm designed for binary classification tasks. It uses hinge loss to update the weights, aiming to correctly classify data points as they arrive.
 ```
model = PAClassifier(C=0.01, mode=1)
X = [{0: 1.0, 1: 2.0}, {0: 2.0, 1: 3.0}, {0: 3.0, 1: 4.0}]
y = [True, False, True]

for xi, yi in zip(X, y):
    model.learn_one(xi, yi)

proba = model.predict_proba_one({0: 2.0, 1: 3.0})
print(f"Probabilities: {proba}")

 ```
