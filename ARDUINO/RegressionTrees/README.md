## Regression Trees : CART (classification and regression trees)

CART is a variant of the decision tree algorithm. It is a predictive algorithm that explains how the values ​​of the target variable can be predicted based on others.  

Regression: The cost function that is minimized to choose the split points is the sum of the squared error over all training samples included in the rectangle.  

Classification: The Gini cost function is used to provide an indication of node purity, where node purity refers to the mix of training data assigned to each node. In other words, it helps determine which class the target variable is most likely to fall into when it is continuous.  

 Gini Index (cost function for evaluating splits in the dataset): The Gini Index is a measure for classification tasks in CART. It stores the sum of the squared probabilities of each class. It calculates the degree of probability of a specific variable being misclassified when chosen randomly and a change in the Gini coefficient. It operates on categorical variables, provides "pass" or "fail" results and therefore only conducts binary splitting. 

 
 1. The degree of the Gini index varies from 0 to 1,  
   - Where 0 represents that all elements are allied to a certain class, or that only one class exists there.
   - Gini value index 1 means that all elements are randomly distributed among different classes, and
   - A value of 0.5 indicates that elements are evenly distributed in certain classes.
 2. Create a division
 3. Build a tree  
  3.1 Leaf nodes (maximum tree depth, minimum number of node records)  
  3.2 Recursive slicing  
  3.3 Building a tree  
 4. Make a prediction  

## [Program ClassificationRegressionTrees.cpp](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/RegressionTrees/ClassificationRegressionTrees.cpp)
## Result of program execution 

```
./ClassificationRegressionTrees
depth: 0, featureIdx: 0, featureValue: 272
depth: 1, label: 2
depth: 1, label: 3
pred: 2, gt: 2
pred: 2, gt: 2
pred: 2, gt: 1
pred: 2, gt: 2
pred: 2, gt: 1
pred: 3, gt: 3
pred: 2, gt: 1
pred: 2, gt: 2
pred: 2, gt: 1
pred: 2, gt: 2
pred: 2, gt: 1
pred: 2, gt: 2
pred: 2, gt: 1
pred: 2, gt: 1
pred: 2, gt: 2
pred: 2, gt: 1
pred: 2, gt: 1
pred: 2, gt: 1
pred: 2, gt: 2



pred: 3, gt: 0
pred: 2, gt: 2
pred: 2, gt: 2
pred: 3, gt: 3
pred: 3, gt: 0
pred: 3, gt: 0
pred: 3, gt: 0
pred: 3, gt: 0
pred: 3, gt: 0
pred: 2, gt: 2
pred: 3, gt: 0
pred: 2, gt: 2
pred: 3, gt: 3
pred: 3, gt: 3
pred: 2, gt: 2
pred: 3, gt: 0
pred: 3, gt: 0
pred: 2, gt: 1
pred: 3, gt: 3
pred: 2, gt: 2
pred: 3, gt: 0

``
