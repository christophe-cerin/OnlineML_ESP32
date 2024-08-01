## LibIsolationForest ##

Isolation Forest is an anomaly detection algorithm based around a collection of randomly generated decision trees.

It is an accurate and efficient anomaly detector especially for large databases. Its capacity in handling high
volume databases is highly desirable for real life applications.

Most existing model-based approaches to anomaly detection construct a profile of normal instances, then identify instances that do not conform to the normal profile as anomalies. 
This method proposes a fundamentally different model-based method that explicitly isolates anomalies instead of profiles normal points.  
The use of isolation enables the proposed method, iForest, to exploit sub-sampling to an extent that is not feasible in existing methods, 
creating an algorithm which has a linear time complexity with a low constant and a low memory requirement.  
iForest also works well in high dimensional problems which have a large number of irrelevant attributes, and in situations where training set does not contain any anomalies.


## The Result of Program Execution ##
```
./IsolationForestiForestpy
Test 1:
-------
Average of control test samples: 8.04
Average of control test samples (normalized): 0.29638
Average of outlier test samples: 5.13
Average of outlier test samples (normalized): 0.461102
Total time for Test 1: 0.00878337 seconds.

Test 2:
-------
Average of control test samples: 8.7015
Average of control test samples (normalized): 0.525309
Average of outlier test samples: 5.0771
Average of outlier test samples (normalized): 0.687113
Total time for Test 1: 0.0489592 seconds.

```
