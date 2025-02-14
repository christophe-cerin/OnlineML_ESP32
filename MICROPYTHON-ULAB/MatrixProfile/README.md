# Matrix Profile: from offline to extreme edge-incremental case

The matrix profile constitutes a data structure and a set of algorithms that facilitates the resolution of challenges in time series analysis. It is characterized by its robustness, scalability, and minimal reliance on parameters. The subject holds significant implications for various analytical techniques, including time series motif discovery, time series joins (joining two long time series based on their most correlated segments), and shapelet discovery for classification purposes. Indeed, shapelets are time series snippets that can be used to classify unlabeled time series. Additionally, it impacts the fields of density estimation, semantic segmentation, visualization, rule discovery, and clustering, among others.

The following links help to enter into the domain of Matrix Profiles:

- [The UCR Matrix Profile Page](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html)
- [STUMPY tutorial](https://stumpy.readthedocs.io/en/latest/tutorials.html)
- [MPF](https://matrixprofile.org/)

# Our work

The file `MatrixProfile.py` implements multiple offline Matrix Profiles algorithms selected according to the following set of criteria:

- ***Self-sufficient algorithm:*** The algorithm must be implemented without using large-scale libraries to fit resource-constrained microcontrollers. 
- The programming language should be Micropython. Hence, we can count only on an N-dimensional array package similar to NumPy.
- ***Execution time:*** It needs to be minimal.
- ***Multiple core processing:*** Is it possible to parallelize the algorithm? Some microcontrollers have more than one core.

Concerning the execution times for our dataset `TourPerret.csv`, we get the following numbers on MacBook-Air M3 with 16GB of RAM:

  Execution time AAMP: 0.03260016441345215 seconds
  Execution time AAMP_ned: 0.041748046875 seconds
  Execution time AAMP_mp: 0.03651022911071777 seconds
  Execution time ACAMP_1: 0.12642884254455566 seconds
  Execution time stump: 11.52841305732727 seconds
  Execution time scrimp++: 0.011008977890014648 seconds

The file `RingBuffer.py` implements an extreme edge-incremental matrix profile algorithm based on ACAMP. Data flow continually through a ring buffer (to limit the RAM footprint), and at different time intervals, we compute the Matrix Profile. Moreover, the implementation is a simulation for the following scenario. A set of sensors periodically sends data (temperature, humidity...) to a microcontroller. The microcontroller can also receive a second type of message to specify that an external agent (user, cloud, or fog) would like to consult the current matrix profile.

  
