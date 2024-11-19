## Generalized Hebbian  Algorithm  (GHA)

The Generalized Hebbian Algorithm is a neural approach to Principal Component Analysis (PCA) developed by Sanger, known as the Generalized Hebbian Algorithm (GHA). This algorithm is mainly used in the field of image processing, in particular for image compression while preserving maximum information (optimal coding). GHA is able to extract the principal components of very high-dimensional vectors, characteristic of the data, which corresponds to the dimension of the representation. Moreover, it focuses only on the calculation of the first, most significant principal components, which results in a considerable reduction in the required computational resources. Due to its adaptive nature, the algorithm can also provide a rough estimate of the results, which can be refined later if necessary, unlike classical PCA software that calculates all eigenvectors with maximum precision, even if this is not always required.

Nevertheless, GHA has some disadvantages. As with neural techniques, it is necessary to measure the learning time empirically, and learning times can be significant to achieve adequate accuracy, especially for components that have lower variance. This algorithm causes calculation errors to cluster from one neuron to another, which inevitably reduces the accuracy of the following components. In practice, it is therefore crucial to focus on the first essential components. Remember that the algorithm does not directly determine the proportion of variance associated with each principal component.
It is therefore possible to easily determine the input variance of each neuron on a sample of the corpus in order to identify the number of truly beneficial components (we will stop learning new neurons if their variance turns out to be too low).

Ultimately, GHA is only justified when it is possible to limit ourselves to the first essential components. However, this is often the case, because they constitute the majority of the information present in the data. Finally, like PCA, GHA is an exclusively linear technique capable of identifying only linear correlations between data (i.e. it is based solely on covariance).

By analogy with neurobiology, the neural network is composed of neurons connected to each other. A system like this consists of a network of interconnected small automata that automatically modify the parameters (weights) recurrently.

- the global mechanism that facilitates the achievement of desired tasks (generally a categorization or a diagnosis)
- the absence of information localization
- a largely parallel operation
- the artificial creation of meaning (by seeking a correlation between the data)
- self-organization
- the emergence of global configurations based on connections between simple components
- a system devoid of determination. 

**Each formal neuron**
- calculates the weighted sum of its inputs
- transmits its internal state to the neurons to which it is connected
- some neurons will serve as input, and others as output, the processing is distributed across the entire network
- Parallel Distributed Processing
  - Elementary and parallel calculations
  - Data/information distributed in the network

**Each neuron connection is assigned a weight modulating the transmission of activity**
- these weights are gradually adjusted by learning procedures from an iterative presentation of the data
- this allows the system to be adapted according to the inputs in order to solve the problem posed
- there are various methods but unsupervised Hebbian learning, which requires input data without other information, is our center of interest

**Application areas**
- Classification
  - divide objects into several classes
  - quantitative data => qualitative information
  - pattern recognition
- Operational Research
  - solve problems for which the solution is not known
- Associative Memory
  - restore data from incomplete and/or noisy information

### Learning method

Learning is a stage in the development of a neural network where the behavior of the network undergoes changes until the desired behavior is achieved.

There are two main categories of learning algorithms:
- Supervised learning: Supervised learning is a method of artificial intelligence (machine learning) in which the algorithm is trained to perform a specific task using a set of matched data, each with an annotation specifying the expected result.
- Unsupervised learning: Unsupervised learning is a branch of machine learning distinguished by the collection and analysis of unlabeled data. In this way, these algorithms acquire the ability to detect patterns or clusters in the information, minimizing any human intervention.

Learning rules: Learning involves modifying the weight of neuronal links.

<picture>
<center>
<img alt="Matrice X" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/images/poids-liaisons.png" width=60% height=60%  title="Matrice X"/>
</center>
</picture>

**Hebb's law; Δw<sub>ij</sub> = Ra<sub>i</sub>a<sub>j</sub>**

If two connected units are active simultaneously, the weight of their connection is increased or decreased. R is a positive constant representing the learning rate.
