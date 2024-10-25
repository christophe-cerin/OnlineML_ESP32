## Principal Component Analysis (PCA) or Neural Factor Analysis (NFA)

PCA aims to return to a limited dimensional space by distorting the possibilities of reality as little as possible. The objective is therefore to obtain the most appropriate synthesis possible of the preliminary information.

Indeed, PCA involves the calculation of a limited number of new dimensions, which represent the linear combinations of the initial dimensions of the data (i.e. descriptive characteristics). These new dimensions do not present any correlation and reflect the greatest variance of the information (based on the mean). In other words, it is a factorial technique of dimension reduction used for the statistical analysis of complex quantitative data.

The eigenvectors, ranked by decreasing eigenvalues, belong to the new axes of the information covariance matrix. These are the main axes of dispersion of the data cloud, ranked in decreasing order of importance. The corresponding eigenvalues ​​designate the proportion of variance expressed by each axis. Thus, the first axes generally produce the majority of the variance. The new data values ​​for each axis were identified as the principal components.

This technique can serve as a dual purpose data compression method and exploration tool in highly multidimensional domains. Thus, the calculation of principal axes thus performed not only facilitates the reduction of information, but also the interpretation of the domain in question, since the new measurements are usually extremely large.

### 1- Mathematical Design

### 1.1- Data in PCA

In PCA, the data are presented in a table ***X*** with ***n*** rows and ***p*** columns where
  
  - each row represents an individual
  - each column represents a variable

The variables have a quantitative nature: the matrix ***X*** is composed of numerical values. It is the variance-covariance matrix (or that of correlations) that will allow an appropriate summary to be made, because it focuses mainly on the distribution of the data in question.

By an appropriate mathematical process, we will extract in small numbers the factors that we are looking for from this matrix. They will facilitate the creation of the desired diagrams in this restricted space (the number of factors chosen), while minimizing the general configuration of the individuals according to all the initial variables.

### 1.2- Weight Metrics

The use of the weight metric in the space of real variables ***F*** gives a very particular meaning to the usual notions defined on Euclidean spaces. The statistical interpretations of the properties are defined below:

- The empirical mean of ***X<sup>j</sup>*** is a static calculated from a sample of data on one or more random data
- The Barycenter of individuals is the center of gravity or the point that allows to reduce certain linear combinations of vectors.
- The Matrix of centered data ***X*** is a constant added to each column so that the mean is zero.
- The Standard Deviation is important information when comparing the dispersion of two sets of data of similar size that have approximately the same mean
- Covariance of ***X<sup>j</sup>*** and ***X<sup>k</sup>*** is slightly different from the variance. If the variance allows to study the variations of a variable in relation to itself. The covariance will allow to study the simultaneous variations of two variables in relation to their respective mean.
- Covariance matrix ***S*** : These values ​​show the magnitude and direction of the distribution of multivariate data in a multidimensional space.
- Correlation of ***X<sup>j</sup>*** and ***X<sup>k</sup>*** is a relationship between two things, two notions, two facts, one of which implies the other and vice versa.

### 1.3- Objectives

- Table ***X*** can be analyzed through its rows (individuals) or through its columns (variables)
- summarize the information while keeping in mind the duality
- Typology of individuals
- there is a variability of temperatures (as an example) between individuals
- form groups of similar individuals
- Key terms : resemblance
- Typology of variables
- there are variables linked to each other
- form groups of linked variables
- Key terms : link – correlation
- Duality : visualize the groups of variables with the most inter-individual variability
- The ***optimal*** graphical representation of individuals (rows), minimizing the deformations of the point cloud, in a subspace ***E<sub>q</sub>*** of dimension ***q (q < p)***
- The graphical representation of variables in a subspace ***F<sub>q</sub>*** by best explaining the initial links between these variables

### 1.4- Basic notions

We consider a data table ***X*** where ***X*** is a matrix with ***n x p*** numerical values ​​where ***n*** individuals are described on ***p*** variables

<picture>
<center>
<img alt="Matrice X" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/images/Matricex.png" width=60% height=60%  title="Matrice X"/>
</center>
</picture>

It should be noted :

***X=(x<sub>ij</sub>)<sub>n X p</sub>*** the raw data matrix where <img alt="x R" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/images/xR.png"  title="Mx R"/> is the value of the i th individual on the j th variable

| <img alt="x R" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/images/xiR.png"  width=50% height=50% title="Mx R"/> | <img alt="x R" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/images/xjR.png" width=50% height=50% title="Mx R"/> |
| --- |  --- |
| The description of the i th individual (line of X) | The description of the j th individual (column of X) |


