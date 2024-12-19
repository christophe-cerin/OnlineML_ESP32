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

| <img alt="x R" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/images/xiR.png"  width=58% height=58% title="Mx R"/> | <img alt="x R" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/images/xjR.png" width=50% height=50% title="Mx R"/> |
| --- |  --- |
| The description of the i th individual (line of X) | The description of the j th individual (column of X) |

### 2- Graphs

It is the interpretation of the graphs that will allow us to understand the structure of the analyzed data. This interpretation will be guided by a certain number of numerical and graphic indicators, called interpretation aids, which are there to help the user make the most accurate and objective interpretation possible.

### 2.1- Individuals

The graphs obtained make it possible to represent “as best as possible” the Euclidean distances between individuals measured by the metric ***M***.

### 2.2- Variables

The graphs obtained allow to represent “at best” the correlations between the variables (cosines of the angles) and, if these are not reduced, their variances (lengths).

### 2.3- Biplot

This remark allows to interpret two other graphical representations in PCA projecting simultaneously individuals and variables.

- the isometric line representation uses the matrices ***C*** and ***V***; it allows to interpret the distances between individuals as well as the scalar products between an individual and a variable which are, in the first principal plane, approximations of the observed values ​​***X<sup>j</sup> (ω<sub>i</sub>)***;
- the isometric column representation uses the matrices ***U*** and ***VΛ<sup>1/2</sup>***; it allows to interpret the angles between variable vectors (correlations) and the scalar products as previously.

### 3- Choice of dimension

The quality of the estimates that the PCA leads to depends, obviously, on the choice of ***q***, that is to say the number of components retained to reconstruct the data, or even on the dimension of the subspace re representation.

Many criteria for choosing ***q*** have been proposed in the literature. We present here the most common ones, based on a heuristic and one based on a quantification of the stability of the subspace of representation.

Other criteria, not explained, are inspired by statistical decision-making practices; under the hypothesis that the error admits a Gaussian distribution, we can exhibit the asymptotic laws of the eigenvalues ​​and therefore construct tests of nullity or equality of the latter. Unfortunately, in addition to the necessary hypothesis of normality, this leads to a procedure of nested tests whose overall level is uncontrollable. Their use therefore remains heuristic.

### 3.1- Inertia share

The “overall quality” of the representations is measured by the explained inertia share:
The value of ***q*** is chosen so that this explained inertia share rq is greater than a threshold value set a priori by the user. This is often the only criterion used.

### 3.2- Elbow

This is the graph showing the decrease in eigenvalues. The principle consists in searching, if it exists, for a “knee” (change of sign in the sequence of order 2 differences) in the graph and to keep only the eigenvalues ​​up to this knee. Intuitively, the larger the gap ***(λ<sub>q</sub> − λ<sub>q+1</sub>)***, for example greater than ***(λ<sub>q−1</sub> − λ<sub>q</sub>)***, the more we can be assured of the stability of ***E<sub>q</sub>***,

### 3.3- Box plots

A graph presenting, in parallel, the box plots of the main variables illustrates their qualities well: stability when a large box is associated with small whiskers, instability in the presence of a small box, large whiskers and isolated points. Intuitively, we keep the first “large boxes”. Isolated points or “outliers” designate points with a strong contribution, or potentially influential, in a main direction. They require a clinical study: another analysis in which they are declared additional (zero weights) in order to evaluate their impact on the orientation of the axes.

### 3.4- Stability

The presentation of the PCA, as a result of the estimation of a model, offers another approach to the problem of choosing a dimension. The quality of the estimates is usually evaluated in statistics by a quadratic mean risk defining a criterion of stability of the representation subspace. It is defined as the expectation of a distance between the “true” model and the estimate made of it. 
