## Principal Component Analysis (PCA) or Neural Factor Analysis (NFA)

PCA aims to return to a limited dimensional space by distorting the possibilities of reality as little as possible. The objective is therefore to obtain the most appropriate synthesis possible of the preliminary information.

Indeed, PCA involves the calculation of a limited number of new dimensions, which represent the linear combinations of the initial dimensions of the data (i.e. descriptive characteristics). These new dimensions do not present any correlation and reflect the greatest variance of the information (based on the mean). In other words, it is a factorial technique of dimension reduction used for the statistical analysis of complex quantitative data.

The eigenvectors, ranked by decreasing eigenvalues, belong to the new axes of the information covariance matrix. These are the main axes of dispersion of the data cloud, ranked in decreasing order of importance. The corresponding eigenvalues ​​designate the proportion of variance expressed by each axis. Thus, the first axes generally produce the majority of the variance. The new data values ​​for each axis were identified as the principal components.

This technique can serve as a dual purpose data compression method and exploration tool in highly multidimensional domains. Thus, the calculation of principal axes thus performed not only facilitates the reduction of information, but also the interpretation of the domain in question, since the new measurements are usually extremely large.

### 1- Mathematical Design
#### 1.1- Data in PCA

In PCA, the data are presented in a table X with n rows and p columns where
  
  - each row represents an individual
  - each column represents a variable

The variables have a quantitative nature: the matrix X is composed of numerical values. It is the variance-covariance matrix (or that of correlations) that will allow an appropriate summary to be made, because it focuses mainly on the distribution of the data in question.

By an appropriate mathematical process, we will extract in small numbers the factors that we are looking for from this matrix. They will facilitate the creation of the desired diagrams in this restricted space (the number of factors chosen), while minimizing the general configuration of the individuals according to all the initial variables.
