## Technique of Dimension Reduction 

### [1- Generalized Hebbian  Algorithm  (GHA)](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/GHA/README.md)

### [2- Principal Component Analysis (PCA) or Neural Factor Analysis (NFA)](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/PCA/README.md)

### [3- Conversion the Original Python Code for Reduction Dimension with GHA algoritm to C++](#conversion)

- [Eigen : C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [Eigen 3 Documentation](https://eigen.tuxfamily.org/dox/index.html)
- [Matplotlib for C++ Documentation](https://matplotlib.org/3.1.1/index.html)
- [Overview](#overview)
- [Code Source](#code)
- [Compilation](#compilation)
- [Result](#result)
- [License](#license)

### [4- Data mining from the Perret Tower with the GHA dimension reduction algorithm](https://github.com/madou-sow/OnlineML_ESP32/tree/main/ARDUINO/GHA-PCA/Perret-Tower)

##### Overview

Eigen is a C++ numerical analysis library composed of template headers, developed by Benoît Jacob and Gaël Guennebaud at INRIA. It is free software under MPL2 license and multiplatform.

Matplotlib is for C++, a C++ wrapper for Python’s matplotlib (MPL) plotting library. Thus, to learn more about the functions that are eventually called the matplotlib documentation might be useful. Most functions have a link to the MPL function they call. Matplotlib for C++ requires a working Python installation as well as Matplotlib. Python2.7 and Python3 (>= 3.6) have been tested, but other versions should work as well. In the linking process the exact version of Python to use can be specified by linking the according library.

We have transformed the original Python code into C++ while maintaining the structure and logic of the algorithm. The code utilizes the Eigen library for matrix operations, which is a popular choice in C++ for linear algebra tasks. This C++ code retains the core functionality of the original Python code while adapting it to the syntax and conventions of C++.

Random Seed Initialization : We set the random seed using std::mt19937 to ensure reproducibility, similar to np.random.seed(123) in Python.

Data Generation : The synthetic multivariate data is generated using a normal distribution. We create a matrix data of size n x p and fill it with random values.

Parameter Initialization : The parameters for the GHA algorithm are initialized. We create a vector gamma to represent the learning rate, similar to the Python code.

Eigenvalues and Eigenvectors Initialization : We initialize lambda_values and U (the eigenvectors). The eigenvectors are normalized by dividing each column by its norm.

Centering Vector : The mean of each column is computed to center the data, analogous to the Python implementation.

GHA Algorithm Iteration : The loop iterates over each observation in the dataset. The GHA algorithm is intended to be applied here, but the actual function call is commented out, as it requires a specific implementation of ghapca in C++.

Results Output : Finally, the updated eigenvalues and eigenvectors are printed to the console.


##### [Code Source](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/online_GhaPca_update.cpp)

```C++
 
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense> //Using Eigen library for matrix operations
#include <cmath>
#include <map>
#include <algorithm>
#include <random>
#include <iomanip>
#include <numbers>
#include <string>
#include "/home/mamadou/src/matplotlibcpp.h"
#include </usr/include/python3.10/pyconfig.h>
#include "/usr/include/python3.10/Python.h"

/*
   Author : M. SOW
   Date : 2024-12-16
   Objective : Performs the GHA update, calculating and updating the eigenvalues and eigenvectors of the data in real time
   Name of program : online_GhaPca_update.cpp 
 */


using Eigen::VectorXd;
using Eigen::MatrixXd;

using namespace std;
using namespace Eigen;
namespace plt = matplotlibcpp;


void ghapca_C(Eigen::MatrixXd &Q,
	       	const Eigen::VectorXd &x,
	       	const Eigen::VectorXd &y,
	       	const Eigen::VectorXd &gamma) {
    /*
    Update the matrix Q based on vectors x, y, and gamma.
    
    Parameters:
    Q : Eigen::MatrixXd
        The matrix to be updated.
    x : Eigen::VectorXd
        The input vector.
    y : Eigen::VectorXd
        The projected vector.
    gamma : Eigen::VectorXd
        The learning rate vector.
    
    Returns:
    Eigen::MatrixXd
        The updated matrix Q.
    */
    int n = Q.rows();
    int k = Q.cols();
    
    // Update Q based on x, y, and gamma
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            Q(j, i) += gamma(i) * (x(j) - Q(j, i) * y(i));
        }
    }
}

std::map<std::string, Eigen::MatrixXd> ghapca(Eigen::VectorXd &lambda_values,
	       	Eigen::MatrixXd &U,
	       	Eigen::VectorXd &x,
	       	Eigen::VectorXd &gamma,
	       	int q = -1,
	       	Eigen::VectorXd center = Eigen::VectorXd(),
	       	bool sort = true) {
    /*
    Perform online PCA update.
    
    Parameters:
    lambda_values : Eigen::VectorXd
        The eigenvalues.
    U : Eigen::MatrixXd
        The eigenvectors.
    x : Eigen::VectorXd
        The new data point.
    gamma : double or Eigen::VectorXd
        The learning rate.
    q : int
        The number of principal components to keep.
    center : Eigen::VectorXd
        The center to subtract from x.
    sort : bool
        Whether to sort the eigenvalues and eigenvectors.
    
    Returns:
    std::map<std::string, Eigen::MatrixXd>
        A dictionary with keys 'values' and 'vectors' for eigenvalues and eigenvectors.
    */
    int d = U.rows();
    int k = U.cols();
    
    if (x.size() != d) {
        throw std::invalid_argument("Length of x must be equal to the number of rows in U.");
    }
    
    if (lambda_values.size() != 0 && lambda_values.size() != k) {
        throw std::invalid_argument("Length of lambda must be equal to the number of columns in U.");
    }
  
    if (center.size() != 0) {
         x -= center;
    }
    
    if (gamma.size() != k) {
        gamma.resize(k);
    }
    
    Eigen::VectorXd y = U.transpose() * x;
    
    ghapca_C(U, x, y, gamma);
    
    if (lambda_values.size() != 0) {
	    lambda_values = lambda_values.array() - gamma.array() * lambda_values.array() + gamma.array() * y.array().square(); 
        if (sort) {
            Eigen::VectorXi ix = Eigen::VectorXi::LinSpaced(k, 0, k - 1);
            std::sort(ix.data(), ix.data() + k, [&lambda_values](int a, int b) { return lambda_values(a) > lambda_values(b); });
            lambda_values = lambda_values(ix);
            U = U.colwise().normalized(); 
        }
        if (q != -1 && q < k) {
            lambda_values = lambda_values.head(q);
        }
    } else {
        lambda_values.resize(0);
    }
    
    if (q != -1 && q < k) {
        U = U.leftCols(q);
    }
    
    return {{"values", lambda_values}, {"vectors", U}};
}

int main() {
    // Set the random seed for reproducibility
    srand(123);
    
    // Generate some synthetic multivariate data for testing
    int n = 100;  // Number of observations
    int p = 5;    // Number of variables
    Eigen::MatrixXd data = Eigen::MatrixXd::Random(n, p);
    cout << data << endl;
    
    // Initialize parameters for the GHA algorithm
    int q = 2;  // Number of principal components to find
    Eigen::VectorXd gamma = Eigen::VectorXd::Constant(q, 1.0 / n);  // Learning rate (gain parameter)
    
    // Initialize eigenvalues and eigenvectors
    Eigen::VectorXd lambda_values = Eigen::VectorXd::Zero(q);  // Initial eigenvalues
    Eigen::MatrixXd U = Eigen::MatrixXd::Random(p, q);  // Initial eigenvectors (random initialization)
    U = U.colwise().normalized();  // Normalize eigenvectors
    
    // Centering vector (mean of each column)
    Eigen::VectorXd center = data.colwise().mean();
    
    // Apply the GHA algorithm iteratively to each data point
    for (int i = 0; i < n; i++) {
        Eigen::VectorXd x = data.row(i).transpose();
        auto gha_result = ghapca(lambda_values, U, x, gamma, q, center, true);
        lambda_values = gha_result["values"];
        U = gha_result["vectors"];
    }
    
    // Print the results
    cout << "Updated Eigenvalues:" << endl;
    cout << lambda_values << endl;
    
    cout << "Updated Eigenvectors:" << endl;
    cout << U << endl;
    
    // Project data onto the new principal components
    Eigen::MatrixXd scores = data * U;
    
    // Plotting would require additional libraries in C++
    // For simplicity, we will skip the plotting part in this conversion.
    
    std::vector<double> x_scores(scores.rows()), y_scores(scores.rows());
    for (int i = 0; i < scores.rows(); ++i) {
        x_scores[i] = scores(i, 0);
        y_scores[i] = scores(i, 1);
    }
    plt::scatter(x_scores, y_scores);
    plt::xlabel("Principal Component 1");
    plt::ylabel("Principal Component 2");
    plt::title("GHA - First Two Principal Components");
    plt::show();


    return 0;
}
```

##### Compilation

```
g++ online_GhaPca_update.cpp -I /home/mamadou/src -L /usr/include/python3.10 -lpython3.10 -o online_GhaPca_update.out
```

##### Result : online_GhaPca_update.out

```
 -0.879897  0.0326784  -0.687509   0.632997   0.106838
  0.576637   0.758171  -0.749769   0.299552  -0.696591
 -0.593863  -0.887224   0.408263  -0.293873 -0.0689467
 -0.302874  -0.245837  -0.437873  -0.871702  -0.727846
 -0.276781   0.861116  -0.112126   0.853755   0.485436
 -0.730723   0.981627  0.0259075    0.79015  -0.483624
 -0.248064    -0.7824  -0.696495   0.926268 -0.0256122
 -0.481357  -0.312428   -0.29503  -0.183068   0.311605
 -0.911367  -0.623203  -0.687443  -0.373332  -0.184518
  0.759123 0.00437989    0.71649  -0.251378   0.933366
  0.260731   -0.85192    0.50923   0.613713   0.541877
  -0.24571  -0.274201  -0.143439   0.582104   0.598969
 -0.360543   0.818578  -0.853478  -0.529095   0.220639
  0.655716   0.926521   -0.87314  -0.837833   -0.47579
 -0.149791   0.753394  -0.451846  -0.753945   0.961791
-0.0267368   0.426235  0.0238528  -0.208246  -0.122953
   0.58139  -0.820134   0.721715  -0.280173  -0.473814
   0.91054   0.807709    0.42463    0.75996  -0.331246
  0.427513    0.58835  -0.633578     0.7706    0.60402
 -0.295925   0.345441   0.875546   -0.92845  -0.267564
  0.683881  -0.173793   0.918069  -0.652042  0.0621528
  -0.27524   0.890232     0.5896   0.094952  -0.583502
 -0.233959   0.430251   0.698635   -0.95916 -0.0662414
 -0.843027   0.163854  -0.742467  -0.732417  -0.164974
 -0.651873  -0.962704   0.141436  -0.235192   0.962414
  0.760998   -0.98206  -0.940567  -0.298568  -0.779496
 -0.562676   0.581752  -0.215143   0.184629  -0.613461
  0.937463   -0.56792  -0.775858   0.342293  0.0180004
  0.585311   0.415371   0.687508  0.0149374   0.536777
 -0.992993    0.11028   0.823505   0.790799   0.906009
 -0.645034  -0.770939  -0.742469   0.693039  -0.714241
  0.705413  -0.551951   0.999998  -0.352066  -0.356385
  0.583644  -0.131548  -0.926264  0.0903505  -0.790583
 -0.238898  -0.658164   0.665795  -0.600834   0.216812
  -0.59746   0.202212  -0.437874  -0.223767 -0.0842304
 -0.693138  -0.270432 -0.0383906 -0.0558947   0.694853
 0.0303796  -0.676537  -0.308298  -0.810684   0.733189
  0.154476   0.419812  -0.134369  -0.297499   0.890157
 -0.174494    0.41714    0.66658   0.761037 0.00645798
  0.119012   -0.29974 0.00425913  -0.184016   -0.45133
-0.0864013  -0.575808   -0.41788   0.451123   0.823524
 -0.913763    0.56522   0.175809    0.37475  -0.451665
  0.873302   0.426059   0.860821  -0.601913  -0.852361
  0.553056   -0.75723  -0.271358   0.922028   0.044163
  0.741953   0.491741   0.302669   0.536918  0.0725445
 -0.276489   0.179454  -0.591026  -0.355857   -0.89057
 -0.473681   0.669005   0.752495  -0.286218    0.92121
  0.323343   0.671607  0.0243835  -0.743255   0.598731
 -0.365948 -0.0128369   0.833604  -0.595898  -0.221816
  0.953832   0.257355  -0.881083  -0.515618    0.52523
 -0.972582  0.0170477   -0.10007  -0.671705  -0.668833
 -0.682068   0.813371   0.751673  -0.247939   0.840337
 -0.321407   0.147588   0.708516   0.579334   0.941728
 -0.206541  -0.552702  -0.401435  -0.630866   0.264925
 -0.525095 -0.0227752  -0.990794  0.0196441  -0.324637
   0.02672   0.184884  -0.150048  -0.655858   0.904142
 -0.445543  -0.534762  -0.342002  0.0705664   0.485429
 -0.087771  -0.441023  -0.205937  -0.795726  0.0619021
-0.0358169   0.616964  0.0740947   0.686435 -0.0778574
 -0.860232   0.880609  -0.654494  -0.914496  0.0222061
-0.0807643   0.669257  -0.382433   0.995072 -0.0320893
  0.319149   0.846024   0.331626   0.379474   0.207901
  0.845181  -0.671342  -0.654496  -0.266562   0.665821
 -0.497121  -0.462291  -0.308697  0.0854228   0.177328
 -0.919749   -0.81214  -0.002579    0.77864  -0.575286
 -0.752279    0.53087 -0.0923703   0.509671  -0.418409
 -0.190258   0.267277   0.652912  -0.970472  -0.127819
  0.110631  -0.488677   0.689123   0.967956  -0.842098
  0.402196 -0.0493176    0.77326  -0.787828  -0.528252
  0.635247  -0.315584   0.319492   0.790565   0.878639
 -0.770357   0.211583  -0.306618   -0.21606  -0.293428
 -0.684205   0.374874  -0.644619   0.663295  -0.704728
  0.721484  -0.750363  -0.504699   0.165316  -0.573026
 -0.897055  -0.362357  -0.445797   0.182027  -0.145788
  0.868851   0.617644  0.0840232   0.585323   0.339435
  0.463437   0.741378    0.79797  -0.297766   0.499518
 -0.173544   0.817096 -0.0368227    0.82617 -0.0363586
  -0.60483   0.286649  -0.163482  -0.700895   0.260645
  -0.21322   0.412985  -0.177646 -0.0410215   0.098249
  0.460508  -0.195741  -0.203219  -0.769728   0.741825
 -0.650998  -0.455995 -0.0445647  -0.216513  -0.214125
 -0.185802  -0.569968   0.722283   0.287273   0.429416
   0.77844   -0.38237  -0.451546 -0.0176675   0.582162
  0.027595   0.691592  -0.336048  -0.637178  -0.272397
  0.607657  -0.122669  -0.679152   0.656407  -0.305659
 -0.746655   0.594855   -0.44234  -0.998023  -0.742475
 -0.945685  -0.123524   0.513904  -0.293037  -0.368255
 -0.837885   0.342569 -0.0211536  -0.273026  -0.820229
  0.165574  -0.846168   0.351723   -0.79375   0.319427
 0.0184981  -0.506561  -0.412001  -0.606601   0.553888
 -0.698118   0.223178   0.324352  -0.187522   0.201977
  -0.91519   0.823089    0.96929  -0.798678  -0.712662
 -0.662353  -0.660536   0.919625   0.772873  -0.238211
 -0.852937   0.551836   0.669856   0.545916  -0.132202
 -0.412311  -0.639202  -0.339407   0.286745   0.464666
 -0.582102  -0.472676  -0.082954   0.551513   0.186503
 -0.605216  0.0827066  -0.422515  0.0555866   0.449389
  0.397431   0.628074  -0.686495   0.316273  -0.663153
  0.528528  0.0386477  -0.393831   0.519469   0.344405
  0.796981  -0.966611  -0.649254   0.267758   0.921138
Updated Eigenvalues:
0.202424
 0.19352
Updated Eigenvectors:
 -0.520435   0.403791
 -0.630568  -0.623871
 -0.521185   0.631786
 -0.236521  -0.133458
-0.0628941   0.175425

```

<picture>
<center>
<img alt="GHA - First Two Principal Components" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/images/fig-onlinegpapca.png" width=55% height=55%  title="GHA - First Two Principal Components"/>
</center>
</picture>
