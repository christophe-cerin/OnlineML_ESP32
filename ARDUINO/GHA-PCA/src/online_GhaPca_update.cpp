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
#include "matplotlibcpp.h"
#include </usr/include/python3.10/pyconfig.h>
#include "/usr/include/python3.10/Python.h"

/*
 
   Created by : Mamadou SOW
   Date : 2024-12-16
   Objective : Performs the GHA update, calculating and updating the eigenvalues and eigenvectors of the data in real time
   Name of programm : online_GhaPca_update.cpp 
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

