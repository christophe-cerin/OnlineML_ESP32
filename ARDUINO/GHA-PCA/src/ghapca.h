#ifndef GHAPCA_H
#define GHAPCA_H

/*
   Author : M. SOW
   Date : 2025-02-19
   Objective : Performs the GHA update, calculating and updating the eigenvalues and eigenvectors of the data in real time. Eigen manages memory dynamically 
   Name of program : ghapca.h 
   
*/

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <map>
#include <algorithm>
#include <random>
#include <iomanip>
#include <numbers>
#include <string>

using Eigen::VectorXd;
using Eigen::MatrixXd;

using namespace std;
using namespace Eigen;

void ghapca_C(Eigen::MatrixXd &Q,
              const Eigen::VectorXd &x,
              const Eigen::VectorXd &y,
              const Eigen::VectorXd &gamma) {
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


#endif // GHAPCA_H
