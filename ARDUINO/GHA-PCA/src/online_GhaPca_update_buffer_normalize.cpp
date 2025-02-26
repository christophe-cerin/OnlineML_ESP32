#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>

#include "/home/mamadou/big-data/cerin04102024/DimensionalityReductionCerin/pyTocpp/lib/matplotlibcpp.h"
#include </usr/include/python3.10/pyconfig.h>
#include "/usr/include/python3.10/Python.h"


#include "ghapca.h"


/*
   Author : M. SOW
   Date : 2025-02-26
   Objective : Performs GHA update, calculates and updates the eigenvalues ​​and eigenvectors of the data in real time.
   The program will process the data in blocks of 1024 lines of data from the Perret Tower, display the iteration 
   number at each step, and update the eigenvalues ​​and eigenvectors accordingly. At the end, the results will be saved 
   in a CSV file and displayed graphically. In this program there is a special feature because a section where the data is normalized. 
   This practice allows transforming the data without distorting it. While standardization consists of harmonizing the data 
   so that all the entries of the different data sets that relate to the same terms follow a similar format.
   Name of program : online_GhaPca_update_buffer_normalize.cpp 
*/

using namespace std;
using namespace Eigen;
using Eigen::VectorXd;
using Eigen::MatrixXd;
namespace plt = matplotlibcpp;


// Function to normalize data
MatrixXd normalize(const MatrixXd& data) {
    VectorXd min_vals = data.colwise().minCoeff();
    VectorXd max_vals = data.colwise().maxCoeff();

    // Ignore columns 1, 3 and 6 when normalizing
    for (int i = 0; i < min_vals.size(); ++i) {
        if (i == 1 || i == 3 || i==6) continue;  // Ignore columns 1, 3 and 6
        if (max_vals(i) == min_vals(i)) {
            cerr << "Error: max_vals == min_vals for column " << i << ". Unable to normalize ." << endl;
            exit(1);
        }
    }

    MatrixXd normalized = data;
    for (int i = 0; i < data.cols(); ++i) {
        if (i == 1 || i == 3 || i ==6) continue;  // Ignore columns 1, 3 and 6
        normalized.col(i) = (data.col(i).array() - min_vals(i)) / (max_vals(i) - min_vals(i));
    }

    return normalized;
}

// Function to load data from a CSV file
MatrixXd load_csv(const string& path) {
    ifstream file(path);
    vector<vector<double>> data;
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> row;
        string cell;
        int col_index = 0;

        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
            col_index++;
        }

        data.push_back(row);
    }

    MatrixXd mat(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[i].size(); ++j) {
            mat(i, j) = data[i][j];
        }
    }

    return mat;
}

int main() {
    // Load data from CSV file
    MatrixXd data = load_csv("TourPerret10col.csv");
    cout << "Data loaded : \n" << data.topRows(6635)  << endl;

    int Nrows = data.rows();
    int p = data.cols();

    // Normalize data
    data = normalize(data);
    cout << "Normalized data :\n" << data.topRows(Nrows) << endl;

    // Initialize settings
    int q = 2;  // Number of principal components
    VectorXd gamma = VectorXd::Constant(q, 0.01);  // Learning rate
    VectorXd lambda_values = VectorXd::Zero(q);
    MatrixXd U = MatrixXd::Random(p, q);
    U = U.array().rowwise() / U.colwise().norm().array();

    // Check settings
    if ((gamma.array() <= 0).any()) {
        cerr << "Error : gamma must contain positive values." << endl;
        exit(1);
    }

    if (U.hasNaN()) {
        cerr << "Erreur : U contains NaNs." << endl;
        exit(1);
    }

    if (lambda_values.hasNaN()) {
        cerr << "Error : lambda_values contains NaN." << endl;
        exit(1);
    }

    // Center the data
    VectorXd center = data.colwise().mean();

    // Buffer size
    const int W = 1024;

     // Iterate over the data in blocks of W rows
    for (int i = 0; i < Nrows; i += W) {
        int block_size = min(W, Nrows - i);  // Handle the last block which might be smaller than W
        cout << "Iteration " << (i / W + 1) << " processing rows " << i << " to " << i + block_size - 1 << endl;




	   // Apply the GHA algorithm
		   for (int j = 0; j < block_size; ++j) {
				VectorXd x = data.row(i + j).transpose();
				auto gha_result = ghapca(lambda_values, U, x, gamma, q, center, true);
				lambda_values = gha_result["values"];
				U = gha_result["vectors"];
		
				// Check the results
				if (lambda_values.hasNaN()) {
				    cerr << "Error: lambda_values ​​contains NaNs during iteration " << i << "." << endl;
				    exit(1);
				}
		
				if (U.hasNaN()) {
				    cerr << "Error: U contains NaNs during iteration " << i << "." << endl;
				    exit(1);
				}
	    }

    }
    // Show results
    cout << "Updated eigenvalues :\n" << lambda_values << endl;
    cout << "Updated Eigenvectors :\n" << U << endl;

    // Projection of data on new principal components
    MatrixXd scores = data * U;

    // Show scores
    cout << "Scores of the first two principal components :\n" << scores.topRows(Nrows) << endl;

    // Save the scores to a CSV file for plotting
    ofstream outfile("scores.csv");
    if (outfile.is_open()) {
        outfile << scores << endl;
        outfile.close();
    } else {
        cerr << "Unable to open file for writing scores." << endl;
    }



    // graphics
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
