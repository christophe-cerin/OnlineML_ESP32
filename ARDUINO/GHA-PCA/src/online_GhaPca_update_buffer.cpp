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
#include <fstream>
#include <sstream>
#include <cassert> // Addition for checks

/*
   Author : M. SOW
   Date : 2025-02-19
   Objective : Performs the GHA update, calculating and updating the eigenvalues and eigenvectors of the data in real time. 
   The program will process the data in blocks of 1024 lines of the Perret Tower data, display the iteration number at each step, 
   and update the eigenvalues and eigenvectors accordingly. At the end, the results will be saved in a CSV file and displayed graphically.
   Name of programm : online_GhaPca_update_buffer.cpp 
*/

#include "ghapca.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

using namespace std;
using namespace Eigen;

#include "/home/mamadou/src/matplotlibcpp.h"
#include </usr/include/python3.10/pyconfig.h>
#include "/usr/include/python3.10/Python.h"
namespace plt = matplotlibcpp;

MatrixXd readCSV(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file");
    }

    vector<vector<double>> data;
    string line;
    int expected_cols = -1; // Variable to check the consistency of the number of columns
    
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;
        
        // cout << "Reading line: " << line << endl; // Debugging
        
        while (getline(ss, cell, ',')) {
            try {
                double value = stod(cell);
                row.push_back(value);
         // cout << "  Parsed value: " << value << endl; // Displaying converted values
            } catch (const exception &e) {
                cerr << "Error converting: " << cell << " - " << e.what() << endl;
                return MatrixXd(); // Return an empty matrix on error
            }
        }
        
        if (!row.empty()) {
            if (expected_cols == -1) {
                expected_cols = row.size(); // Set the number of columns from the first line read
            } else if (row.size() != expected_cols) {
                cerr << "Inconsistent column count: expected " << expected_cols << " but got " << row.size() << endl;
                return MatrixXd(); // Return empty matrix on inconsistency 
            }
            data.push_back(row);
        }
    }

    file.close();

    int rows = data.size();
    int cols = rows > 0 ? data[0].size() : 0;
    cout << "CSV Dimensions: " << rows << " x " << cols << endl;

    if (rows == 0 || cols == 0) {
        throw runtime_error("CSV file appears to be empty or incorrectly formatted.");
    }

    MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        assert(data[i].size() == cols);
        for (int j = 0; j < cols; ++j) {
            mat(i, j) = data[i][j];
        }
    }

    return mat;
}

int main() {
    // Read data from CSV file
    MatrixXd data = readCSV("TourPerret10col.csv");

    // Display data
    cout << "Data from CSV:" << endl;
    cout << data << endl;

    int n = data.rows(); // Number of observations
    int p = data.cols(); // Number of variables

    // Initialize parameters for the GHA algorithm
    int q = 2;  // Number of principal components to find
    Eigen::VectorXd gamma = Eigen::VectorXd::Constant(q, 1.0 / n);  // Learning rate (gain parameter)

    // Initialize eigenvalues and eigenvectors
    Eigen::VectorXd lambda_values = Eigen::VectorXd::Zero(q);  // Initial eigenvalues
    Eigen::MatrixXd U = Eigen::MatrixXd::Random(p, q);  // Initial eigenvectors (random initialization)
    U = U.colwise().normalized();  // Normalize eigenvectors

    // Centering vector (mean of each column)
    Eigen::VectorXd center = data.colwise().mean();

    // Buffer size
    const int W = 1024;

    // Iterate over the data in blocks of W rows
    for (int i = 0; i < n; i += W) {
        int block_size = min(W, n - i);  // Handle the last block which might be smaller than W
        cout << "Iteration " << (i / W + 1) << " processing rows " << i << " to " << i + block_size - 1 << endl;

        for (int j = 0; j < block_size; ++j) {
            Eigen::VectorXd x = data.row(i + j).transpose();
            auto gha_result = ghapca(lambda_values, U, x, gamma, q, center, true);
            lambda_values = gha_result["values"];
            U = gha_result["vectors"];
        }
    }

    // Print the results
    cout << "Updated Eigenvalues:" << endl;
    cout << lambda_values << endl;

    cout << "Updated Eigenvectors:" << endl;
    cout << U << endl;

    // Project data onto the new principal components
    Eigen::MatrixXd scores = data * U;

    // Save the scores to a CSV file for plotting
    ofstream outfile("scores.csv");
    if (outfile.is_open()) {
        outfile << scores << endl;
        outfile.close();
    } else {
        cerr << "Unable to open file for writing scores." << endl;
    }

    // Graphics
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

    return EXIT_SUCCESS;
}
