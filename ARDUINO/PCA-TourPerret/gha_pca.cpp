#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

/*

Created : 2025, August 2025
Implemented by : M. SOW

The Generalized Hebbian Algorithm (GHA), also known as Sanger's Rule, is a "neural network model for unsupervised learning" that can perform Principal Component Analysis (PCA). 
Unlike traditional PCA which often relies on eigenvalue decomposition, GHA is an "iterative, online algorithm". 
This means it can update its weights with each new data point, making it suitable for large datasets or streaming data.
Its core idea is based on the "Hebbian learning rule" ("neurons that fire together, wire together"), extended to allow for the extraction of multiple principal components in a "sequential and orthogonal manner". 
Each principal component is learned by subtracting the variance explained by the previously learned components, ensuring they are uncorrelated.

How it works conceptually :
- Imagine the algorithm trying to find directions (principal components) in your data that capture the most variance.
- It starts with random guesses for these directions.
- Then, for each data point, it slightly adjusts its guesses. If a data point aligns strongly with a guess, that guess is reinforced (moved closer to the data point).
- The "generalized" part ensures that each new direction found is independent of the ones already found, effectively making them orthogonal.
- This iterative process continues until the directions converge to the true principal components of the data.


1. This C++ code implements a dimensionality reduction technique similar to Principal Component Analysis (PCA) using the Generalized Hebbian Algorithm (GHA). 
It reads specific columns from a CSV file, normalizes the data, applies GHA to reduce dimensions, and then saves the projected data to a new CSV file for visualization.

2. Dataset from CampusIOT ELSYS, TourPerrethead11col.csv


*/



// Function to read data from CSV file
MatrixXd readCSV(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error : Unable to open file" << filename << endl;
        exit(1);
    }
    string line;
    getline(file, line); // Ignore header
    vector<vector<double>> data;
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;
        // We read columns 1, 3, 5, 6 (accMotion, humidity, temperature, vdd)
        for (int i = 0; getline(ss, cell, ','); ++i) {
            if (i == 1 || i == 3 || i == 5 || i == 6) {
                row.push_back(stod(cell));
            }
        }
        data.push_back(row);
    }
    MatrixXd mat(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            mat(i, j) = data[i][j];
        }
    }
    return mat;
}

// Function to normalize data (mean and standard deviation)
MatrixXd normalizeData(const MatrixXd& data) {
    MatrixXd normalized = data;
    VectorXd mean = normalized.colwise().mean();
    MatrixXd centered = normalized.rowwise() - mean.transpose();
    VectorXd stddev = (centered.array().square().colwise().sum() / (centered.rows() - 1)).sqrt();
    for (int i = 0; i < centered.cols(); ++i) {
        centered.col(i) /= stddev(i);
    }
    return centered;
}

// Implementation of the GHA algorithm
MatrixXd GHA(const MatrixXd& data, int numComponents, double learningRate, int numIterations) {
    int numFeatures = data.cols();
    MatrixXd W = MatrixXd::Random(numFeatures, numComponents); // Random initialization of weights
    W.normalize(); // Weight normalization

    for (int iter = 0; iter < numIterations; ++iter) {
        for (int i = 0; i < data.rows(); ++i) {
            VectorXd x = data.row(i).transpose();
            for (int j = 0; j < numComponents; ++j) {
                VectorXd wj = W.col(j);
                VectorXd residual = x;
                for (int k = 0; k < j; ++k) {
                    residual -= (W.col(k).dot(x)) * W.col(k);
                }
                W.col(j) += learningRate * (residual.dot(wj)) * residual;
                W.col(j).normalize(); // Weight normalization
            }
        }
        learningRate *= 0.99; // Decrementing the learning rate
    }
    return W;
}

int main() {
    // 1. Data loading and preprocessing
    MatrixXd data = readCSV("TourPerrethead11col.csv");
    MatrixXd normalizedData = normalizeData(data);

    // 2. Dimension reduction with GHA
    int numComponents = 2; // 2-dimensional reduction for visualization
    double learningRate = 0.01;
    int numIterations = 1024;
    
    MatrixXd W = GHA(normalizedData, numComponents, learningRate, numIterations);
    
    // 3. Projecting data onto new dimensions
    MatrixXd projectedData = normalizedData * W;

    // 4. Saving projected data for plotting
    ofstream outFile("projected_data.csv");
    if (outFile.is_open()) {
        outFile << "PC1,PC2" << endl;
        for (int i = 0; i < projectedData.rows(); ++i) {
            outFile << projectedData(i, 0) << "," << projectedData(i, 1) << endl;
        }
        outFile.close();
        cout << "Projected data saved in 'projected_data.csv'" << endl;
    } else {
        cerr << "Error: Unable to write to 'projected_data.csv'" << endl;
    }

    return 0;
}
