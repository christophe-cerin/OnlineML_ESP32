#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <string>
#include <omp.h>

#include "matplotlibcpp.h"
#include "/usr/include/python3.10/Python.h"

using namespace std;
using namespace Eigen;
namespace plt = matplotlibcpp;

/*
   Author : M. SOW
   Date : 2025-06-16
   Objective : The provided Python code online_lasso.py has been converted into C++ while maintaining the original logic and structure
   Name of programm : online_lasso_regression_optimized_en.cpp
*/

struct Point {
    double x, y;
    Point(double x = 0.0, double y = 0.0) : x(x), y(y) {}
};

bool compare_points(const Point& a, const Point& b) {
    return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}

MatrixXd load_csv(const string& path) {
    ifstream file(path);
    vector<vector<double>> data;
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> row;
        string cell;
        
        while (getline(ss, cell, ',')) {
            try {
                row.push_back(stod(cell));
            } catch (...) {
                continue;
            }
        }
        if (!row.empty()) data.push_back(row);
    }

    MatrixXd mat(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i)
        for (size_t j = 0; j < data[i].size(); ++j)
            mat(i, j) = data[i][j];
    
    return mat;
}

void parallel_quicksort(vector<Point>& arr, int left, int right) {
    if (left >= right) return;

    int i = left, j = right;
    Point pivot = arr[(left + right) / 2];

    while (i <= j) {
        while (compare_points(arr[i], pivot)) ++i;
        while (compare_points(pivot, arr[j])) --j;
        if (i <= j) swap(arr[i++], arr[j--]);
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        if (left < j) parallel_quicksort(arr, left, j);
        #pragma omp section
        if (i < right) parallel_quicksort(arr, i, right);
    }
}

class OptimizedLasso {
    double lr, l1;
    int max_iter;
    VectorXd w;
    double b;

public:
    OptimizedLasso(double lr = 0.01, int max_iter = 1000, double l1 = 1.0)
        : lr(lr), max_iter(max_iter), l1(l1) {}

    void fit(const MatrixXd& X, const VectorXd& y, double tol = 1e-5) {
        int m = X.rows(), n = X.cols();
        w = VectorXd::Zero(n);
        b = 0;
        VectorXd dw_prev = VectorXd::Ones(n);
        double db_prev = 1.0;

        for (int iter = 0; iter < max_iter; ++iter) {
            VectorXd y_pred = X * w + VectorXd::Constant(m, b);
            VectorXd error = y_pred - y;

            VectorXd dw = (X.transpose() * error) / m;
            double db = error.sum() / m;

            for (int j = 0; j < n; ++j) {
                if (w[j] > 0) dw[j] += l1 / m;
                else if (w[j] < 0) dw[j] -= l1 / m;
                if (abs(dw[j]) < 1e-10) dw[j] = (dw[j] >= 0) ? 1e-10 : -1e-10;
            }

            w -= lr * dw;
            b -= lr * db;

            if (dw.norm() < tol && abs(db) < tol) {
                cout << "Convergence after " << iter << " iterations\n";
                break;
            }
        }
    }

    VectorXd predict(const MatrixXd& X) const {
        return X * w + VectorXd::Constant(X.rows(), b);
    }

    double score(const MatrixXd& X, const VectorXd& y) const {
        VectorXd y_pred = predict(X);
        double mse = (y - y_pred).array().square().sum() / y.size();
        return isfinite(mse) ? mse : 0.0;
    }

    // Added accessors for w and b
    VectorXd get_weights() const { return w; }
    double get_bias() const { return b; }
};

tuple<MatrixXd, MatrixXd, VectorXd, VectorXd> train_test_split(
    const MatrixXd& X, const VectorXd& y, double test_size = 0.3) {
    
    vector<int> indices(y.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), default_random_engine{});

    int test_samples = test_size * y.size();
    MatrixXd X_train(X.rows() - test_samples, X.cols());
    MatrixXd X_test(test_samples, X.cols());
    VectorXd y_train(y.size() - test_samples);
    VectorXd y_test(test_samples);

    #pragma omp parallel for
    for (int i = 0; i < indices.size(); ++i) {
        if (i < test_samples) {
            X_test.row(i) = X.row(indices[i]);
            y_test[i] = y[indices[i]];
        } else {
            X_train.row(i - test_samples) = X.row(indices[i]);
            y_train[i - test_samples] = y[indices[i]];
        }
    }

    return {X_train, X_test, y_train, y_test};
}

int main() {
    const int W = 4096;
    const int N = 10;
    const int sqrt_W = static_cast<int>(sqrt(W));

    MatrixXd data = load_csv("TourPerret10col.csv");
    cout << "Data loaded : " << data.rows() << "x" << data.cols() << endl;

    if (data.rows() == 0 || data.cols() < 3) {
        cerr << "Error: Insufficient Data or Incorrect Format" << endl;
        return 1;
    }

    vector<Point> points(min(W*N, (int)data.rows()));
    #pragma omp parallel for
    for (int i = 0; i < points.size(); ++i) {
        points[i] = Point(data(i, 0), data(i, 2));
    }

    vector<Point> my_points(points.begin(), points.begin() + min(W, (int)points.size()));
    parallel_quicksort(my_points, 0, my_points.size()-1);

    vector<int> sample_indices;
    for (int i = sqrt_W-1; i < W-sqrt_W; i += sqrt_W) {
        sample_indices.push_back(i);
    }

    for (int i = 0; i < N-1; ++i) {
        #pragma omp parallel for
        for (int k = 0; k < sample_indices.size(); ++k) {
            int idx = (i+1)*W + k;
            if (idx < points.size() && sample_indices[k] < my_points.size()) {
                my_points[sample_indices[k]] = points[idx];
            }
        }

        if (i == N-2) {
            MatrixXd X(my_points.size(), 1);
            VectorXd y(my_points.size());
            #pragma omp parallel for
            for (int j = 0; j < my_points.size(); ++j) {
                X(j, 0) = my_points[j].x;
                y[j] = my_points[j].y;
            }

            double x_mean = X.mean();
            double x_std = sqrt((X.array() - x_mean).square().sum() / X.size());
            X = (X.array() - x_mean) / x_std;

            double y_mean = y.mean();
            double y_std = sqrt((y.array() - y_mean).square().sum() / y.size());
            y_std = (y_std < 1e-10) ? 1.0 : y_std;
            y = (y.array() - y_mean) / y_std;

            auto [X_train, X_test, y_train, y_test] = train_test_split(X, y, 0.3);

            OptimizedLasso model(0.01, 1000, 1.0);
            model.fit(X_train, y_train);

            VectorXd y_pred = model.predict(X_test);
            double mse = model.score(X_test, y_test);
            cout << "Score MSE: " << mse << endl;

            cout << "\nExamples of Predictions vs Real Values :\n";
            for (int k = 0; k < min(10, (int)y_test.size()); ++k) {
                cout << "Prediction : " << y_pred[k] 
                     << " | Actual : " << y_test[k] 
                     << " | Error : " << abs(y_pred[k] - y_test[k]) << endl;
            }

            // Using accessors instead of private members
            VectorXd weights = model.get_weights();
            double bias = model.get_bias();

            vector<double> x_plot(X_test.data(), X_test.data() + X_test.rows());
            vector<double> y_actual(y_test.data(), y_test.data() + y_test.rows());
            vector<double> y_pred_vec(y_pred.data(), y_pred.data() + y_pred.rows());

            plt::figure_size(1200, 800);
            plt::scatter(x_plot, y_actual, 15, {{"label", "Real Values"}, {"color", "blue"}});
            plt::scatter(x_plot, y_pred_vec, 15, {{"label", "Predictions"}, {"color", "red"}});
            
            vector<double> reg_line_x = {*min_element(x_plot.begin(), x_plot.end()), 
                                      *max_element(x_plot.begin(), x_plot.end())};
            vector<double> reg_line_y = {
                weights[0] * reg_line_x[0] + bias,
                weights[0] * reg_line_x[1] + bias
            };
            plt::plot(reg_line_x, reg_line_y, {{"label", "Regression Line"}, {"color", "green"}, {"linewidth", "2"}});

            plt::title("Regression LASSO - Real Values Predictions");
            plt::xlabel("X (standardized)");
            plt::ylabel("Y (standardized)");
            plt::grid(true);
            plt::legend();
            
            plt::save("regression_lasso_optimized.png");
            plt::show();
        }

        parallel_quicksort(my_points, 0, my_points.size()-1);
    }

    cout << "\nProcessing Completed Successfully" << endl;
    return 0;
}
