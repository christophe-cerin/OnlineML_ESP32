/*

Created : 2024, July 2024
Implemented by : M. SOW
Dataset from CampusIOT ELSYS

Simple linear regression algorithm:

1. calculate mean and variance
2. calculate covariance
3. estimate coefficients (b1=cov(x,y)/var(x), b0=mean(y)-b1*mean(x))
4. make predicitons

Note: 
sum(y_i - y_mean) = sum(y_i) - sum(y_mean) = 0
sum((x_i - x_mean) * (y_i * y_mean)) = sum(x_i * (y_i * y_mean))
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <climits>
#include <cfloat>
#include <algorithm>

using namespace std;

double getMean(vector<double> values) {
    double sum = 0;
    for (auto value : values) {
        sum += value;
    }
    return sum / values.size();
}

double getVariance(vector<double> values) {
    double mean = getMean(values);
    double variance = 0;
    for (auto value : values) {
        variance += pow(value - mean, 2);
    }
    return variance;
}

double getCovariance(vector<double> valuesA, vector<double> valuesB) {
    double meanA = getMean(valuesA);
    double meanB = getMean(valuesB);
    double covariance = 0;
    for (int i=0; i<valuesA.size(); i++) {
        covariance += (valuesA[i] - meanA) * (valuesB[i] - meanB);
    }
    return covariance;
}

vector<double> getCoefficients(const vector<vector<double>> &dataset) {
    // get xs and ys
    vector<double> xs;
    vector<double> ys;
    for (auto pair : dataset) {
        xs.push_back(pair[0]);
        ys.push_back(pair[1]);
    }
    
    // calculate coefficients
    double b1 = getCovariance(xs, ys) / getVariance(xs);
    double b0 = getMean(ys) - b1 * getMean(xs);
    
    return {b0, b1};
}

vector<double> myGetCoefficients(const vector<vector<double>> &dataset) {
    // get xs and ys
    vector<double> xs;
    vector<double> ys;
    for (auto pair : dataset) {
        xs.push_back(pair[0]);
        ys.push_back(pair[1]);
    }
    
    // calculate coefficients
    double numeratro = 0;
    for (int i=0; i<dataset.size(); i++) {
        numeratro += dataset[i][0] * (ys[i] - getMean(ys));
    }
    double denuminator = 0;
    for (int i=0; i<dataset.size(); i++) {
        denuminator += dataset[i][0] * (xs[i] - getMean(xs));
    }
    
    double b1 = numeratro / denuminator;
    double b0 = getMean(ys) - b1 * getMean(xs);
    
    return {b0, b1};
}

double predict(double x, vector<double> coefficents) {
    return x * coefficents[1] + coefficents[0];
}

double evaluation(
    const vector<vector<double>> &dataset, 
    const vector<double> &predictions) {
    
    // rmse
    double sum = 0;
    for (int i=0; i<dataset.size(); i++) {
        sum += pow(dataset[i][1] - predictions[i], 2);
    }
    return sqrt(sum / dataset.size());
}

int main() {

	vector<vector<double>> trainSet = {
		{264,20},
		{269,21.7},
		{262,20.1},
		{266,21.6},
		{252,20},
		{276,21.6},
		{259,20},
		{270,21.6},
		{261,20},
		{264,21.6},
		{261,21.6},
		{266,20},
		{262,21.6},
		{256,20},
		{263,21.5},
		{250,20},
		{253,20},
		{256,21.6},
		{267,20},
		{276,21.6},
		{263,20},
		{268,20},
		{269,21.5},
		{251,20.1},
		{281,21.6},
		{273,21.7},
		{257,20.3},
		{277,21.8},
		{257,20.4},
		{262,21.8},
		{260,20.3},
		{265,21.9},
		{259,20.4},
		{278,21.9},
		{257,20.4},
		{271,21.9},
		{256,20.5},
		{258,20.5},
		{268,22},
		{268,20.5},
		{277,22.1},
		{249,20.5},
		{267,22.1},
		{253,20.5},
		{268,22.1},
		{259,20.5},
		{266,22.1},
		{267,22.1},
		{266,20.5},
		{282,22.2},
		{258,20.6},
		{271,22.2},
		{258,20.5},
		{274,22.1},
		{258,20.6},
		{269,22.2},
		{265,20.6},
		{275,22.2},
		{255,20.6},
		{266,22.2},
		{252,20.6},
		{268,22.2},
		{256,20.5},
		{277,22.2},
		{263,20.6},
		{281,22.2},
		{263,20.6},
		{267,22.3},
		{253,20.7},
		{271,22.3},
		{258,20.7},
		{272,22.3},
		{261,20.7},
		{275,22.3},
		{263,22.4},
		{260,20.7},
		{276,22.4},
		{260,20.8},
		{273,22.5},
		{264,20.7},
		{283,22.5},
		{263,20.7},
		{266,22.5},
		{265,20.7},
		{265,22.4},
		{278,22.5},
		{264,22.6},
		{264,22.7},
		{263,22.7},
		{254,20.8},
		{267,22.7},
		{273,22.7},
		{266,22.8},
		{266,22.7},
		{266,20.9},
		{255,22.7},
		{253,21},
		{268,22.7},
		{267,21},
		{270,22.7},
		{279,22.7},
		{267,21},
		{271,22.7},
		{252,20.9},
		{262,22.6},
		{261,22.7},
		{278,22.6},
		{280,22.6},
		{267,22.6},
		{272,22.6},
		{258,20.9},
		{277,22.6},
		{274,22.5},
		{276,22.5},
		{267,20.8},
		{262,22.4},
		{257,20.8},
		{277,22.4},
		{268,22.3},
		{270,22.3},
		{267,20.7},
		{274,22.2},
		{267,20.7},
		{279,22.2},
		{260,20.6},
		{271,22.2},
		{260,20.6},
		{271,22.2},
		{257,20.5},
		{262,20.6},
		{267,22.1},
		{263,20.6},
		{261,22.1},
		{260,20.6},
		{273,22.2},
		{271,22.2},
		{271,20.6},
		{279,22.1},
		{258,20.5},
		{266,22.1},
		{276,22.1},
		{261,20.5},
		{256,20.5},
		{268,22.1},
		{252,20.5},
		{269,22.1},
		{256,20.5},
		{273,22.1},
		{265,20.5},
		{273,22},
		{261,20.4},
		{274,22.1},
		{263,20.5},
		{277,22.1},
		{256,20.5},
		{258,22.1},
		{268,20.4},
		{265,22.1},
		{261,20.4},
		{289,22.1},
		{259,20.4},
		{285,22.1},
		{280,22.1},
		{259,20.4},
		{268,22.1},
		{259,20.4},
		{281,22.1},
		{256,20.4},
		{266,22.1},
		{257,20.5},
		{277,22.2},
		{262,20.5},
		{275,22.2},
		{258,20.5},
		{283,22.2},
		{269,20.4},
		{281,22.2},
		{257,20.4},
		{269,22.2},
		{250,20.4},
		{269,22.2},
		{257,20.5},
		{275,22.2},
		{261,20.5},
		{274,22.2},
		{258,20.5},
		{273,22.2},
		{270,20.5},
		{268,22.2},
		{277,22.2},
		{264,20.4},
		{272,22.2},
		{269,20.3},
		{266,22.2},
		{281,22.2},
		{257,20.3},
		{275,22.1},
		{245,20.3},
		{279,22.1},
		{275,20.3},
		{273,22.1},
		{255,20.3},
		{268,22.1},
		{267,20.3},
		{261,22.1},
		{261,20.3},
		{281,22.1},
		{265,20.3},
		{284,22.1},
		{285,22.1},
		{260,20.3},
		{256,20.4},
		{276,22.1},
		{274,22.1},
		{257,20.4},
		{250,22.2},
		{268,22.2},
		{263,20.4},
		{262,22.2},
		{281,22.2},
		{271,20.4},
		{285,22.3},
		{253,20.4},
		{279,22.3},
		{281,20.4},
		{274,22.3},
		{257,20.4},
		{270,22.3},
		{261,22.3},
		{272,20.4},
		{272,22.3},
		{252,20.4},
		{263,22.3},
		{271,22.3},
		{258,20.4},
		{275,22.4},
		{267,20.4},
		{269,22.4},
		{277,22.4},
		{265,20.4},
		{266,20.4},
		{263,22.4},
		{276,22.4},
		{267,22.3},
		{269,20.4},
		{273,22.3},
		{258,20.4},
		{271,22.4},
		{264,20.3},
		{284,22.4},
		{277,20.4},
		{268,22.4},
		{259,20.4},
		{264,20.4},
		{265,22.3},
		{259,20.4},
		{274,22.3},
		{267,22.3},
		{271,20.4},
		{277,22.3},
		{280,22.3},
		{251,22.3},
		{285,22.3},
		{266,20.4},
		{269,22.3},
		{278,22.3},
		{275,20.4},
		{267,22.3},
		{259,20.4},
		{257,22.2},
		{269,20.4},
		{282,22.3},
		{272,22.2},
		{271,22.3},
		{279,22.2},
		{273,22.3},
		{267,20.3},
		{265,22.2},
		{261,22.2},
		{278,20.3},
		{258,20.2},
		{263,22.2},
		{264,20.2},
		{265,22.2},
		{264,20.3},
		{278,22.2},
		{273,20.2},
		{275,22.2},
		{250,20.2},
		{272,22.1},
		{290,22.1},
		{265,20.2},
		{261,22.1},
		{266,20.2},
		{270,22.2},
		{265,20.2},
		{266,22.1},
		{255,20.2},
		{261,22.1},
		{269,20.2},
		{273,22.1},
		{268,20.2},
		{271,22.1},
		{281,20.2},
		{267,22.1},
		{273,20.1},
		{281,22.1},
		{269,20.1},
		{281,22.2},
		{265,20.1},
		{257,22.1},
		{262,20.1},
		{273,22.1},
		{264,20.1},
		{276,20.1},
		{279,22.1},
		{264,20.1},
		{278,22.2},
		{272,20.1},
		{271,22.1},
		{267,20.1},
		{276,22.1},
		{257,20.1},
		{264,22.1},
		{270,20.1},
		{247,22.2},
		{264,20.1},
		{280,22.1},
		{276,20.1},
		{274,22.2},
		{281,22.1},
		{271,20.1},
		{267,22.2},
		{283,20.1},
		{271,22.1},
		{274,22.1},
		{258,20.1},
		{265,22.1},
		{271,22.2},
		{268,20.1},
		{269,22.1},
		{267,22.2},
		{272,20.1},
		{271,22.2},
		{280,22.1},
		{268,20},
		{275,22.2},
		{272,22.2},
		{277,22.1},
		{270,20},
		{273,22.1},
		{285,19.9},
		{264,22.2},
		{268,20},
		{273,22.1},
		{261,22.1},
		{273,19.9},
		{276,22.1},
		{281,22.1},
		{279,20},
		{271,22.2},
		{268,20},
		{273,22.2},
		{271,20.1},
		{270,22.2},
		{279,20},
		{266,22.2},
		{278,22.2},
		{265,22.1},
		{279,20},
		{276,22.2},
		{274,20.1},
		{271,22.2},
		{274,20},
		{268,22.3},
		{272,20.1},
		{270,22.2},
		{271,22.3},
		{278,20},
		{270,22.4},
		{264,20.1},
		{263,20.1},
		{258,22.4},
		{271,20.1},
		{260,20.1},
		{259,22.4},
		{259,22.5},
		{272,20.2},
		{267,22.5},
		{274,22.5},
		{269,22.5},
		{275,22.6},
		{270,22.7},
		{280,20.2},
		{262,22.6},
		{274,20.3},
		{272,22.6},
		{263,20.2},
		{258,22.6},
		{275,20.3},
		{262,22.6},
		{268,20.2},
		{272,22.6},
		{262,22.6},
		{277,22.6},
		{264,20.3},
		{281,22.6},
		{273,22.6},
		{277,22.6},
		{263,20.3},
		{267,20.3},
		{292,22.5},
		{254,20.3},
		{269,22.5},
		{273,20.3},
		{265,22.6},
		{274,20.3},
		{272,22.5},
		{268,20.4},
		{277,22.5},
		{272,20.4},
		{276,22.5},
		{267,20.4},
		{289,22.5},
		{267,20.4},
		{274,22.5},
		{272,20.4},
		{270,20.4},
		{270,22.5},
		{276,20.4},
		{278,22.5},
		{277,20.4},
		{280,22.5},
		{275,20.4},
		{268,22.5},
		{265,20.4},
		{278,22.5},
		{266,20.4},
		{272,22.5},
		{264,20.4},
		{286,22.5},
		{275,20.3},
		{278,22.5},
		{269,20.3},
		{262,22.5},
		{274,20.3},
		{266,22.5},
		{270,20.3},
		{289,22.5},
		{281,20.4},
		{279,22.5},
		{258,20.3},
		{283,22.5},
		{269,20.4},
		{265,22.5},
		{282,20.4},
		{275,22.5},
		{270,22.5},
		{285,22.5},
		{274,22.5},
		{263,20.4},
		{275,22.5},
		{267,20.3},
		{298,22.5},
		{266,20.3},
		{278,22.5},
		{277,22.5},
		{261,20.3},
		{273,22.5},
		{270,20.3},
		{282,22.5},
		{277,20.3},
		{274,22.5},
		{267,20.3},
		{284,22.5},
		{267,20.3},
		{294,22.5},
		{267,20.3},
		{291,22.5},
		{286,22.5},
		{279,20.3},
		{275,22.5},
		{267,20.3},
		{279,22.5},
		{282,22.5},
		{283,20.3},
		{299,22.5},
		{286,20.3},
		{287,20.2},
		{285,22.5},
		{277,20.3},
		{284,22.5},
		{264,20.3},
		{266,22.5},
		{277,20.3},
		{284,22.5},
		{285,20.3},
		{286,22.5},
		{283,20.3},
		{286,22.5},
		{267,20.3},
		{289,22.5},
		{265,20.3},
		{278,22.5},
		{277,20.3},
		{270,20.3},
		{284,22.5},
		{283,22.5},
		{261,20.3},
		{274,22.5},
		{268,20.2},
		{286,22.4}
		};
	
	auto coefficents = getCoefficients(trainSet);    
	cout << "coefficents = {" << coefficents[0] << ", " << coefficents[1] << "}"<< endl;
	vector<double> predictions;
	for (auto pair : trainSet) {
		predictions.push_back(predict(pair[0], coefficents));
	}
        double rms = evaluation(trainSet, predictions);
    	cout << rms << endl;	    
		
}