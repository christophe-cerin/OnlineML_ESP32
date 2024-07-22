/*

Created : 2024, July 2024
Implemented by : M. SOW
Dataset from CampusIOT ELSYS

Multivariate linear regression with SGD algorithm:

1. estimate coefficients with SGD
    1.1 init weights
    1.2 get loss
    1.3 update weights and bias
        -- bias = bias - lr * (y_hat - y)
        -- weights = weight - lr * (y_hat - y) * x
2. make predicitons
*/

#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

double predict(
    const vector<double> &xs, 
    const vector<double> &coefficents) {
    
    double prediction = coefficents[0];
    for (int i=0; i<xs.size(); i++) {
        prediction += coefficents[i+1] * xs[i];
    }
    return prediction;
}

vector<double> estimateCoefficientsWithSGD(
    const vector<vector<double>> trainSet, 
    const double lr, const int numOfEpochs) {
    
    // init weights
    int numOfCoefficents = (trainSet[0].size() - 1) + 1;
    vector<double> coefficents(numOfCoefficents, 0);
    
    for (int e=0; e<numOfEpochs; e++) {
        double sumOfError = 0;
        for (auto pair : trainSet) {
            vector<double> x = {pair[0]};
            double prediction = predict(x, coefficents);
            double error = pow(pair[1] - prediction, 2);
            sumOfError += error;
            
            // update
            //// bias = bias - lr * (y_hat - y)
            coefficents[0] = coefficents[0] - lr * (prediction - pair[1]);
            //// weights = weight - lr * (y_hat - y) * x
            for (int i=1; i<numOfCoefficents; i++) {
                coefficents[i] = coefficents[i] - lr * (prediction - pair[1]) * pair[0];
            }
        }
        
        cout << "epoch: " << e << ", error: " << sumOfError << endl;
    }
    
    return coefficents;
}

int main() {

    // Nombre de lignes à traiter max 83 lignes, la première colonne doit être l'index
    vector<vector<double>> trainSet = {
			{1,264},
			{2,269},
			{3,262},
			{4,266},
			{5,252},
			{6,276},
			{7,259},
			{8,270},
			{9,261},
			{10,264},
			{11,261},
			{12,266},
			{13,262},
			{14,256},
			{15,263},
			{16,250},
			{17,253},
			{18,256},
			{19,267},
			{20,276},
			{21,263},
			{22,268},
			{23,269},
			{24,251},
			{25,281},
			{26,273},
			{27,257},
			{28,277},
			{29,257},
			{30,262},
			{31,260},
			{32,265},
			{33,259},
			{34,278},
			{35,257},
			{36,271},
			{37,256},
			{38,258},
			{39,268},
			{40,268},
			{41,277},
			{42,249},
			{43,267},
			{44,253},
			{45,268},
			{46,259},
			{47,266},
			{48,267},
			{49,266},
			{50,282},
			{51,258},
			{52,271},
			{53,258},
			{54,274},
			{55,258},
			{56,269},
			{57,265},
			{58,275},
			{59,255},
			{60,266},
			{61,252},
			{62,268},
			{63,256},
			{64,277},
			{65,263},
			{66,281},
			{67,263},
			{68,267},
			{69,253},
			{70,271},
			{71,258},
			{72,272},
			{73,261},
			{74,275},
			{75,263},
			{76,260},
			{77,276},
			{78,260},
			{79,273},
			{80,264},
			{81,283},
			{82,263}
		};    



    // vector<double> coefficents = {0.4, 0.8};
    double lr = 0.001;
    int numOfEpochs = 50;
    vector<double> coefficents = estimateCoefficientsWithSGD(trainSet, lr, numOfEpochs);
    cout << "coefficents = {" << coefficents[0] << ", " << coefficents[1] << "}"<< endl;
    
    for (auto pair : trainSet) {
        cout << predict({pair[0]}, coefficents) << endl;
    }
}
