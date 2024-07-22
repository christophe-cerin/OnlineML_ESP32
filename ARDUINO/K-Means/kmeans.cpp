/* 

Created : 2024, July 2024
Implemented by : M. SOW
Dataset from CampusIOT ELSYS

k-means clustering is the task of finding groups of 
pointrs in a dataset such that the total variance within
groups is minimized.
 
--> find argmin(sum(xi - ci)^2)

algorithm:

1. init the clusters

iterations {
    2. assign each point to the nearest centroid
    3. redefine the cluster
}

*/

#include <ctime>   
#include <fstream> 
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>  
#include <cfloat> 

using namespace std;

struct Point {
    int x, y;
    int cluster;
    double minDistance;

    Point(int _x, int _y) {
        x = _x;
        y = _y;
        cluster = -1;
        minDistance = DBL_MAX;
    }

    double distance(Point p) {
        return pow((this->x - p.x), 2) + pow(this->y - p.y, 2);
    }
};

vector<Point> readCSV(string path) {
    vector<Point> points;
    string line;
    ifstream file(path);

    getline(file, line); // pop header
    while (getline(file, line)) {
        stringstream lineStream(line);

        double x, y;
        string bit;
        getline(lineStream, bit, ',');
        getline(lineStream, bit, ',');
        getline(lineStream, bit, ',');
        getline(lineStream, bit, ',');
        x = stof(bit);
        getline(lineStream, bit, '\n');
        y = stof(bit);
        
        points.push_back(Point(x, y));
    }

    file.close();
    return points;
}

void kMeansClustering(vector<Point> &points, int epochs, int k) {
    
    // 1. init centroids
    vector<Point> centroids;
    srand(time(0)); // need to set the random seed
    int numOfPoints = points.size();
    for (int i=0; i<k; i++) {
        //int pointIdx = rand() % numOfPoints;
        int pointIdx = i;
        centroids.push_back(points.at(pointIdx));
        centroids.back().cluster = i;
    }

    // do some iterations
    for (int e=0; e<epochs; e++) {

        // 2. assign points to a cluster
        for (auto &point : points) {
            point.minDistance = DBL_MAX;
            for (int c=0; c<centroids.size(); c++) {
                double distance = point.distance(centroids[c]);
                if (distance < point.minDistance) {
                    point.minDistance = distance;
                    point.cluster = c;
                }
            }
        }

        // 3. redefine centroids
        vector<int> sizeOfEachCluster(k, 0);
        vector<double> sumXOfEachCluster(k, 0);
        vector<double> sumYOfEachCluster(k, 0);
        for (auto point : points) {
            sizeOfEachCluster[point.cluster] += 1;
            sumXOfEachCluster[point.cluster] += point.x;
            sumYOfEachCluster[point.cluster] += point.y;
        }
        for (int i=0; i<centroids.size(); i++) {
            centroids[i].x = (sizeOfEachCluster[i] == 0) ? 0 : sumXOfEachCluster[i] / sizeOfEachCluster[i];
            centroids[i].y = (sizeOfEachCluster[i] == 0) ? 0 : sumYOfEachCluster[i] / sizeOfEachCluster[i];
        }

        // 4. write to a file
        ofstream file1;
        file1.open("points_iter_" + to_string(e) + ".csv");
        file1 << "x,y,clusterIdx" << endl;
        for (auto point : points) {
            file1 << point.x << "," << point.y << "," << point.cluster << endl;
        }
        file1.close();
        
        ofstream file2;
        file2.open("centroids_iter_" + to_string(e) + ".csv");
        file2 << "x,y,clusterIdx" << endl;
        for (auto centroid : centroids) {
            file2 << centroid.x << "," << centroid.y << "," << centroid.cluster << endl;
        }
        file2.close();

    }
    
}

int main() {
    // load csv
    vector<Point> points = readCSV("./icluster1dataTime5120w512headItimeClusterCo2Temp.csv");
    
    //Column nUmber x=5, research cluster y=4
    kMeansClustering(points, 5, 4);

   // line by line reading of the results obtained

   string ligne, maligne;

   for(int i=0; i<4; i++) //mall_customer.csv
   {
     // create an output stream
     std::ostringstream oss;
     // Ecrire la variable dans le flux de sortie
     oss << i;
     // get a string
     std::string nomfichier = "points_iter_" + oss.str() + ".csv";
     std::ifstream fichier(nomfichier.c_str(), ios::in | ios::binary); 

     if(!fichier)
     {
       cout << "get a string" << " " << nomfichier  << "in reading mode" << "" << endl;
     }
       while(getline(fichier, ligne))
       {
         maligne += ligne;
         //cout << maligne << "\n" << endl;
         cout << ligne << "\n" << endl;
       }
   }

}
