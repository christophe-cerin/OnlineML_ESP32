## Online machine learning and Online technique with CampusIOT's LoRaWAN Datasets and the Kmeans Clustering Algorithm Implemented

Online machine learning is a machine learning method in which data becomes available in sequential order and is used to update 
the best predictor of future data at each step. The batch learning technique generates the best predictor by learning 
the entire training data in one go. Continuous learning means constantly improving the model by dealing with continuous 
flows of information being difficult since the continuous acquisition of progressively available products.

### Motivations and Challenges

Our goal is to design a model capable of learning from new data without revisiting past data, a model that is also robust 
to conceptual drift. This developed model will be very close to what happens in a production context, which is generally event-based. 
It will integrate well with the rest of the programming ecosystem for embedded systems.

Clustering is the target because online data is produced at low frequency from a small number of sensors. Smart building systems use 
sensors and data analytics to monitor energy consumption in real time, allowing building managers to optimize energy consumption 
and reduce costs.
The clustering application is a visualization tool that will simplify tasks such as controlling building temperature, 
maintaining equipment via mobile devices and computers.

### Tower Perret (Grenobles) : Data acquisition and exploitation by LoRaWAN

LoRaWAN : radio communication protocol that defines how terminal equipment communicates wirelessly via gateways, thus forming a low-consumption extended network

Endpoints: Endpoints are installed on four sides (SWW, SSE, NEE, NNW) of the top of the tower (in IP55 enclosures for Elsys EMS).
Data with 15 main parameters: The tourperret.log.gz file contains a dataset of 421937 messages received between June 2021 and June 2023 (2 years).
https://github.com/CampusIoT/datasets
https://perscido.univ-grenoble-alpes.fr/datasets/DS395
The field or parameter: The object contains Humidity and Temperature, the measurement of weather conditions at the top of the tower.
Weather conditions can be correlated with weather data from services such as OpenWeatherMap. The field or parameter: date contains the date and time with this example format “2021-01-01T00:05:28.165Z” =⇒ “20210101010528”

The data from the ELSYS-ERS sensor from January 1 at 00:05 a.m. to January 31 at 9:48 p.m. 2021, the quantity of C02, the Temperature, the date and the chronograph recorded every 5 minutes will be necessary to carry out interesting scientific work.

### Data analysis : Using data from Tour Perret, Local MQTT server and Arduino IDE

The CO<sub>2</sub> concentration and temperature will be published from our local MQTT server with the subjects "C02/celsius" and "final/final" the data is received via the ESP32 board and used by the Arduino IDE where a program has been implemented works to carry out clustering with the K-MEANS algorithm. We considered in our case, a structure as a means of defining this type of data although it is more than a simple container for primitive data types because we can also define other functionalities what it does with the elements or data points, it is a movement that meets the criteria at the end of the range. it then returns an iterator to the first element of the deleted elements (which are actually just moved).

The objective is to analyze the behavior of the ESP32 controller in the face of a stream of messages received continuously subject to an implemented algorithm containing both a clustering algorithm which makes it possible to identify a maximum of 4 clusters.

During the experiments, we managed to send several message streams (256, 512, 1024 and 5120) containing CO2 quantities and temperatures from the MQTT server to the ESP32 microcontroller.
Using our integrated Arduino development environment, we have carried out a distribution of the data to be processed by the microcontroller. This variable is called W. It varies from 8 to 512. 512 is the maximum value taking into account the saturation that can occur. 

```

/*

M. SOW
12/09/2023

ArduinoMqttClient - WiFi Simple Receive

This example connects to a MQTT broker and subscribes to a two topics.
When the messages are received it prints the messages to the Serial Monitor.

The circuit:
- ESP32, Arduino MKR 1000, MKR 1010 or Uno WiFi Rev2 board
- Open a Terminal and execute the program "test-29062021-broker-10.10.6.142-500points-Portable-Dell-pub.sh"
- We received a message, print out the topic and contents on the Serial Monitor of Arduino IDE 
*/

#include <ArduinoMqttClient.h>
#include <WiFi.h>
#include <PubSubClient.h> //Librairie pour la gestion Mqtt 

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <iostream> // std:cout
#include <list> //std::list
#include <iterator> // for back_inserter 
#include <string> //std::string, std::find
#include <vector> //vector
#include <random>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <bits/stdc++.h>



#include "src/simple_svg.hpp"


#ifdef USING_OMP
#include <src/omp.h>
#endif


char ssid[] = SECRET_SSID; // your network SSID (name)
char pass[] = SECRET_PASS; // your network password (use for WPA, or use as key for WEP)


WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);


const char broker[] = "10.10.6.228";//IP address

int port = 1883;

const char topic1[] = "co2/celsius";
const char topic2[] = "final/final";



#include "src/concurrentqueue.h"
#include "src/semaphore.h"

using namespace std;

struct Point {
double x, y; // coordinates
int cluster; // no default cluster
double minDist; // default infinite distance to nearest cluster

Point() : x(0.0), y(0.0), cluster(-1), minDist(__DBL_MAX__) {}
Point(double x, double y) : x(x), y(y), cluster(-1), minDist(__DBL_MAX__) {}

/**
* Computes the (square) euclidean distance between this point and another
*/
double distance(Point p) {
    return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
}

};

// Custom less comparator
bool lessPoints (const Point & lhs, const Point & rhs) {
    return (lhs.x < rhs.x) || ((lhs.x == rhs.x) && (lhs.y < rhs.y));
}

// Custom less or equal comparator
bool lessEqualPoints (const Point & lhs, const Point & rhs) {
    return (lhs.x <= rhs.x) || ((lhs.x == rhs.x) && (lhs.y <= rhs.y));
}

// Custom greater comparator
bool greaterPoints (const Point & lhs, const Point & rhs) {
    return (lhs.x > rhs.x) || ((lhs.x == rhs.x) && (lhs.y > rhs.y));
}

// Custom greater or equal comparator
bool greaterEqualPoints (const Point & lhs, const Point & rhs) {
    return (lhs.x >= rhs.x) || ((lhs.x == rhs.x) && (lhs.y >= rhs.y));
}

// Custom equal comparator
bool equalPoints (const Point & lhs, const Point & rhs) {
    return (lhs.x == rhs.x) && (lhs.y == rhs.y);
}

// A utility function to print an array
void printarr (Point arr[], int n) {
    for (int i = 0; i < n; ++i)
    printf ("%d -> %f, %f\n", i, arr[i].x, arr[i].y);
    printf ("\n");
}

/* Function to sort an array using insertion sort*/
void insertionSort (Point arr[], int n){
    int i, j;
    Point key = Point (0.0, 0.0);
    for (i = 1; i < n; i++){
        key = arr[i];
        j = i - 1;
        
        /* Move elements of arr[0..i-1], that are
        greater than key, to one position ahead
        of their current position */
        while (j >= 0 && greaterPoints (arr[j], key)) {
              swap (arr[j + 1], arr[j]);
              j = j - 1;
        }
        swap (arr[j + 1], key);
    }
}

/* This function partitions a[] in three parts
a) a[l..i] contains all elements smaller than pivot
b) a[i+1..j-1] contains all occurrences of pivot
c) a[j..r] contains all elements greater than pivot 
*/
void partition (Point arr[], int l, int r, int &i, int &j){
        i = l - 1, j = r;
        int p = l - 1, q = r;
        Point v = arr[r];
        
        while (true){
                // From left, find the first element greater than
                // or equal to v. This loop will definitely
                // terminate as v is last element
                while (lessPoints (arr[++i], v))
                ;
                
                // From right, find the first element smaller than
                // or equal to v
                while (lessPoints (v, arr[--j]))
                if (j == l)
                break;
                
                // If i and j cross, then we are done
                if (i >= j)
                break;
                
                // Swap, so that smaller goes on left greater goes
                // on right
                swap (arr[i], arr[j]);
                
                // Move all same left occurrence of pivot to
                // beginning of array and keep count using p
                if (equalPoints (arr[i], v)) {
                      p++;
                      swap (arr[p], arr[i]);
                }
                
                // Move all same right occurrence of pivot to end of
                // array and keep count using q
                if (equalPoints (arr[j], v)) {
                      q--;
                      swap (arr[j], arr[q]);
                }
        }
        
        // Move pivot element to its correct index
        swap (arr[i], arr[r]);
        
        // Move all left same occurrences from beginning
        // to adjacent to arr[i]
        j = i - 1;
        for (int k = l; k < p; k++, j--)
        swap (arr[k], arr[j]);
        
        // Move all right same occurrences from end
        // to adjacent to arr[i]
        i = i + 1;
        for (int k = r - 1; k > q; k--, i++)
        swap (arr[i], arr[k]);
}

// 3-way partition based quick sort
void quicksort (Point a[], int l, int r) {
      if (r <= l)
          return;
      if ((r - l + 1) <= 1024){ // dependant of the cache structure
          insertionSort (a, r - l + 1);
          return;
      }
      
      int i, j;
      
      // Note that i and j are passed as reference
      partition (a, l, r, i, j);
      
      // Recur -- in parallel
      #pragma omp task shared(a) firstprivate(l,j)
      { quicksort (a, l, j);}
      #pragma omp task shared(a) firstprivate(i,r)
      { quicksort (a, i, r);}
      #pragma omp taskwait
}

#define MAX_LEVELS 64

int quickSort_without_stack_or_recursion (Point arr[], int elements) {
        int beg[MAX_LEVELS], end[MAX_LEVELS], L, R;
        int i = 0;
        
        beg[0] = 0;
        end[0] = elements;
        while (i >= 0) {
                L = beg[i];
                R = end[i];
                if (R - L > 1){    
                          int M = L + ((R - L) >> 1);
                          Point piv = arr[M];
                          swap (arr[M], arr[L]);
                          
                          if (i == MAX_LEVELS - 1)
                          return -1;
                          R--;
                          while (L < R) {
                                while (greaterEqualPoints (arr[R], piv) && L < R)
                                R--;
                                if (L < R)
                                swap (arr[L++], arr[R]);
                                while (lessEqualPoints (arr[L], piv) && L < R)
                                L++;
                                if (L < R)
                                swap (arr[R--], arr[L]);
                          }
                          arr[L] = piv;
                          M = L + 1;
                          while (L > beg[i] && equalPoints (arr[L - 1], piv))
                          L--;
                          while (M < end[i] && equalPoints (arr[M], piv))
                          M++;
                          if (L - beg[i] > end[i] - M) {
                                  beg[i + 1] = M;
                                  end[i + 1] = end[i];
                                  end[i++] = L;
                          }
                          else
                          {
                                beg[i + 1] = beg[i];
                                end[i + 1] = L;
                                beg[i++] = M;
                          }
                }
                else {
                        i--;
                }
        }
        return 0;
}
//== Fin 



/**
* Perform k-means clustering
* @param points - pointer to vector of points
* @param epochs - number of k means iterations
* @param k - the number of initial centroids
*/

void kMeansClustering(vector<Point>*points, int epochs, int k) {
            int n = points->size();
            // Randomly initialise centroids
            // The index of the centroid within the centroids vector
            // represents the cluster label.
            vector<Point> centroids;
            srand(time(0));
            for (int i = 0; i < k; ++i) {
                  centroids.push_back(points->at(rand() % n));
            }
            
            for (int i = 0; i < epochs; ++i) {
                    // For each centroid, compute distance from centroid to each point
                    // and update point's cluster if necessary
                    for (vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c) {
                              int clusterId = c - begin(centroids);
                              
                              for (vector<Point>::iterator it = points->begin();
                              it != points->end(); ++it) {
                                    Point p = *it;
                                    double dist = c->distance(p);
                                    if (dist < p.minDist) {
                                            p.minDist = dist;
                                            p.cluster = clusterId;
                                    }
                                    *it = p;
                              }
                    }
                    
                    // Create vectors to keep track of data needed to compute means
                    vector<int> nPoints;
                    vector<double> sumX, sumY;
                    for (int j = 0; j < k; ++j) {
                            nPoints.push_back(0);
                            sumX.push_back(0.0);
                            sumY.push_back(0.0);
                    }
                    
                    // Iterate over points to append data to centroids
                    for (vector<Point>::iterator it = points->begin(); it != points->end(); ++it) {
                              int clusterId = it->cluster;
                              nPoints[clusterId] += 1;
                              sumX[clusterId] += it->x;
                              sumY[clusterId] += it->y;
                              
                              it->minDist = __DBL_MAX__; // reset distance
                    }
                    // Compute the new centroids
                    for (vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c) {
                              int clusterId = c - begin(centroids);
                              c->x = sumX[clusterId] / nPoints[clusterId];
                              c->y = sumY[clusterId] / nPoints[clusterId];
                    }
            }
            
            
            cout<<"The data points and the Centroids with their cluster-id : "<<endl;
            
            for (vector<Point>::iterator it = points->begin(); it != points->end(); ++it) {
                            cout << it->x << ", " << it->y << ", " << it->cluster << endl;
            }

}


/*
Be careful here, struct constitutes all the data points.
The struct is variable mycount, it is a multiple of 8. 
The total of data can be 8,16,32,64,..,256,...,512 or ...
*/

//struct Point arr[32];
//struct Point arr[64]; // 66 iterations co2 5120 points
struct Point arr[128]; // 13 iterations co2 5120 points
//struct Point arr[256]; // 1 iteration co2 5120 points
//struct Point arr[512];// 1 iteration co2 5120 points

int mycount = 0; //data point numbering variable mycount
//char my_str[256];
char my_str[1024];
//char my_str[5120];
//char my_str[10240];

int my_round = 1;
bool ok = true; 

void onMqttMessage(int messageSize) {
int i = 0;
while (mqttClient.available()) {
            my_str[i++] = (char)mqttClient.read();          
            my_str[i]= '\0';

if (!strcmp((const char *)my_str,(const char *)"0.0,0.0")) { 
            //Serial.print("End of the DATA reception, the name of data file is : "); 
            //Serial.println();
            return; 
} 
else {
            
            string s(my_str); 
            int pos = s.find(",");
            string sub1 = s.substr(0,pos); 
            
            string sub2 = s.substr(pos + 1); 

            int length = sub1.length(); 
            // declaring character array (+1 for null terminator) 
            char* char_array_sub1 = new char[length + 1]; 
            // copying the contents of the string to char array 
            strcpy(char_array_sub1, sub1.c_str()); 
            
            length = sub2.length(); 
            // declaring character array (+1 for null terminator) 
            char* char_array_sub2 = new char[length + 1]; 
            // copying the contents of the string to char array 
            strcpy(char_array_sub2, sub2.c_str()); 
            // atof() — Convert Character String to Float
            Point myPoint = Point(atof(char_array_sub1),atof(char_array_sub2));
            arr[mycount++] = myPoint;
            
            //if(mycount == 32){
            //if(mycount == 64){ 
            //if(mycount == 256){
            //if(mycount == 512){
            if(mycount == 128){
            printf("==========\n");
            printf("===== Calculation Turn Number = %d with 1024 vector points and with an iteration of 128 points=====\n",my_round++);
            printf("==========\n");
            
            vector<Point> points;
            for (int i = 0; i < sizeof(arr) / sizeof(arr[0]); i++) {
                    points.push_back(arr[i]);
            }
            
            
            
            //kMeansClustering(&points, 32, 4);
            //kMeansClustering(&points, 64, 4);            
            //kMeansClustering(&points, 128, 2);
            //kMeansClustering(&points, 256, 4);
            //kMeansClustering(&points, 512, 4);
            //kMeansClustering(&points, 512, 2);
            kMeansClustering(&points, 128, 4);
            
            
            
            // Tri rapide des données
            //Serial.println("Displaying a set of 128 data points after quicksorting");
            //quicksort(arr, 0, mycount -1);
            //printarr(arr, mycount);
            // == end : Display and Quick Data 
            
            //=== Debut Clear Buffer points 
            Point arr[mycount];
            std::copy(points.begin(), points.end(), arr); 
            
            int taille = (int) sqrt((double) mycount);
            int mysize = mycount - 1;
            
            for (int i = taille - 1; i < mysize; i += taille) {
                        auto it = std::find_if(points.begin(), points.end(), [&](const Point v) {
                        return equalPoints(v, arr[i]);}
                        );
                        
                        // if (mykmeans._Centers.end() == it) {
                                    if (points.end() == it) {
                                            auto remove_start = remove_if(points.begin(), points.end(), [&](const Point v) { 
                                                    return equalPoints(v, arr[i - 1]); 
                                            });
                                            points.erase(remove_start,points.end());
                                    }
                                    else {
                                            auto remove_start = remove_if(points.begin(), points.end(), [&](const Point v) { 
                                            return equalPoints(v, arr[i]); 
                                    });
                                            points.erase(remove_start,points.end());
                                    }
                        }
                        
                        //=== Fin Clear Buffer points
                        mycount=0;
            
            } 
            
            }
}


void setup() {
            //Initialize serial and wait for port to open:
            Serial.begin(115200);
            while (!Serial) {
                      ; // wait for serial port to connect. Needed for native USB port only
            }
            // attempt to connect to Wifi network:
            WiFi.begin(ssid, pass);
            while (WiFi.status() != WL_CONNECTED) {
                  delay (500 );
                  Serial.print ("Tentative de connection du Micro-Contrôleur ESP32 au Réseau Wifi : ");
                  Serial.println( ssid);
            }
            
            Serial.println("You're connected to the network");
            Serial.println();
            Serial.println(WiFi.macAddress());
            Serial.print("Adresse IP du Micro-Contrôleur ESP32 : ");
            Serial.println(WiFi.localIP());
            

            // You can provide a username and password for authentication
            // mqttClient.setUsernamePassword("username", "password");
            Serial.print("Attempting to connect to the MQTT broker: ");
            Serial.println(broker);
            
            if (!mqttClient.connect(broker, port)) {
                    Serial.print("MQTT connection failed! Error code = ");
                    Serial.println(mqttClient.connectError());
                    while (1);
            }
            
            Serial.println("You're connected to the MQTT broker!");
            Serial.println();
            
            // set the message receive callback
            mqttClient.onMessage( onMqttMessage );
            
            Serial.print("Subscribing to topic1 : ");
            Serial.println(topic1);
            Serial.println();
            
            Serial.print("Subscribing to topic2 : ");
            Serial.println(topic2);
            Serial.println(); 
            
            // subscribe to a topic
            mqttClient.subscribe(topic1);
            mqttClient.subscribe(topic2);
            
            // topics can be unsubscribed using:
            // mqttClient.unsubscribe(topic);
            Serial.print("Waiting for messages on topic1 : ");
            Serial.println(topic1);
            Serial.println();
            
            Serial.print("Waiting for messages on topic2 : ");
            Serial.println(topic2);
            Serial.println();
            //}
            
            // return 0;

}

void loop() {
        // call poll() regularly to allow the library to receive MQTT messages and
        // send MQTT keep alive which avoids being disconnected by the broker
        mqttClient.poll();

}

```

By referring to an equation with 3 unknowns, namely x (set of messages), y (splitting or subset of messages) and z (number of splits of messages processed), the satisfactory result corresponds to the case allowing the observation of a maximum of messages processed in a message set.

<img alt="Message Flow, Processing Slice and Number of Possible Iterations" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/Data-processing-with-Arduino-IDE-and-IOT/images/TableauWmessages.png" width=60% height=60%  title="Message Flow, Processing Slice and Number of Possible Iterations"/>

###### **Table  : Message Flow, Processing Slice and Number of Possible Iterations**

Looking at this table, we can say that the ESP32 microcontroller can easily receive a stream of 1024 messages with optimized batch processing of 128 messages



## KMeans Clustering for the CampusIOT's LoRaWAN Datasets
### ELSYS-ERS-CO2, ERS-CO2-48B30 W=512, Data = 5120, K=4


**About the dataset**

This input file contains the basic information (ID, CO2, T(°C), Cluster) about the datas recorded by the LNS. Cluster is something assign to the dataset based on the defined parameters like the behavior of datas recorded.

**About K-Means Clustering**

Clustering involves dividing points in a data set into a number of groups such that data points in the same groups are more similar to other data points in the same group than to those in other groups. The goal in clustering is to separate groups with similar characteristics and distribute them into clusters.

K-means clustering is one of the unsupervised machine learning algorithms. We define a target number k, which refers to the number of centroids we need in the dataset. As for the centroid, it is the imaginary or real location representing the center of the cluster. Each point in the dataset is assigned to each of the clusters by reducing the sum of squares within the cluster. Alternatively, the K-means algorithm identifies k number of centroids and then allocates each data point to the closest cluster, while keeping the centroids as small as possible. The “means” in K-means refer to the average of the data; i.e. find the center of gravity.

**Data Exploration**

Using the Seaborn module as well as the Matplotlib module, we will represent the distribution diagram with different variations of the data such as [cluster1data5120w512.csv](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/Data-processing-with-Arduino-IDE-and-IOT/ONLINE-MACHINE-LEARNING-AND-ONLINE-TECHNIQUE-WITH-REAL-DATA/data/cluster1data5120w512.csv). The Distplot represents the data by a histogram and a line in combination with it.
A Distplot or distribution diagram represents the variation in the distribution of data. Seaborn Distplot represents the overall distribution of continuous data variables.

  <img alt="Distribution Diagram" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/Data-processing-with-Arduino-IDE-and-IOT/ONLINE-MACHINE-LEARNING-AND-ONLINE-TECHNIQUE-WITH-REAL-DATA/images/cluster1data5120wk4displot.png" width=70% height=70%  title="Distribution Diagram"/>


###### **Distribution Diagram ELSYS_ERS_CO2, ERS_CO2_48B30 W=512, Data= 5120, K=4** 

**Applying KMeans Clustering with scikit-learn and  K=5**

K is The number of clusters to form as well as the number of centroids to generate.
Initialization method with k-means++ allows to select the initial cluster centroids using sampling based on an empirical probability distribution of the points contribution to the overall inertia.

  <img alt="KMeans Clustering with scikit-learn and  K=5" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/Data-processing-with-Arduino-IDE-and-IOT/ONLINE-MACHINE-LEARNING-AND-ONLINE-TECHNIQUE-WITH-REAL-DATA/images/algokmeansk5data5120w512k4.png" width=70% height=70%  title="KMeans Clustering with scikit-learn and  K=5"/>

###### **Applying KMeans Clustering with scikit-learn and  K=5** 

the scatter plots in the form of a cloud of points thus show how the Temperature variable is affected by the CO2 concentration variable. We cannot deduce either a linear or non-linear correlation. While we observe a linear evolution of the 5 centroids
