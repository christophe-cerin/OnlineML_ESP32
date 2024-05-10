## Online machine learning and online learning technique with real data

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

### La Tour Perret (Grenobles) – Data acquisition and exploitation by LoRaWAN

LoRaWAN : radio communication protocol that defines how terminal equipment communicates wirelessly via gateways, thus forming a low-consumption extended network

Endpoints: Endpoints are installed on four sides (SWW, SSE, NEE, NNW) of the top of the tower (in IP55 enclosures for Elsys EMS).
Data with 15 main parameters: The tourperret.log.gz file contains a dataset of 421937 messages received between June 2021 and June 2023 (2 years).
https://github.com/CampusIoT/datasets
https://perscido.univ-grenoble-alpes.fr/datasets/DS395
The field or parameter: The object contains Humidity and Temperature, the measurement of weather conditions at the top of the tower.
Weather conditions can be correlated with weather data from services such as OpenWeatherMap. The field or parameter: date contains the date and time with this example format “2021-01-01T00:05:28.165Z” =⇒ “20210101010528”

The data from the ELSYS-ERS sensor from January 1 at 00:05 a.m. to January 31 at 9:48 p.m. 2021, the quantity of C02, the Temperature, the date and the chronograph recorded every 5 minutes will be necessary to carry out interesting scientific work.

### Data analysis: Using data from Tour Perret – Local MQTT server and Arduino IDE

The CO2 concentration and temperature will be published from our local MQTT server with the subjects "C02/celsius" and "final/final" the data is received via the ESP32 board and used by the Arduino IDE where a program has been implemented works to carry out clustering with the K-MEANS algorithm. We considered in our case, a structure as a means of defining this type of data although it is more than a simple container for primitive data types because we can also define other functionalities what it does with the elements or data points, it is a movement that meets the criteria at the end of the range. it then returns an iterator to the first element of the deleted elements (which are actually just moved).

The objective is to analyze the behavior of the ESP32 controller in the face of a stream of messages received continuously subject to an implemented algorithm containing both a clustering algorithm which makes it possible to identify a maximum of 4 clusters.

During the experiments, we managed to send several message streams (256, 512, 1024 and 5120) containing CO2 quantities and temperatures from the MQTT server to the ESP32 microcontroller.
Using our integrated Arduino development environment, we have carried out a distribution of the data to be processed by the microcontroller. This variable is called W. It varies from 8 to 512. 512 is the maximum value taking into account the saturation that can occur. 


