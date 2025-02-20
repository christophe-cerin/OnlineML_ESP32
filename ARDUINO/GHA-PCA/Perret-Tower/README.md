[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Data mining from the Perret Tower with the GHA dimension reduction algorithm

The program will process the data in blocks of 1024 lines of the Perret Tower data - [TourPerret10col.csv](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/TourPerret10col.csv), display the iteration number at each step, and update the eigenvalues ​​and eigenvectors accordingly. At the end, the results will be saved in a CSV file and displayed graphically.

Explanation:

- 1024-line buffer: We have introduced a constant W that defines the size of the buffer (1024 lines). The main loop now runs through the data in blocks of 1024 lines.

- Iteration number display: At each iteration, we display the current iteration number and the lines currently being processed.

- Block processing: For each block of 1024 lines, we apply the GHA algorithm on each line of the block.

- Last block management: If the total number of lines is not a multiple of 1024, the last block will be smaller. We use min(W, n - i) to handle this.

- Memory management: Eigen already manages memory dynamically

#### [Code Source](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/)

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Compilation](#compilation)
  - [Results](#results)
  - [Graphic obteined with online_GhaPca_update_buffer.out](#graphic obteined with online_GhaPca_update_buffer.out)

## Overview

The details of the 10 attributes from [TourPerret10colHead.csv](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/TourPerret10colHead.csv])
1. accMotion: sensor vibrations. This measurement counts the number of movements of the box object containing the sensor detected by the accelerometer;
2. digital: There are no magnets on the Perret Tower sensors, so the measurement must be zero.
3. humidity: outside relative humidity;
4. pulseAbs: Relative pulse count;
5. temperature: outside temperature;
6. vdd: battery voltage in mV. It varies with ambient temperature. The battery is almost empty at around 2.9 - 2.8V;
7. waterleak: Water Leak strength or detection;
8. x: x-axis sensor orientation;
9. y: y-axis sensor orientation;
10. z: z-axis sensor orientation.


## Dependencies
The header file [ghapca.h](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/ghapca.h) of the module contains everything that is necessary to know
to have the right to use, in [online_GhaPca_update_buffer.cpp](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/online_GhaPca_update_buffer.cpp), the functionalities of the module online_GhaPca_update_buffer.out

## Usage
### Compilation

```
g++ online_GhaPca_update_buffer.cpp -I /home/mamadou/src -L /usr/include/python3.10 -lpython3.10 -o online_GhaPca_update_buffer.out
```
### Results

Iteration over the data in blocks of W rows 1024 :

```
Iteration 1 processing rows 0 to 1023
Iteration 2 processing rows 1024 to 2047
Iteration 3 processing rows 2048 to 3071
Iteration 4 processing rows 3072 to 4095
Iteration 5 processing rows 4096 to 5119
Iteration 6 processing rows 5120 to 6143
Iteration 7 processing rows 6144 to 6634
```

Updated Eigenvalues :

``` 
1527.38
1457.97
``` 

Updated Eigenvectors :

``` 
   -0.764374    -0.764374
 -1.4892e-07  1.18823e-07
    0.150348     0.150348
   4.208e-07  1.23561e-08
  -0.0978109   -0.0978117
   -0.612758    -0.612759
-2.32336e-07  1.25131e-07
   0.0386407    0.0386405
  -0.0748256   -0.0748259
  -0.0315502   -0.0315501
``` 

### Graphic obteined with online_GhaPca_update_buffer.out

<figure>
  <img alt="GHA - First Two Principal Components" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/images/FigureBufer1024.png"  title="Dimensionality Reduction on the Tour Perret Dataset with batch ghapca"/>

  <figcaption><b>Figure : </b> Dimensionality Reduction on the Tour Perret Dataset with  ghapca and buffer=1024.</figcaption>
</figure>
