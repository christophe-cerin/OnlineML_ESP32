[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Data mining from the Perret Tower with the GHA dimension reduction algorithm

The program will process the data in blocks of 1024 lines of the Perret Tower data - [TourPerret10col.csv](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/TourPerret10col.csv), display the iteration number at each step, and update the eigenvalues ​​and eigenvectors accordingly. At the end, the results will be saved in a CSV file and displayed graphically.

Explanation:

- 1024-line buffer: We have introduced a constant W that defines the size of the buffer (1024 lines). The main loop now runs through the data in blocks of 1024 lines.

- Iteration number display: At each iteration, we display the current iteration number and the lines currently being processed.

- Block processing: For each block of 1024 lines, we apply the GHA algorithm on each line of the block.

- Last block management: If the total number of lines is not a multiple of 1024, the last block will be smaller. We use min(W, n - i) to handle this.

- Memory management: the template librairy Eigen already manages memory dynamically

#### [Code Source](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/)

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Compilation](#compilation)
  - [Results1](#results1)
  - [Graphic obteined with online_GhaPca_update_buffer.out](#Graphic1)
  - [Graphic obteined with python and the data scores (scores.csv)](#Graphic2)
  - [Results2](#results2)
  - [Graphic obteined with online_GhaPca_update_buffer_normalize.out](#Graphic3)
- Analysis(#analysis)
- Solutions(#solutions)
    
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

The header file [ghapca.h](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/ghapca.h) contains the declarations of variables, constants and shared functions needed for the [online_GhaPca_update_buffer.cpp](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/online_GhaPca_update_buffer.cpp) source code to be compiled into online_GhaPca_update_buffer.out.
In this second program [online_GhaPca_update_buffer_normalize.cpp](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/online_GhaPca_update_buffer_normalize.cpp), there is a special feature: a section where the data is normalized.
This practice allows transforming the data without distorting it. While normalization consists of harmonizing the data
so that all the entries of the different data sets that relate to the same terms follow a similar format.

## Usage
### Compilation

```
1- g++ online_GhaPca_update_buffer.cpp -I /home/mamadou/src -L /usr/include/python3.10 -lpython3.10 -o online_GhaPca_update_buffer.out

2- g++ online_GhaPca_update_buffer_normalize.cpp -I /home/mamadou/src -L /usr/include/python3.10 -lpython3.10 -o online_GhaPca_update_buffer_normalize.out
```
### Results1

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

### Graphic1

<figure>
  <img alt="GHA - First Two Principal Components" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/images/FigureBufer1024.png"  title="Dimensionality Reduction on the Tour Perret Dataset with batch ghapca"/>

  <figcaption><b>Figure : </b> Dimensionality Reduction on the Tour Perret Dataset with  [online_GhaPca_update_buffer.cpp](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/online_GhaPca_update_buffer.cpp).</figcaption>
</figure>

### Graphic2

```PYTHON
import matplotlib.pyplot as plt
import pandas as pd

# Lire les scores depuis le fichier CSV
scores = pd.read_csv("scoresbon.csv", header=None)
print(scores)
```

```
         0        1
0 -2206.21 -2206.21
1 -2209.68 -2209.68
2 -2209.93 -2209.93
3 -2213.74 -2213.74
4 -2235.08 -2235.08
...    ...      ...  
6630 -2358.19 -2358.19
6631 -2245.78 -2245.78
6632 -2247.35 -2247.35
6633 -2208.48 -2208.48
6634 -2201.68 -2201.69
```

```PYTHON
plt.scatter(scores[0], scores[1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("GHA - First Two Principal Components")
plt.show()
```

<figure>
  <img alt="GHA - First Two Principal Components" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/images/figure-buffer-python.png"  title="Dimensionality Reduction"/>

  <figcaption><b>Figure : </b> Plot of Dimensionality Reduction on the Tour Perret Dataset with python</figcaption>
</figure>

### Results2

Updated Eigenvalues :

``` 
0.100697
0.100696
``` 

Updated Eigenvectors :

``` 
   -0.354593     -0.35579
-0.000275093  0.000169886
     0.31469     0.314211
 0.000777325   1.7666e-05
    -0.34374    -0.344359
   -0.749517    -0.748832
-0.000429183  0.000178904
    0.225717     0.225591
   -0.169892    -0.170084
   -0.124456    -0.124628

``` 

### Graphic3


<figure>
  <img alt="GHA - First Two Principal Components" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/images/Figure_online_GhaPca_update_buffer_normalize.png"  title="Dimensionality Reduction"/>

  <figcaption><b>Figure : </b> Plot of Dimensionality Reduction on the Tour Perret Dataset with [online_GhaPca_update_buffer_normalize.cpp](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/online_GhaPca_update_buffer_normalize.cpp)</figcaption>
</figure>

### Analysis

The differences between the results of the online_GhaPca_update_buffer_normalize.cpp and testing_batch_ghapca_json.py programs can be attributed to several factors, including:

1- Data normalization: The C++ program normalizes the data by ignoring some columns (1, 3, and 6), while the Python program normalizes all columns. This can lead to differences in the values ​​of the processed data.

Possible causes:

1.1- Constant column: Columns 1, 3, and 6 may contain a constant value (e.g., all elements are 0, 1, or some other fixed value).

1.2- Missing or poorly formatted data: If column 1,3, and 6 contain missing or poorly formatted data, it could lead to a situation where all values ​​are the same after processing.

1.3- Data loading issue: The CSV file might not be loaded properly, resulting in incorrect values ​​in column 1.3 and 6.

2- Data block processing: The C++ program processes the data in blocks of 1024 rows, while the Python program processes the data sequentially. This may affect how the eigenvalues ​​and eigenvectors are updated.

3- Parameter initialization: The initial parameters, such as eigenvalues ​​and eigenvectors, may be initialized differently in the two programs, which may influence the final results.

4- Online vs. batch learning: The C++ program uses an online approach to update the eigenvalues ​​and eigenvectors, while the Python program uses a batch approach. This may lead to differences in the convergence of the results.

5- Handling missing data: The Python program ignores rows without payload, which can reduce the number of data rows processed, while the C++ program does not seem to have this logic.

### Solutions

1- Ignore the problematic column
If columns 1,3 and 6 are not important for your analysis, you can ignore it when normalizing.

2- Replace constant values
If columns 1,3 and 6 are important but contain constant values, you can replace these values ​​with a default value or a small variation to allow normalization

3- Standardization instead of normalization
If normalization is a problem, you can consider using standardization (subtract the mean and divide by the standard deviation) instead of normalization. This works even if the values ​​are constant (although the standard deviation is zero in this case, which would also require special handling).


