[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Data mining from the Perret Tower with the GHA dimension reduction algorithm

The program will process the data in blocks of 1024 lines of the Perret Tower data, display the iteration number at each step, and update the eigenvalues ​​and eigenvectors accordingly. At the end, the results will be saved in a CSV file and displayed graphically.

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
  - [Graphic](#graphic)

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
### Graphic

<center>
<img alt="GHA - First Two Principal Components" align="center" src="https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/images/FigureBufer1024.png" width=55% height=55%  title="GHA - First Two Principal Components"/>
</center>
</picture>
