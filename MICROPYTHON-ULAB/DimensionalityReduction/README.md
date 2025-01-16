# Generalized Hebbian Algorithm (GHA) for Online Principal Component Analysis (PCA)

This repository implements the **Generalized Hebbian Algorithm (GHA)**, also known as **Sanger's Rule**, for performing online Principal Component Analysis (PCA). The GHA algorithm incrementally updates the principal components and eigenvalues from streaming data, allowing for an efficient, adaptive computation of PCA.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Example 1](#example1)
  - [Example 2](#example2)
- [License](#license)

## Overview

Principal Component Analysis (PCA) is a dimensionality reduction technique that finds the principal components of the data. In many scenarios where data arrives incrementally, such as in real-time systems, it is important to perform PCA online. This code implements an online PCA using the GHA algorithm, which updates the eigenvalues and eigenvectors incrementally as new data points are provided.

The two main functions provided in this code are:
1. `ghapca_C`: Updates the matrix `Q` (the eigenvectors) based on the current data point `x`, the projected data point `y`, and the learning rate `gamma`.
2. `ghapca`: Performs the GHA update, calculating and updating the eigenvalues and eigenvectors of the data in an online fashion.

## Dependencies

To run this code, you need to have the following Python packages installed:

- `numpy`: For efficient matrix and vector operations.
- `matplotlib`: This is used to plot the results.
- `pandas`: For data analysis and manipulation tool. (only necessary to run `testing_ghapca_json.py`)

You can install these dependencies via pip:

```bash
pip install numpy matplotlib pandas
```

## Usage

### Example 1

```bash
python3 testing_ghapca.py
```
A random dataset is generated, and then the `ghapca` algorithm is applied.

### Example 2

```bash
python3 testing_ghapca_json.py
```

First, a dataset [available online](https://github.com/CampusIoT/datasets/tree/main/TourPerret), namely the Tour Perret dataset (`tourperret.ndjson.gz`), is loaded, and then the `ghapca` algorithm is applied. We consider the following 10 attributes from Payload key:

1. accMotion: sensor vibrations. This measurement counts the number of movements of the box object containing the sensor detected by the accelerometer;
2. digital: There are no magnets on the Perret Tower sensors, so the measurement must be zero.
3. humidity: outside relative humidity;
4. pulseAbs: Relative pulse count;
5. temperature: outside temperature;
6. vdd: battery voltage in mV. It varies with ambient temperature. The battery is almost empty at around 2.9 - 2.8V;
7. waterleak: Water Leak strength or detection;
8. x: x-axis sensor orientation;
9. y: y-axis sensor orientation;
10 z: z-axis sensor orientation.

<figure>
    <img src="Figure_2.png"
         alt="Dimensionality reduction on JSON data">
    <figcaption>Dimensionality Reduction on the Tour Perret Dataset.</figcaption>
</figure>


