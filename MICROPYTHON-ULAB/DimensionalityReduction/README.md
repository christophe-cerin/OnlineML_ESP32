# Generalized Hebbian Algorithm (GHA) for Online Principal Component Analysis (PCA)

This repository provides an implementation of the **Generalized Hebbian Algorithm (GHA)**, also known as **Sanger's Rule**, for performing online Principal Component Analysis (PCA). The GHA algorithm incrementally updates the principal components and eigenvalues from streaming data, allowing for an efficient, adaptive computation of PCA.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Function Descriptions](#function-descriptions)
  - [Example](#example)
- [License](#license)

## Overview

Principal Component Analysis (PCA) is a dimensionality reduction technique that finds the principal components of the data. In many scenarios where data arrives incrementally, such as in real-time systems, it is important to perform PCA in an online manner. This code implements an online PCA using the GHA algorithm, which updates the eigenvalues and eigenvectors incrementally as new data points are provided.

The two main functions provided in this code are:
1. `ghapca_C`: Updates the matrix `Q` (the eigenvectors) based on the current data point `x`, the projected data point `y`, and the learning rate `gamma`.
2. `ghapca`: Performs the GHA update, calculating and updating the eigenvalues and eigenvectors of the data in an online fashion.

## Dependencies

To run this code, you need to have the following Python packages installed:

- `numpy`: For efficient matrix and vector operations.
- `matplotlib`: For plotting the results.

You can install these dependencies via pip:

```bash
pip install numpy matplotlib

