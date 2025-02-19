## Data mining from the Perret Tower with the GHA dimension reduction algorithm

The program will process the data in blocks of 1024 lines of the Perret Tower data, display the iteration number at each step, and update the eigenvalues ​​and eigenvectors accordingly. At the end, the results will be saved in a CSV file and displayed graphically.

Explanation:

- 1024-line buffer: We have introduced a constant W that defines the size of the buffer (1024 lines). The main loop now runs through the data in blocks of 1024 lines.

- Iteration number display: At each iteration, we display the current iteration number and the lines currently being processed.

- Block processing: For each block of 1024 lines, we apply the GHA algorithm on each line of the block.

- Last block management: If the total number of lines is not a multiple of 1024, the last block will be smaller. We use min(W, n - i) to handle this.

- Memory management: Eigen already manages memory dynamically

##### [Code Source](https://github.com/madou-sow/OnlineML_ESP32/blob/main/ARDUINO/GHA-PCA/src/)
