# Online ML for ESP32
Online Machine Learning Algorithms for Embedded Systems (ESP32...)

## Introduction

In the context of Smart Buildings, data arrive continually from sensors. Learning about those data is challenging if the option is to realize it at the edge. Under this hypothesis, learning may occur on resource-constrained embedded devices like ESP32 devices. This repository shares our experience developing an online machine learning (ML) library for such devices. Initially, we specifically covered the case of online clustering for the Arduino and Micropython ecosystems. Hopefully, the context of smart buildings lends itself well to this study due to the nature and temporality of the data, for a building considered. Low technology has a future ahead of it. Based on the use case we can explain all the challenging tasks of building an online machine-learning ecosystem for low-power IoT, from the algorithmic, benchmarking, and toolkit perspectives. 

## Description of directories content

This repository is mainly divided into 3 directories. The first is dedicated to Arduino (C/C++) developments, the second is devoted to Micropython-ulab developments (Python), and the third contains material related to IoT, dataset, courses...

## Friends projects

1. [River](https://github.com/online-ml/river)
2. [Clustering on Low-Cost Machines](https://github.com/christophe-cerin/mosquitto-clustering)
3. [Clustering and TinyGo](https://github.com/antaresatlantide/implementation-with-tinygo)
4. [Arduino IDE](https://www.arduino.cc/en/software)
5. [Micropython ulab](https://micropython-ulab.readthedocs.io/en/latest/)
6. [Expressif ESP32](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/)

## Bibliography

```
@article{DBLP:journals/jpdc/CerinKS22,
  author       = {Christophe C{\'{e}}rin and
                  Keiji Kimura and
                  Mamadou Sow},
  title        = {Data stream clustering for low-cost machines},
  journal      = {J. Parallel Distributed Comput.},
  volume       = {166},
  pages        = {57--70},
  year         = {2022},
  url          = {https://doi.org/10.1016/j.jpdc.2022.04.009},
  doi          = {10.1016/J.JPDC.2022.04.009},
  timestamp    = {Mon, 28 Aug 2023 21:32:03 +0200},
  biburl       = {https://dblp.org/rec/journals/jpdc/CerinKS22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@article{DBLP:journals/information/ClossonCDB24,
  author       = {Louis Closson and
                  Christophe C{\'{e}}rin and
                  Didier Donsez and
                  Jean{-}Luc Baudouin},
  title        = {Design of a Meaningful Framework for Time Series Forecasting in Smart
                  Buildings},
  journal      = {Inf.},
  volume       = {15},
  number       = {2},
  pages        = {94},
  year         = {2024},
  url          = {https://doi.org/10.3390/info15020094},
  doi          = {10.3390/INFO15020094},
  timestamp    = {Wed, 20 Mar 2024 21:21:43 +0100},
  biburl       = {https://dblp.org/rec/journals/information/ClossonCDB24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

