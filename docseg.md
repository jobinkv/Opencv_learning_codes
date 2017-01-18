# The main functions of docseg


## Table of Contents
* **[How to run](#how-to-run)** 



### How to run
The SVM based training is shown below

``` bash
#!/bin/bash
make
./docSeg gmmGbrTr "/users/jobinkv/threeClassTrainData/newTrain" "/users/jobinkv/threeClassTrainData/newtrGt" tdeepfet.xml
mkdir -p output
./docSeg gmmGrTestF "/users/jobinkv/threeClassTrainData/testImg" "./output" tdeepfet.xml
```
