# Description of Data files
This document outlines the structure of each document in data/, and gives sample data rows. Also listed are the number of rows, and the range of values in the first column.

data/clean_accelerometer.csv:
```
angle,value,dimension,data type
0,804,x,recorded
0,716,x,recorded
...
162,13456,z,recorded
162,13352,z,recorded
...
288,-3.031456715,z,calculated
306,-5.766173325,z,calculated
```
N: 6365
X_range: 0-360

data/clean_infrared.csv
```
distance,value,dataset
0.5,181,A
0.5,181,A
...
120,107,A
120,106,A
...
150,125,B
150,123,B
```
N: 3900
X_range: 0-500

data/clean_ultrasonic.csv
```
distance,value,surface
1,148.2352941,phone
1,148.2352941,phone
...
4,231.7647059,phone
5,264.1176471,phone
...
100,5765.882353,hand
100,5622.941176,hand
```
N: 307
X_range: 0-100

data/calibrated_accelerometer.csv
```
angle,reading,dimension,data type
0,-1.96,x,recorded
0,-1.86,x,recorded
...
360,-1.2,x,recorded
0,-10.65,z,recorded
...
342,-9.329864425,z,calculated
360,-9.81,z,calculated
```
N: 4242
X_range: 0-360

data/calibrated_infrared.csv
```
distance, measured distance
20.00,20.32
20.00,20.32
...
40.00,41.83
45.00,46.93
...
100.00,109.52
100.00,109.52
```
N: 1700
X_range: 20-100

data/calibrated_ultrasonic.csv
```
distance, measured distance
5.00, 5.61
5.00, 5.61
...
25.00, 26.27
30.00, 31.03
...
100.00, 103.76
100.00, 103.76
```
N: 2000
X_range: 5-100