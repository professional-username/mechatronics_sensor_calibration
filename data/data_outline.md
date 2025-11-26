# Description of Data files
This document outlines the structure of each document in data/, and gives sample data rows. Also listed are the number of rows, and the range of values in the first column.

data/clean_acceletomerer.csv:
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