# Digit Recognition ANN
Artificial neural network for recognizing digits from MNIST in C++ for a Kaggle submission
The best result from Kaggle was 0.98129, this can be achieved (or maybe improved) by tweaking the amount of epochs, the learning rate and the NN layout.

## Running

```
make
./neural --train data/train.csv --predict data/test.csv out.csv
```

## Configuring
The current number of epochs is set to 1, it can (should) be increased in src/main.cpp by changing NUM_EPOCH macro.
The ANN layout is set in the data/train.csv, it indicates the amount of neurons in each layer. The default is 784 500 10.

## TODO
Implement cross-entropy cost funtion.
Make it a CNN.
