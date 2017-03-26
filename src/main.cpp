#include <cstdio>
#include <string>
#include <vector>
#include <iostream>

#include "helper.hpp"
#include "neuralnet.hpp"

#define NUM_EXAMPLES 42000
#define NUM_INPUT 784
#define NUM_OUTPUT 10
#define NUM_EPOCH 1
#define NUM_PREDICT 28000

using namespace std;

FILE *output_f;

void set_input(string param, vector<vector<double>> &example_input, 
    vector<vector<double>> &example_label, vector<int> &NNlayout);

void set_predict(string param1, string param2, vector<vector<double>> &predict_input);

int main(int argc, char **argv) {
  srand(time(NULL));

  FILE *weights_f;
  double error, start_time;
  bool predicting = false, set_weight = false;
  string parameter, weight_filename;

  vector<vector<double>> example_input(NUM_EXAMPLES, vector<double>(NUM_INPUT));
  vector<vector<double>> predict_input(NUM_PREDICT, vector<double>(NUM_INPUT));
  vector<vector<double>> example_label(NUM_EXAMPLES, vector<double>(NUM_OUTPUT));
  vector<int> NNlayout;

  // Parse parameters:
  //   train: file with data for training
  //   predict: file with data to predict
  //   weight: file with pre-calculated weights
  for (int i = 1; i < argc; ) {
    parameter = string(argv[i]);
    if (parameter == "--train") {
      set_input(string(argv[i+1]), example_input, example_label, NNlayout);
      i++;
    } else if (parameter == "--predict") {
      set_predict(string(argv[i+1]), string(argv[i+2]), predict_input);
      i += 2;  
      predicting = true;
    } else if (parameter == "--weight") {
      set_weight = true;
      weight_filename = string(argv[i+1]);
      i++;
    }
    i++;
  }

  // Create neural network and set weights if necessary
  NeuralNetwork NN(NNlayout, set_weight);
  if (set_weight) {
    NN.read_weights(fopen(weight_filename.c_str(), "r"));
  }

  // Run epochs
  for (int k = 0; k < NUM_EPOCH; ++k) {
    error = 0.0;
		start_time = timestamp();

    // Iterate training data, input values to neural network and train it
    for (int j = 0; j < NUM_EXAMPLES; ++j) {
      NN.set_input(example_input[j]);
      NN.set_label(example_label[j]);
      NN.train(1, 0.00001);
      cout << j << "\r";
      error += NN.get_error();
    }

    printf("ERROR = %lf (%d) | %lf ms\n", error, k + 1, (timestamp() - start_time));
  }

  // Test training data only
  if (!predicting) {
    int correct = 0;
    for (int j = 0; j < NUM_EXAMPLES; ++j) {
      vector<double> output = NN.predict(example_input[j]);

      int result = 0, desired = 0;
      double max1 = -1, max2 = -1;
      for (int jj = 0; jj < (int) output.size(); ++jj) {
        // The NN output is a vector with 10 values, the answer is the greatest value (0-9)
        if (output[jj] > max1) {
          max1 = output[jj];
          result = jj;
        }

        if (example_label[j][jj] > max2) {
          max2 = example_label[j][jj];
          desired = jj;
        }
      }

      printf("%d (%lf) - %d (%c)\n", result, max1, desired, (result == desired) ? ('R') : ('W'));
      correct += (result == desired);
    }

    printf("Accuracy = %lf\n", (double) correct / NUM_EXAMPLES);
  } else {
    // Predict results
    fprintf(output_f, "ImageId,Label\n");
    for (int j = 0; j < NUM_PREDICT; ++j) {
      vector<double> output = NN.predict(predict_input[j]);

      int ans = 0;
      double max = -1;

      // The NN output is a vector with 10 values, the answer is the greatest value (0-9)
      for (int jj = 0; jj < (int) output.size(); ++jj) {
        if (output[jj] > max) {
          max = output[jj];
          ans = jj;
        }
      }
        
      cout << j << "\r";
      fprintf(output_f, "%d,%d\n", j + 1, ans);
    }

    fclose(output_f);
  }

  weights_f = fopen("weight.out", "w+");
  NN.print_weights(weights_f);
  fclose(weights_f);
	
  return 0;
}

// Read training data
void set_input(string param, 
    vector<vector<double>> &example_input, 
    vector<vector<double>> &example_label,
    vector<int> &NNlayout) {
  int holder, num_layers;
  double greater = -1;
  FILE *input_f = fopen(param.c_str(), "r");

  // Number of layers
  fscanf(input_f, "%d", &num_layers);
  NNlayout.resize(num_layers);

  // Neural network layout
  for (int i = 0; i < num_layers; ++i) {
    fscanf(input_f, "%d", &NNlayout[i]);
  }

  for (int i = 0; i < NUM_EXAMPLES; ++i) {
    // Desired value
    fscanf(input_f, "%d", &holder); 
    for (int j = 0; j < NUM_OUTPUT; ++j) {
      example_label[i][j] = 0.0;
    }

    example_label[i][holder] = 1.0;

    // Input
    for (int j = 0; j < NNlayout[0]; ++j) {
      fscanf(input_f, "%lf", &example_input[i][j]);
      greater = max(greater, example_input[i][j]);
    }

    for (int j = 0; j < NNlayout[0]; ++j) {
      example_input[i][j] /= greater;
    }
  }

  fclose(input_f);
}

// Read predicting data
void set_predict(string param1, string param2, 
    vector<vector<double>> &predict_input) {
  double greater = -1;
  FILE *predict_f = fopen(param1.c_str(), "r");

  for (int i = 0; i < NUM_PREDICT; ++i) {
    for (int j = 0; j < NUM_INPUT; ++j) {
      fscanf(predict_f, "%lf", &predict_input[i][j]);
      greater = max(greater, predict_input[i][j]);
    }

    for (int j = 0; j < NUM_INPUT; ++j) {
      predict_input[i][j] /= greater;
    }
  }

  output_f = fopen(param2.c_str(), "w+");

  fclose(predict_f);
}
