#include "neuralnet.hpp"

using namespace std;

NeuralNetwork::NeuralNetwork(const vector<int> nn_layout, bool weight_set) {
  this->layout = nn_layout;
  this->input.resize(layout[0]);
  this->weight_set = weight_set;
  output.resize(*(layout.rbegin()));

  // The layout vector specifies the structure of the network, so
  // this creates the layers of the ANN with the correct input and
  // output sizes
  for (int i = 0; i < (int) this->layout.size() - 1; ++i) {
    create_layer(this->layout[i], this->layout[i + 1]);
  }
}

NeuralNetwork::~NeuralNetwork() {
  input.clear();
  output.clear();
  desired_output.clear();
  layers.clear();
  layout.clear();
}

// Set input values of the first layer to start learning/predicting process
void NeuralNetwork::set_input(const vector<double> input) {
  this->input = input;
  layers[0].set_values(input);
}

// Set the desired output for training
void NeuralNetwork::set_label(const vector<double> label) {
  this->desired_output = label;
  for (int i = 0; i < (int) output.size(); ++i) {
    output[i] = 0.0;
  }
}

// Propagate values from one layer to the next
void NeuralNetwork::propagate() {
  int i;
  vector<t_neuron> *wave;

  // The output of one layer is writen to the input
  // of the next input using a reference to the next layer
  for (i = 0; i < (int) layout.size() - 2; ++i) {
    wave = layers[i+1].get_input_ref();
    layers[i].propagate(*wave);
  }

  // Last propagation step just writes results in the
  // ANN output vector
  layers[i].propagate(this->output);
}

// Propagate values from one layer to the previous
void NeuralNetwork::backpropagate() {
  vector<double> aux = get_error_vector();

  // Calculate every delta value for every neuron
  for (int i = layers.size() - 1; i >= 0; --i) {
    layers[i].backpropagate(aux);
    aux = layers[i].get_next_error();
  }

  // Update weights (this uses the delta values computed in the previous loop)
  for (int i = 0; i < (int) layers.size(); ++i) {
    layers[i].update_weights();
  }
}

// The training process is just ONE propagation and ONE backpropagation
// the "max_iter" and "tol" parameters are only for testing
void NeuralNetwork::train(int max_iter, double tol) {
  int i = 0;
  double error = tol + 1;
  while (error > tol && i < max_iter) {
    propagate();
    backpropagate();
    error = get_error();
    i++;
  }
}

// Prediction consists of setting the input and propagating it,
// it should be executed after training, since the initial weights
// are random
vector<double> NeuralNetwork::predict(const vector<double> input) {
  set_input(input);
  propagate();
  return output;
}

// Insert new layer (with s_input neurons in the input and s_output
// neurons in the output) to the layers vector
void NeuralNetwork::create_layer(int s_input, int s_output) {
  layers.push_back(Layer(s_input, s_output, input.size(), this->weight_set));
}

vector<double> NeuralNetwork::get_output() {
  return output;
}

double NeuralNetwork::get_error() {
  double error = 0.0;
  for (int i = 0; i < (int) desired_output.size(); ++i) {
    error += (desired_output[i] - output[i]) * (desired_output[i] - output[i]);
  }

  return 0.5 * error;
}

vector<double> NeuralNetwork::get_error_vector() {
  vector<double> aux;
  for (int i = 0; i < (int) output.size(); ++i) {
    aux.push_back(desired_output[i] - output[i]);
  }

  return aux;
}

// Print every weight value in the ANN to a file to be used later
void NeuralNetwork::print_weights(FILE *file) {
  vector<double> aux;

  for (auto i : layers) {
    aux = i.get_weights();
    for (auto j : aux) {
      fprintf(file, "%lf ", j);
    }
    aux.clear();
    fprintf(file, "\n");
  }
}

// Load pre-calculated weight values to the layers
void NeuralNetwork::read_weights(FILE *file) {
  double value;
  vector<double> aux;

  for (int k = 0; k < (int) layers.size(); ++k) {
    for (int i = 0; i < layout[k + 1]; ++i) {
      for (int j = 0; j < layout[k] + 1; ++j) {
        fscanf(file, "%lf", &value);
        aux.push_back(value);
      }
    }

    layers[k].set_weights(aux);
    aux.clear();
  }
}
