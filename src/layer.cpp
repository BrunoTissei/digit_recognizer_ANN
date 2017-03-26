#include "layer.hpp"
#include <omp.h>

using namespace std;

inline double Layer::sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

inline double Layer::sigmoid_deriv(double x) {
  double aux = exp(-x);
  return aux / ((1 + aux) * (1 + aux));
}

// Create single layer and resize vectors with correct sizes
Layer::Layer(int s_input, int s_output, double nn_input, bool weight_set) {
  this->s_input = s_input;
  this->s_output = s_output;

  weights.resize(s_output);
  double val = 1 / sqrt(nn_input);

  // Set weights with random values
  for (int i = 0; i < s_output; ++i) {
    // Reserve space for bias
    weights[i].resize(s_input + 1);

    // weight_set indicates when there is pre-calculated weights as input
    if (!weight_set) {
      for (int j = 0; j < s_input; ++j) {
         weights[i][j] = ((((double) rand()) / RAND_MAX) * 2 * val) - val;
      }
    }
  }

  // Set bias
  input_neurons.resize(s_input + 1);
  input_neurons[s_input] = 1.0;

  v.resize(s_output);
  delta.resize(s_output);
}

Layer::~Layer() {
  for (int i = 0; i < s_output; ++i) {
    weights[i].clear();
  }

  weights.clear();
  input_neurons.clear();
}

// Set input values
void Layer::set_values(const vector<double> input) {
  for (int i = 0; i < s_input; ++i) {
    this->input_neurons[i] = input[i];
  }
}
 
// Propagate values from input to &output
void Layer::propagate(vector<t_neuron> &output) {
  for (int i = 0; i < s_output; ++i) {
    v[i] = 0.0;
    for (int j = 0; j < s_input + 1; ++j) {
      v[i] += input_neurons[j] * weights[i][j];
    }
    
    output[i] = sigmoid(v[i]);
  }
}

// Backpropagate values using error vector obtained from
// get_next_error method from past layer to set delta values
// from current layer
void Layer::backpropagate(vector<double> error) {
  for (int i = 0; i < s_output; ++i) {
    delta[i] = sigmoid_deriv(v[i]) * error[i];
  }
}

// Update weights from current layer using delta and input values
void Layer::update_weights() {
  for (int i = 0; i < s_output; ++i) {
    for (int j = 0; j < s_input + 1; ++j) {
      // The learning rate is fixed at 0.45 (to be improved)
      weights[i][j] += 0.45 * (input_neurons[j]) * delta[i];
    }
  }
}

vector<t_neuron> *Layer::get_input_ref() {
  return &(this->input_neurons);
}

// Calculate "next error" to be used on backpropagation
vector<double> Layer::get_next_error() {
  vector<double> sum(s_input+1);

  for (int j = 0; j < s_output; ++j) {
    for (int i = 0; i < s_input + 1; ++i) {
      sum[i] += delta[j] * weights[j][i];
    }
  }

  return sum;
}

// Save weights to be used in the future
vector<double> Layer::get_weights() {
  vector<double> aux;

  for (int i = 0; i < s_output; ++i) {
    for (int j = 0; j < s_input + 1; ++j) {
      aux.push_back(weights[i][j]);
    }
  }

  return aux;
}

// Set weights (instead of random values)
void Layer::set_weights(vector<double> w) {
  int k = 0;
  for (int i = 0; i < s_output; ++i) {
    for (int j = 0; j < s_input + 1; ++j) {
      weights[i][j] = w[k++];
    }
  }
}
