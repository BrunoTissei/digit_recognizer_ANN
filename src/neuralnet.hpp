#ifndef _NEURALNET_HPP
#define _NEURALNET_HPP

#include <vector>
#include <cstdio>
#include "layer.hpp"

class NeuralNetwork {

  public:
    NeuralNetwork(const std::vector<int> nn_layout, bool weight_set);
    ~NeuralNetwork();

    void set_input(const std::vector<double> input);
    void set_label(const std::vector<double> label);

    void train(int max_iter, double tol);
    std::vector<double> predict(const std::vector<double> input);

    double get_error();
    std::vector<double> get_output();
    std::vector<double> get_error_vector();

    void print_weights(FILE *file);
    void read_weights(FILE *file);

  private:
    bool weight_set;

    std::vector<double> input, output;
    std::vector<double> desired_output;
    std::vector<Layer> layers;
    std::vector<int> layout;
    
    void propagate();
    void backpropagate();
    void create_layer(int s_input, int s_output);

};

#endif
