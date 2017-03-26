#ifndef _LAYER_HPP
#define _LAYER_HPP

#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstdio>

#include "helper.hpp"

typedef double t_neuron;

class Layer {

  public:
    Layer(int s_input, int s_output, double nn_input, bool weight_set);
    ~Layer();

    void set_values(const std::vector<double> input);
    void propagate(std::vector<t_neuron> &output);
    void backpropagate(std::vector<double> error);
    void update_weights();

    std::vector<t_neuron> *get_input_ref();
    std::vector<double> get_next_error();
    std::vector<double> get_weights();
    void set_weights(std::vector<double> w);

  private:
    int s_input, s_output;

    inline double sigmoid(double x);
    inline double sigmoid_deriv(double x);

    // weigth on j -> i
    std::vector<std::vector<double>> weights;
    std::vector<t_neuron> input_neurons;
    std::vector<double> delta;
    std::vector<double> v;

};

#endif
