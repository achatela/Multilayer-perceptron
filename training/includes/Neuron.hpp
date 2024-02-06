#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

class Neuron
{

public:
    // for the input layer
    Neuron(std::vector<float> inputs, int featureNumber);
    // for the hidden layers
    Neuron(int sizePreviousLayer, int featureNumber);
    ~Neuron();

    void heInitialization(int sizePreviousLayer, int featureNumber);

    float &getBias() { return this->_bias; };
    std::vector<float> &getWeights() { return this->_weights; };

    std::vector<float> &getInputs() { return this->_inputs; };
    void setInputs(std::vector<float> inputs) { this->_inputs = inputs; };

    void setOutput(float output) { this->_output = output; };
    float getOutput() { return this->_output; };

    void setActivated(bool activated) { this->_activated = activated; };
    bool getActivated() { return this->_activated; };

private:
    float _bias;
    std::vector<float> _weights;
    std::vector<float> _inputs;
    float _output;
    bool _activated = false;
};