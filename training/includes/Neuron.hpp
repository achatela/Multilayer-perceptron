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

    void setOutput(float output) { this->_output = output; };
    

private:
    float _bias;
    std::vector<float> _weights;
    std::vector<float> _inputs;
    float _output;
};