#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

class Neuron
{

public:
    // for the input layer
    Neuron(std::vector<double> inputs, int featureNumber);
    // for the hidden layers
    Neuron(int sizePreviousLayer, int featureNumber, int weightsNumber);
    // for the output layer
    Neuron();
    ~Neuron();

    void xavierInitialization(int sizePreviousLayer, int featureNumber);

    double &getBias() { return this->_bias; };
    void setBias(double bias) { this->_bias = bias; };

    void updateBias(double delta) { this->_bias -= delta; };

    std::vector<double> &getWeights() { return this->_weights; };

    void setOutput(double output) { this->_output = output; };
    double getOutput() { return this->_output; };

    void setError(double error) { this->error = error; };
    double getError() { return this->error; };

private:
    double error;
    double _bias;

    std::vector<double> _weights;

    std::vector<double> _inputs;
    double _output;
};