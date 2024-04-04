#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

class Neuron
{

public:
    // for the input layer
    Neuron(std::vector<double> inputs);
    // for the hidden layers
    Neuron(int weightsNumber);
    // to load the model
    Neuron(std::vector<double> weights, bool biasNeuron);
    Neuron();
    ~Neuron();

    void xavierInitialization(int weightsNumber);

    std::vector<double> &getWeights() { return this->_weights; };

    void setOutput(double output) { this->_output = output; };
    double getOutput() { return this->_output; };

    void setError(double error) { this->error = error; };
    double getError() { return this->error; };

private:
    double error;

    std::vector<double> _weights;

    std::vector<double> _inputs;
    double _output;
};