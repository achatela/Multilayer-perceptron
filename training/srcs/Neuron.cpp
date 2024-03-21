#include "../includes/Neuron.hpp"

// for the output layer
Neuron::Neuron()
{
    // this->_bias = (double)rand() / (double)RAND_MAX;
    this->_bias = 0.01;
}

// for the hidden layers and output layer
Neuron::Neuron(int sizePreviousLayer, int featureNumber, int weightsNumber)
{
    (void)featureNumber;
    xavierInitialization(sizePreviousLayer, weightsNumber);
    // this->_bias = (double)rand() / (double)RAND_MAX;
    this->_bias = 0.01;
}

// for the input layer
Neuron::Neuron(std::vector<double> inputs, int featureNumber)
{
    (void)featureNumber;
    _inputs = inputs;
    // this->_bias = (double)rand() / (double)RAND_MAX;
    this->_bias = 0.01;
}

Neuron::~Neuron()
{
}

void Neuron::xavierInitialization(int sizePreviousLayer, int featureNumber)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, std::sqrt(6.0 / (sizePreviousLayer + featureNumber)));

    for (int i = 0; i < featureNumber; ++i)
    {
        _weights.push_back(dis(gen));
    }
}