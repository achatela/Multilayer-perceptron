#include "../includes/Neuron.hpp"
Neuron::Neuron() {}
Neuron::~Neuron() {}

Neuron::Neuron(int weightsNumber) : _bias((double)rand() / (double)RAND_MAX) // hidden layers and output layer
{
    xavierInitialization(weightsNumber);
}

Neuron::Neuron(std::vector<double> inputs) : _bias((double)rand() / (double)RAND_MAX), _inputs(inputs) {} // input layer

void Neuron::xavierInitialization(int weightsNumber)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, std::sqrt(6.0 / (weightsNumber * 2)));

    for (int i = 0; i < weightsNumber; ++i)
        _weights.push_back(dis(gen));
}