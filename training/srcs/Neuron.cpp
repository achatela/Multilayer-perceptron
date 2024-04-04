#include "../includes/Neuron.hpp"

Neuron::Neuron() {}
Neuron::~Neuron() {}

Neuron::Neuron(int weightsNumber) // hidden layers and output layer
{
    xavierInitialization(weightsNumber);
}

Neuron::Neuron(std::vector<double> inputs) : _inputs(inputs) {} // input layer

void Neuron::xavierInitialization(int weightsNumber)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, std::sqrt(6.0 / (weightsNumber * 2))); // TODO change weightsNumber * 2 to the number of neurons in the previous layer

    for (int i = 0; i < weightsNumber; ++i)
        _weights.push_back(dis(gen));
}