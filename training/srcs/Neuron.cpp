#include "../includes/Neuron.hpp"

Neuron::Neuron() {}
Neuron::~Neuron() {}

Neuron::Neuron(int weightsNumber) { xavierInitialization(weightsNumber); } // hidden layers and output layer

Neuron::Neuron(std::vector<double> inputs) : _inputs(inputs) {} // input layer

Neuron::Neuron(std::vector<double> weights, bool biasNeuron) : _weights(weights) { (void)biasNeuron; } // load model

void Neuron::xavierInitialization(int weightsNumber)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, std::sqrt(6.0 / (weightsNumber)));

    for (int i = 0; i < weightsNumber; ++i)
        _weights.push_back(dis(gen));
}