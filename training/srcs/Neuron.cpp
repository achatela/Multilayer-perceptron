#include "../includes/Neuron.hpp"

Neuron::Neuron()
{
    // this->_bias = (float)rand() / (float)RAND_MAX;
    this->_bias = 0.01;
}

Neuron::Neuron(int sizePreviousLayer, int featureNumber, int neuronsNumber)
{
    for (int i = 0; i < neuronsNumber; ++i)
        _weights.push_back(std::vector<float>());
    heInitialization(sizePreviousLayer, featureNumber);
    // this->_bias = (float)rand() / (float)RAND_MAX;
    this->_bias = 0.01;
}

Neuron::Neuron(std::vector<float> inputs, int featureNumber, int neuronsNumber)
{
    _inputs = inputs;
    for (int i = 0; i < featureNumber; ++i)
        _weights.push_back(std::vector<float>());
    heInitialization(neuronsNumber, featureNumber);
    // this->_bias = (float)rand() / (float)RAND_MAX;
    this->_bias = 0.01;
}

Neuron::~Neuron()
{
}

void Neuron::heInitialization(int sizePreviousLayer, int featureNumber)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, std::sqrt(2.0 / sizePreviousLayer));

    for (int j = 0; j < _weights.size(); ++j)
    {
        for (int i = 0; i < featureNumber; ++i)
        {
            _weights[j].push_back(dis(gen));
        }
    }
}