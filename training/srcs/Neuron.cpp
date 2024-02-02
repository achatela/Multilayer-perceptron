#include "../includes/Neuron.hpp"

Neuron::Neuron(int sizePreviousLayer, int featureNumber)
{
    heInitialization(sizePreviousLayer, featureNumber);
    this->_bias = (float)rand() / (float)RAND_MAX;
    std::cout << this->_bias << std::endl;
    // this->_bias = 0;
}

Neuron::~Neuron()
{
}

void Neuron::heInitialization(int sizePreviousLayer, int featureNumber)
{
    float square = sqrt(2.0 / sizePreviousLayer);

    for (int i = 0; i < featureNumber; i++)
    {
        _weights.push_back((float)rand() / (float)RAND_MAX * square);
    }
}