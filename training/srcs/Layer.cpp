#include "../includes/Layer.hpp"

Layer::Layer(int neuronsNumber, int sizePreviousLayer, int featureNumber)
{
    for (int i = 0; i < neuronsNumber; i++)
    {
        Neuron neuron(sizePreviousLayer, featureNumber);
        _neurons.push_back(neuron);
    }
    this->_biasNeuron = 1;
}

Layer::Layer(std::vector<std::vector<float>> inputs)
{
}

Layer::~Layer()
{
}

void Layer::forwardPropagation(Layer previousLayer)
{
}