#include "../includes/Layer.hpp"

// for the hidden layers
Layer::Layer(int neuronsNumber, int sizePreviousLayer, int featureNumber)
{
    for (int i = 0; i < neuronsNumber; i++)
    {
        Neuron neuron(sizePreviousLayer, featureNumber);
        _neurons.push_back(neuron);
    }
    this->_biasNeuron = 1;
}

// for the input layer
Layer::Layer(std::vector<std::vector<float>> inputs)
{
    for (int i = 0; i < inputs.size(); i++)
    {
        Neuron neuron(inputs[i], inputs[i].size());
        _neurons.push_back(neuron);
    }
    this->_biasNeuron = 1;
}

Layer::~Layer()
{
}

float Layer::reluFunction(float x)
{
    if (x > 0)
        return x;
    else
        return 0;
}

void Layer::forwardPropagation(Layer &previousLayer)
{
    for (int i = 0; i < this->_neurons.size(); i++)
    {
        float sum = 0;
        std::cout << this->_neurons.size() << " ";
        std::cout << previousLayer.getNeurons().size() << std::endl;
        for (int j = 0; j < previousLayer.getNeurons().size(); j++)
        {
            for (int k = 0; k < previousLayer.getNeurons()[j].getInputs().size(); k++)
            {
                if (k == 0)
                    std::cout << previousLayer.getNeurons()[j].getInputs()[k] << " " << _neurons[i].getWeights()[k] << std::endl;
                sum += previousLayer.getNeurons()[j].getInputs()[k] * _neurons[i].getWeights()[k];
            }
        }
        sum += previousLayer.getBiasNeuron();
        float activated = reluFunction(sum);
        this->_neurons[i].setOutput(sum);
        std::cout << "sum: " << sum << std::endl;
    }
}