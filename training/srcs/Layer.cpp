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

void Layer::feedForward(Layer &previousLayer)
{
    int number = 0;
    std::vector<float> outputs;
    for (int i = 0; i < this->_neurons.size(); i++)
    {
        float sum = 0;
        for (int j = 0; j < previousLayer.getNeurons().size(); j++)
        {
            std::vector<float> previousNeuronsInputs = previousLayer.getNeurons()[j].getInputs();

            for (int k = 0; k < previousNeuronsInputs.size(); k++)
            {
                float computed = previousNeuronsInputs[k] * _neurons[i].getWeights()[k];
                outputs.push_back(computed);
                sum += computed;
            }
        }
        sum += previousLayer.getBiasNeuron();
        float activated = reluFunction(sum);
        this->_neurons[i].setInputs(outputs);
        outputs.clear();
        if (activated > 0)
        {
            number++;
            this->_neurons[i].setActivated(true);
        }
    }
    std::cout << "number of neurons actived " << number << std::endl;
}