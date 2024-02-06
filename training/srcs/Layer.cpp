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
        _neurons[i].setActivated(true);
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

float Layer::softMaxFunction(float x, std::vector<float> outputs)
{
    float sum = 0;
    for (int i = 1; i < outputs.size(); i++)
    {
        float compute = std::exp(outputs[i]);
        sum += compute;
    }
    return std::exp(x) / sum;
}

void Layer::feedForward(Layer &previousLayer, int mode)
{
    int number = 0;
    std::vector<float> outputs(32, 0);
    for (int i = 0; i < this->_neurons.size(); i++)
    {
        float sum = 0;
        for (int j = 0; j < previousLayer.getNeurons().size(); j++)
        {
            if (!previousLayer.getNeurons()[j].getActivated())
                continue;
            std::vector<float> previousNeuronsInputs = previousLayer.getNeurons()[j].getInputs();

            outputs[0] = previousNeuronsInputs[0];
            for (int k = 1; k < previousNeuronsInputs.size(); k++) // start from 1 because the first element is the answer
            {
                float computed = previousNeuronsInputs[k] * _neurons[i].getWeights()[k];
                outputs[k] += computed;
                sum += computed;
            }
        }
        sum += previousLayer.getBiasNeuron();
        float activated;
        if (mode == 1)
            activated = reluFunction(sum);
        else
            activated = softMaxFunction(sum, outputs);
        this->_neurons[i].setInputs(outputs);
        if (activated > 0)
        {
            number++;
            this->_neurons[i].setActivated(true);
        }
    }
    std::cout << "number of neurons actived " << number << std::endl;
}

void Layer::backPropagation()
{
}