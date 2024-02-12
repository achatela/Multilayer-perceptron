#include "../includes/Layer.hpp"
#include <numeric>

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

// for the output layer
Layer::Layer(int neuronsNumber, int sizePreviousLayer, int featureNumber, bool isOutputLayer)
{
    for (int i = 0; i < neuronsNumber; i++)
    {
        Neuron neuron(sizePreviousLayer, featureNumber);
        neuron.setClassPredicted(i);
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

std::vector<float> Layer::softmaxFunction(std::vector<std::vector<float>> inputs)
{
    std::vector<float> outputs;
    std::vector<float> inputsSum;
    float sum = 0;
    float sums = 0;
    std::cout << "Size: " << inputs.size() << std::endl;
    for (int i = 0; i < inputs.size(); i++)
    {
        for (int j = 0; j < inputs[i].size(); j++)
            sum += inputs[i][j];
        inputsSum.push_back(sum);
        sums += exp(sum);
        sum = 0;
    }

    for (int i = 0; i < inputs.size(); i++)
    {
        outputs.push_back(exp(inputsSum[i]) / sums);
        std::cout << "Output: " << outputs[i] << std::endl;
    }

    return outputs;
}

void Layer::feedForward(Layer &previousLayer, int mode)
{
    std::vector<float> outputs(32, 0); // TODO: calculate the size of the outputs

    for (int i = 0; i < this->_neurons.size(); i++) // in every neuron of the actual layer
    {
        float sum = 0;
        for (int j = 0; j < previousLayer.getNeurons().size(); j++) // for every neuron of the previous layer
        {
            if (!previousLayer.getNeurons()[j].getActivated()) // if the previous neuron is not activated, we skip it
                continue;

            std::vector<float> previousNeuronsInputs = previousLayer.getNeurons()[j].getInputs();
            outputs[0] = previousNeuronsInputs[0];                 // the first element is the answer
            for (int k = 1; k < previousNeuronsInputs.size(); k++) // start from 1 because the first element is the answer
            {
                float computed = previousNeuronsInputs[k] * _neurons[i].getWeights()[k];
                outputs[k] += computed;
                sum += computed;
            }
        }
        sum += previousLayer.getBiasNeuron();

        if (mode == 1)
        {
            this->_neurons[i].setInputs(outputs);
            if (reluFunction(sum) > 0)
                this->_neurons[i].setActivated(true);
        }
        else if (mode == 2 && i == this->_neurons.size() - 1)
        {
            std::vector<std::vector<float>> inputs;
            for (int i = 0; i < this->_neurons.size(); i++)
            {
                inputs.push_back(this->_neurons[i].getInputs());
            }

            std::vector<float> result = softmaxFunction(inputs);
        }
    }
    debugNeuronsActivated();
}

void Layer::debugNeuronsActivated()
{
    int number = 0;
    for (int i = 0; i < this->_neurons.size(); i++)
    {
        if (this->_neurons[i].getActivated())
            number++;
    }
    std::cout << "Number of activated neurons: " << number << std::endl;
}

void Layer::backPropagation()
{
}