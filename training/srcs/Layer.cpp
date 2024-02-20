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
    std::vector<float> exponentials;
    float sum = 0.0;

    for (const auto &input : inputs)
    {
        exponentials.push_back(0);
        float max_input = 0;
        for (float val : input)
        {
            if (val > max_input)
                max_input = val;
        }

        // Calculate the sum of exponentials
        for (float val : input)
        {
            float exponential = exp(val - max_input);
            exponentials[exponentials.size() - 1] += exponential;
        }
        sum += exponentials[exponentials.size() - 1];
    }
    // Calculate softmax output
    for (float exp_val : exponentials)
    {
        outputs.push_back(exp_val / sum);
    }
    for (auto &output : outputs)
    {
        std::cout << output << " ";
    }
    std::cout << std::endl;

    return outputs;
}

int Layer::calculatePrediction(std::vector<float> weights)
{
    std::vector<int> predictions;
    std::vector<float> exponentials;
    float sum = 0.0;

    for (int i = 0; i < weights.size(); i++)
    {
        exponentials.push_back(exp(weights[i]));
        sum += exponentials[i];
    }
    for (int i = 0; i < exponentials.size(); i++)
    {
        predictions.push_back(exponentials[i] / sum);
    }
    return std::distance(predictions.begin(), std::max_element(predictions.begin(), predictions.end()));
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

        if (mode == 1) // hidden layers
        {
            this->_neurons[i].setInputs(outputs);
            if (reluFunction(sum) > 0)
                this->_neurons[i].setActivated(true);
        }
        else if (mode == 2 && i == this->_neurons.size() - 1) // output layer last neuron
        {
            std::vector<std::vector<float>> inputs;
            for (int i = 0; i < this->_neurons.size(); i++)
            {
                inputs.push_back(this->_neurons[i].getWeights());
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

void Layer::backPropagation(std::vector<Layer> layers, std::vector<std::vector<float>> inputs, float learningRate)
{
    for (int i = layers.size() - 1; i > -1; i--)
    {
        for (int j = 0; j < layers[i].getNeurons().size(); j++)
        {
            if (i == layers.size() - 1) // output layer
            {
                int prediction = calculatePrediction(layers[i].getNeurons()[j].getWeights());
                // calculate gradient of output layer
                float error = prediction - (j + 1);
                for (int l = 0; l < layers[i].getNeurons()[j].getWeights().size(); l++)
                {
                    float gradient = error * (j + 1);
                    float updatedWeight = layers[i].getNeurons()[j].getWeights()[l] - learningRate * gradient;
                    layers[i].getNeurons()[j].setOneWeight(updatedWeight, l);
                }
            }
            else
            {
                // calculate gradient of hidden layers
                for (int k = 0; k < layers[i].getNeurons()[j].getWeights().size(); k++)
                {
                    float error = 0;
                    for (int l = 0; l < layers[i + 1].getNeurons().size(); l++)
                    {
                        error += layers[i + 1].getNeurons()[l].getWeights()[k] * layers[i + 1].getNeurons()[l].getSlope();
                    }
                    layers[i].getNeurons()[j].setSlope(error);
                    float gradient = error * layers[i].getNeurons()[j].getInputs()[k];
                    float updatedWeight = layers[i].getNeurons()[j].getWeights()[k] - learningRate * gradient;
                    layers[i].getNeurons()[j].setOneWeight(updatedWeight, k);
                }
            }
        }
    }
}