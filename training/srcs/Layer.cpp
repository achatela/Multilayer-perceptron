#include "../includes/Layer.hpp"
#include <numeric>
#include <strings.h>
// for the input layer
Layer::Layer(std::vector<std::vector<double>> inputs)
{
    for (int i = 0; i < inputs.size(); i++)
    {
        Neuron neuron(inputs[i], inputs[i].size());
        neuron.setActivated(true);
        _neurons.push_back(neuron);
    }
    this->_biasNeuron = 0.01;
}

// for the hidden layers
Layer::Layer(int neuronsNumber, int sizePreviousLayer, int featureNumber, int weightsNumber)
{
    for (int i = 0; i < neuronsNumber; i++)
    {
        Neuron neuron(sizePreviousLayer, featureNumber, weightsNumber);
        _neurons.push_back(neuron);
    }
    this->_biasNeuron = 0.01;
}

// for the output layer
Layer::Layer(int neuronsNumber, int sizePreviousLayer, int featureNumber, int weightsNumber, bool isOutputLayer)
{
    for (int i = 0; i < neuronsNumber; i++)
    {
        Neuron neuron(sizePreviousLayer, featureNumber, weightsNumber);
        neuron.setClassPredicted(i);
        neuron.setActivated(true);
        _neurons.push_back(neuron);
    }
    this->_biasNeuron = 0.01; // useless
}

Layer::~Layer()
{
}

void Layer::reluActivation(double sum, int i)
{
    if (sum > 0) // ReLU
    {
        this->_neurons[i].setActivated(true);
        this->_neurons[i].setOutput(sum);
        return;
    }
    this->_neurons[i].setActivated(false);
    this->_neurons[i].setOutput(0);
}

std::vector<double> Layer::softmaxFunction(std::vector<double> inputs)
{
    std::vector<double> outputs;
    double maxInput = *std::max_element(inputs.begin(), inputs.end());

    std::vector<double> exponentials;
    double sum = 0;
    for (int i = 0; i < inputs.size(); i++)
    {
        // Subtract the max value to prevent overflow
        double exponential = exp(inputs[i] - maxInput);
        exponentials.push_back(exponential);
        sum += exponential;
    }

    for (int i = 0; i < exponentials.size(); i++)
    {
        outputs.push_back(exponentials[i] / sum);
        std::cout << outputs.back() << " ";
    }
    std::cout << std::endl;

    return outputs;
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

void Layer::inputLayerFeedForward(Layer &previousLayer)
{
    for (int i = 0; i < this->_neurons.size(); i++)
    {
        double sum = 0;
        std::vector<Neuron> &neurons = previousLayer.getNeurons();

        for (int j = 0; j < neurons.size(); j++)
            for (int k = 1; k < neurons[j].getInputs().size(); k++)
                sum += this->_neurons[i].getWeights()[k] * neurons[j].getInputs()[k];

        sum += previousLayer.getBiasNeuron();
        reluActivation(sum, i);
    }
    debugNeuronsActivated();
}

void Layer::hiddenLayerFeedForward(Layer &previousLayer)
{
    for (int i = 0; i < this->_neurons.size(); i++)
    {
        double sum = 0;
        std::vector<Neuron> &neurons = previousLayer.getNeurons();

        for (int j = 0; j < neurons.size(); j++)
            sum += this->_neurons[i].getWeights()[j] * neurons[j].getOutput();

        sum += previousLayer.getBiasNeuron();
        reluActivation(sum, i);
    }
    debugNeuronsActivated();
}

std::vector<double> Layer::outputLayerFeedForward(Layer &previousLayer)
{
    std::vector<double> logits;
    for (int i = 0; i < this->_neurons.size(); i++)
    {
        double sum = 0;
        std::vector<Neuron> &neurons = previousLayer.getNeurons();

        for (int j = 0; j < neurons.size(); j++)
            sum += this->_neurons[i].getWeights()[j] * neurons[j].getOutput();

        sum += previousLayer.getBiasNeuron();
        logits.push_back(sum);
    }
    std::cout << std::endl;
    return logits;
}

void Layer::feedForward(Layer &previousLayer, int mode, std::vector<std::vector<double>> inputs)
{
    if (mode == 0)
        this->inputLayerFeedForward(previousLayer);
    else if (mode == 1)
        this->hiddenLayerFeedForward(previousLayer);
    else if (mode == 2)
    {
        std::vector<double> probabilities = softmaxFunction(this->outputLayerFeedForward(previousLayer));
        for (int i = 0; i < probabilities.size(); i++)
            this->_neurons[i].setOutput(probabilities[i]);

        double error = crossEntropyLoss(probabilities, inputs);
        this->setLoss(error);
    }
}

double Layer::crossEntropyLoss(std::vector<double> probabilities, std::vector<std::vector<double>> inputs)
{
    // Binary Cross Entropy Loss Function

    double loss = 0;
    for (int i = 0; i < inputs.size(); i++)
    {
        double y = inputs[i][0];         // True label
        double y_hat = probabilities[y]; // Predicted probability
        if (y_hat == 1)
            y_hat = 0.9999999;
        else if (y_hat == 0)
            y_hat = 0.0000001;
        loss += y * log(y_hat) + (1 - y) * log(1 - y_hat);
    }
    loss = -(1.0 / inputs.size()) * loss;
    std::cout << "loss : " << loss << std::endl;
    return loss;
}

void Layer::backPropagation(std::vector<Layer> &layers, std::vector<std::vector<double>> inputs, double learningRate)
{
}