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
        // std::cout << outputs.back() << " ";
    }
    // std::cout << std::endl;

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
    // std::cout << "Number of activated neurons: " << number << std::endl;
}

void Layer::firstHiddenLayerFeed(Layer &previousLayer, std::vector<double> input)
{
    for (int i = 0; i < this->_neurons.size(); i++)
    {
        double sum = 0;
        // std::vector<Neuron> &neurons = previousLayer.getNeurons();

        // for (int j = 0; j < neurons.size(); j++)
        // for (int k = 1; k < neurons[j].getInputs().size(); k++)
        // sum += this->_neurons[i].getWeights()[k] * neurons[j].getInputs()[k];
        for (int j = 0; j < input.size(); j++)
            sum += this->_neurons[i].getWeights()[j] * input[j];

        sum += previousLayer.getBiasNeuron();
        reluActivation(sum, i);
    }
    debugNeuronsActivated();
}

void Layer::hiddenLayerFeed(Layer &previousLayer)
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

std::vector<double> Layer::outputLayerFeed(Layer &previousLayer)
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
    // std::cout << std::endl;
    return logits;
}

void Layer::feedForward(Layer &previousLayer, int mode, std::vector<std::vector<double>> inputs, std::vector<double> input)
{
    if (mode == 0)
        this->firstHiddenLayerFeed(previousLayer, input);
    else if (mode == 1)
        this->hiddenLayerFeed(previousLayer);
    else if (mode == 2)
    {
        std::vector<double> probabilities = softmaxFunction(this->outputLayerFeed(previousLayer));
        for (int i = 0; i < probabilities.size(); i++)
            this->_neurons[i].setOutput(probabilities[i]);

        double error = crossEntropyLoss(probabilities, input);
        this->setLoss(error);
        std::cout << "Loss: " << error << std::endl;
    }
}

double Layer::crossEntropyLoss(std::vector<double> probabilities, std::vector<double> inputs)
{
    // Binary Cross Entropy Loss Function
    double loss = 0;
    double y = inputs[0];            // True label
    double y_hat = probabilities[y]; // Predicted probability

    loss += y * log(y_hat) + (1 - y) * log(1 - y_hat);
    loss = -loss;
    return loss;
}

void Layer::backPropagation(std::vector<Layer> &layers, std::vector<double> target, double learningRate)
{
    std::vector<double> gradients;
    std::vector<double> y_hats;
    double loss = this->getLoss();

    Layer &outputLayer = layers.back();
    for (int i = 0; i < outputLayer.getNeurons().size(); i++)
        y_hats.push_back(outputLayer.getNeurons()[i].getOutput());

    for (int i = layers.size() - 1; i > 1; i--)
    {
        Layer &currentLayer = layers[i];
        Layer &previousLayer = layers[i - 1];
        std::vector<Neuron> &currentNeurons = currentLayer.getNeurons();
        std::vector<Neuron> &previousNeurons = previousLayer.getNeurons();

        for (int j = 0; j < currentNeurons.size(); j++)
        {
            for (int k = 0; k < currentNeurons[j].getWeights().size(); k++)
            {
                // derivative of the loss function with respect to the output of the current neuron
                double gradient = 0;
                double delta = y_hats[target[0]] - target[0];
                gradient = delta * previousNeurons[k].getOutput();
                // std::cout << gradient << " " << previousNeurons[k].getOutput() << std::endl;
                // update weights
                double old = currentNeurons[j].getWeights()[k];
                currentNeurons[j].getWeights()[k] -= learningRate * gradient;
                // if (old != currentNeurons[j].getWeights()[k])
                // std::cout << "difference " << old - currentNeurons[j].getWeights()[k] << std::endl;
            }
        }
    }
}

// double gradient = 0;

// gradient = (target[0] - y_hats[target[0]]) * previousNeurons[k].getOutput();

// // for (int l = 0; l < y_hats.size(); l++)
// //     gradient += ((loss / y_hats[l]) * (y_hats[l] / target[l]));
// // std::cout << "gradient  before " << gradient << std::endl;
// // gradient *= (currentNeurons[j].getOutput() / currentNeurons[j].getWeights()[k]);

// // update weights
// double before = currentNeurons[j].getWeights()[k];
// currentNeurons[j].getWeights()[k] -= learningRate * gradient;
// // if (before != currentNeurons[j].getWeights()[k])
// // std::cout << "gradient  after " << gradient << std::endl;