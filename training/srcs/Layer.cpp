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
    // Calculate delta for the output layer
    std::vector<double> probabilities;
    for (int i = 0; i < this->_neurons.size(); i++)
    {
        double output = this->_neurons[i].getOutput();
        probabilities.push_back(output);
    }
    double delta = (probabilities[target[0]] - target[0]) * probabilities[target[0]] * (1 - probabilities[target[0]]);
    this->_neurons[0].setDelta(delta);

    // Backpropagate the delta to previous layers
    for (int l = layers.size() - 2; l >= 0; l--)
    {
        Layer &layer = layers[l];
        Layer &nextLayer = layers[l + 1];

        for (int i = 0; i < layer._neurons.size(); i++)
        {
            double deltaSum = 0;
            for (int j = 0; j < nextLayer._neurons.size(); j++)
            {
                deltaSum += nextLayer._neurons[j].getWeights()[i] * nextLayer._neurons[j].getDelta();
            }
            double output = layer._neurons[i].getOutput();
            // / Derivative of the activation function (softmax)
            delta = deltaSum * output * (1 - output);
            layer._neurons[i].setDelta(delta);
        }
    }

    // Update weights and biases for all layers except the input layer
    for (int l = 1; l < layers.size(); l++)
    {
        Layer &layer = layers[l];
        Layer &prevLayer = layers[l - 1];

        for (int i = 0; i < layer._neurons.size(); i++)
        {
            Neuron &neuron = layer._neurons[i];
            for (int j = 0; j < neuron.getWeights().size(); j++)
            {
                double delta = neuron.getDelta();
                double previousOutput = prevLayer._neurons[j].getOutput();
                neuron.getWeights()[j] -= learningRate * delta * previousOutput; // Update weights
            }
            neuron.getBias() -= learningRate * neuron.getDelta(); // Update bias
        }
    }
}