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
        // std::cout << "Loss: " << error << std::endl;
    }
}

double Layer::crossEntropyLoss(std::vector<double> probabilities, std::vector<double> inputs)
{
    // Categorical cross-entropy loss
    double loss = 0;
    double y_hat = probabilities[static_cast<int>(inputs[0])]; // Predicted probability
    return -log(y_hat + 1e-9);
}

double Layer::getValidationLoss(std::vector<std::vector<double>> validationSet, std::vector<double> probabilities)
{
    double loss = 0;
    for (int i = 0; i < validationSet.size(); i++)
    {
        loss += -log(probabilities[static_cast<int>(validationSet[i][0])] + 1e-9);
    }
    return loss / validationSet.size();
}

void Layer::backPropagation(std::vector<Layer> &layers, std::vector<double> target, double learningRate)
{
    // Calculate delta for output layer
    std::vector<double> deltaOutput;
    for (int k = 0; k < layers.back()._neurons.size(); ++k)
    {
        double aLk = layers.back()._neurons[k].getOutput();
        double tk = target[k];
        deltaOutput.push_back(aLk - tk);
    }

    // Update weights for the output layer
    for (int k = 0; k < layers.back()._neurons.size(); ++k)
    {
        for (int j = 0; j < layers[layers.size() - 2]._neurons.size(); ++j)
        {
            double gradient = learningRate * deltaOutput[k] * layers[layers.size() - 2]._neurons[j].getOutput();
            layers.back()._neurons[k].getWeights()[j] -= gradient;
        }
        // double biasUpdate = learningRate * deltaOutput[k];
        // layers.back()._neurons[k].adjustBias(-biasUpdate);
    }

    // Calculate delta for hidden layers
    for (int i = layers.size() - 2; i > 1; --i)
    {
        std::vector<double> deltaHidden;
        for (int j = 0; j < layers[i]._neurons.size(); ++j)
        {
            double sum = 0;
            for (int k = 0; k < layers[i + 1]._neurons.size(); ++k)
            {
                sum += deltaOutput[k] * layers[i + 1]._neurons[k].getWeights()[j];
            }
            deltaHidden.push_back(sum);
        }

        // Update weights for the hidden layers
        for (int j = 0; j < layers[i]._neurons.size(); ++j)
        {
            for (int k = 0; k < layers[i - 1]._neurons.size(); ++k)
            {
                double gradient = learningRate * deltaHidden[j] * layers[i - 1]._neurons[k].getOutput();
                layers[i]._neurons[j].getWeights()[k] -= gradient;
            }
            // double biasUpdate = learningRate * deltaHidden[j];
            // layers[i]._neurons[j].adjustBias(-biasUpdate);
        }
        deltaOutput = deltaHidden;
    }

    // Update weights for the first hidden layer using target
    std::vector<double> deltaHidden;
    for (int j = 0; j < layers[1]._neurons.size(); ++j)
    {
        double sum = 0;
        for (int k = 0; k < layers[2]._neurons.size(); ++k)
        {
            sum += deltaOutput[k] * layers[2]._neurons[k].getWeights()[j];
        }
        deltaHidden.push_back(sum);
    }

    for (int j = 0; j < layers[1]._neurons.size(); ++j)
    {
        for (int k = 0; k < layers[0]._neurons[0].getInputs().size(); ++k)
        {
            double gradient = learningRate * deltaHidden[j] * target[k];
            layers[1]._neurons[j].getWeights()[k] -= gradient;
        }
        // double biasUpdate = learningRate * deltaHidden[j];
        // layers[1]._neurons[j].adjustBias(-biasUpdate);
    }
}