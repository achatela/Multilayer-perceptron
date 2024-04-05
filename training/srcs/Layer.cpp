#include "../includes/Layer.hpp"
#include <numeric>
#include <strings.h>

Layer::Layer() {}
Layer::~Layer() {}

Layer::Layer(std::vector<std::vector<double>> &neuronsWeights, double biasNeuron) // load model
{
    for (size_t i = 0; i < neuronsWeights.size(); i++)
        _neurons.push_back(Neuron(neuronsWeights[i], true));
    setBiasNeuron(biasNeuron);
}

Layer::Layer(std::vector<std::vector<double>> &inputs) // input layer
{
    for (size_t i = 0; i < inputs.size(); i++)
        _neurons.push_back(Neuron(inputs[i]));
}

Layer::Layer(int neuronsNumber, int weightsNumber) // hidden layers and output layer
{
    setBiasNeuron((double)rand() / (double)RAND_MAX);
    for (int i = 0; i < neuronsNumber; i++)
        _neurons.push_back(Neuron(weightsNumber));
}

void Layer::sigmoid(double sum, int i)
{
    this->_neurons[i].setOutput(1.0 / (1.0 + exp(-sum)));
}

std::vector<double> Layer::softmaxFunction(std::vector<double> &inputs)
{
    std::vector<double> outputs;
    std::vector<double> exponentials;
    double maxInput = *std::max_element(inputs.begin(), inputs.end());
    double sum = 0;

    for (size_t i = 0; i < inputs.size(); i++)
    {
        exponentials.push_back(exp(inputs[i] - maxInput));
        sum += exponentials.back();
    }

    for (size_t i = 0; i < exponentials.size(); i++)
        outputs.push_back(exponentials[i] / sum);

    return outputs;
}

void Layer::firstHiddenLayerFeed(std::vector<double> &input)
{
    for (size_t i = 0; i < this->_neurons.size(); i++)
    {
        double sum = this->getBiasNeuron();

        for (size_t j = 1; j < input.size(); j++)
            sum += this->_neurons[i].getWeights()[j] * input[j];

        sigmoid(sum, i);
    }
}

void Layer::hiddenLayerFeed(Layer &previousLayer)
{
    for (size_t i = 0; i < this->_neurons.size(); i++)
    {
        std::vector<Neuron> &neurons = previousLayer.getNeurons();
        double sum = this->getBiasNeuron();

        for (size_t j = 0; j < neurons.size(); j++)
            sum += this->_neurons[i].getWeights()[j] * neurons[j].getOutput();

        sigmoid(sum, i);
    }
}

void Layer::outputLayerFeed(Layer &previousLayer, std::vector<double> &input)
{
    std::vector<double> logits;
    for (size_t i = 0; i < this->_neurons.size(); i++)
    {
        std::vector<Neuron> &neurons = previousLayer.getNeurons();
        double sum = this->getBiasNeuron();

        for (size_t j = 0; j < neurons.size(); j++)
            sum += this->_neurons[i].getWeights()[j] * neurons[j].getOutput();
        logits.push_back(sum);
    }
    std::vector<double> probabilities = softmaxFunction(logits);

    for (size_t i = 0; i < probabilities.size(); i++)
        this->_neurons[i].setOutput(probabilities[i]);

    this->setLoss(crossEntropyLoss(probabilities, input[0]));
}

double Layer::crossEntropyLoss(std::vector<double> &probabilities, int result)
{
    // Binary cross entropy
    double y_hat = probabilities[1];
    return (result * log(y_hat) + (1 - result) * log(1 - y_hat));
    // return y_hat * log(y_hat) + (1 - y_hat) * log(1 - y_hat);
}

void Layer::getValidationLoss(std::vector<std::vector<double>> &validationSet, std::vector<Layer> &layers, std::vector<double> &loss, std::vector<double> &accuracy)
{
    double lossSet = 0;
    double accuracySet = 0;

    std::vector<Layer> layersCopy = layers;

    for (size_t i = 0; i < validationSet.size(); i++)
    {
        std::vector<double> probabilities;

        layersCopy[1].firstHiddenLayerFeed(validationSet[i]);
        for (size_t j = 2; j < layersCopy.size() - 1; j++)
            layersCopy[j].hiddenLayerFeed(layersCopy[j - 1]);
        layersCopy[layersCopy.size() - 1].outputLayerFeed(layersCopy[layersCopy.size() - 2], validationSet[i]);

        for (auto &output : layersCopy.back().getNeurons())
            probabilities.push_back(output.getOutput());
        if (std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end())) == static_cast<int>(validationSet[i][0]))
            accuracySet++;

        lossSet += layersCopy.back().getLoss();
    }

    accuracySet /= validationSet.size();
    lossSet = -(1.0 / validationSet.size()) * lossSet;
    std::cout << lossSet;

    accuracy.push_back(accuracySet);
    loss.push_back(lossSet);

    // std::cout << " accuracy:" << accuracy;
    // return -loss / validationSet.size();
}

void Layer::backPropagation(std::vector<Layer> &layers, std::vector<double> &target, double learningRate)
{

    std::vector<double> deltaOutput;
    for (size_t k = 0; k < layers.back()._neurons.size(); k++)
    {
        double activation = layers.back()._neurons[k].getOutput();
        double targetValue = (k == target[0]) ? 1.0 : 0.0;
        deltaOutput.push_back(activation - targetValue);
    }

    std::vector<std::vector<double>> layer_gradients;

    // Update weights for the output layer
    for (size_t k = 0; k < layers[layers.size() - 1]._neurons.size(); k++)
    {
        layer_gradients.push_back(std::vector<double>());
        for (size_t j = 0; j < layers[layers.size() - 2]._neurons.size(); j++)
        {
            double gradient = deltaOutput[k] * layers[layers.size() - 2]._neurons[j].getOutput();
            layer_gradients.back().push_back(gradient);
            layers.back()._neurons[k].getWeights()[j] -= learningRate * gradient;
        }

        double biasUpdate = layers.back().getBiasNeuron() - (learningRate * deltaOutput[k]);
        layers.back().setBiasNeuron(biasUpdate);
    }

    // Calculate gradient for hidden layers using relu derivative
    for (size_t i = layers.size() - 2; i > 1; i--)
    {
        std::vector<std::vector<double>> next_layer_gradients = std::move(layer_gradients);
        for (size_t j = 0; j < layers[i]._neurons.size(); j++)
        {
            layer_gradients.push_back(std::vector<double>());
            double weights_derivative_sums = 0;
            double derivates_sum = 0;

            for (size_t l = 0; l < next_layer_gradients.size(); l++)
                derivates_sum += next_layer_gradients[l][j] * layers[i + 1]._neurons[l].getWeights()[j];

            double loss_derivative = (layers[i]._neurons[j].getOutput() * (1 - layers[i]._neurons[j].getOutput())) * derivates_sum;

            for (size_t k = 0; k < layers[i - 1]._neurons.size(); k++)
            {
                double gradient = loss_derivative * layers[i - 1]._neurons[k].getOutput();
                layer_gradients.back().push_back(gradient);
                layers[i]._neurons[j].getWeights()[k] -= learningRate * gradient;
                weights_derivative_sums += layers[i - 1]._neurons[k].getOutput() * (1 - layers[i - 1]._neurons[k].getOutput());
            }

            double biasUpdate = layers[i].getBiasNeuron() - (learningRate * derivates_sum * weights_derivative_sums);
            layers[i].setBiasNeuron(biasUpdate);
        }
    }

    std::vector<std::vector<double>> next_layer_gradients = std::move(layer_gradients);
    for (size_t j = 0; j < layers[1]._neurons.size(); j++)
    {
        double derivates_sum = 0;

        for (size_t l = 0; l < next_layer_gradients.size(); l++)
            derivates_sum += next_layer_gradients[l][j] * layers[2]._neurons[l].getWeights()[j];

        double loss_derivative = (layers[1]._neurons[j].getOutput() * (1 - layers[1]._neurons[j].getOutput())) * derivates_sum;

        double weights_derivative_sums = 0;
        for (size_t k = 1; k < target.size(); k++)
        {
            double gradient = loss_derivative * target[k];
            layers[1]._neurons[j].getWeights()[k] -= learningRate * gradient;
            weights_derivative_sums += layers[1]._neurons[k].getOutput() * (1 - layers[1]._neurons[k].getOutput());
        }

        double biasUpdate = layers[1].getBiasNeuron() - (learningRate * derivates_sum * weights_derivative_sums);
        layers[1].setBiasNeuron(biasUpdate);
    }
}