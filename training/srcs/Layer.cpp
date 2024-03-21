#include "../includes/Layer.hpp"
#include <numeric>
#include <strings.h>
// for the input layer
Layer::Layer(std::vector<std::vector<double>> inputs)
{
    for (size_t i = 0; i < inputs.size(); i++)
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
    (void)isOutputLayer;
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

void Layer::sigmoid(double sum, int i)
{
    // changed to sigmoid
    double result = 1.0 / (1.0 + exp(-sum));
    this->_neurons[i].setActivated(true);
    this->_neurons[i].setOutput(result);
    return;
    // if (sum > 0) // ReLU
    // {
    //     this->_neurons[i].setActivated(true);
    //     this->_neurons[i].setOutput(sum);
    //     return;
    // }
    // this->_neurons[i].setActivated(false);
    // this->_neurons[i].setOutput(0);
}

std::vector<double> Layer::softmaxFunction(std::vector<double> inputs)
{
    std::vector<double> outputs;
    double maxInput = *std::max_element(inputs.begin(), inputs.end());

    std::vector<double> exponentials;
    double sum = 0;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        // Subtract the max value to prevent overflow
        double exponential = exp(inputs[i] - maxInput);
        exponentials.push_back(exponential);
        sum += exponential;
    }

    for (size_t i = 0; i < exponentials.size(); i++)
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
    for (size_t i = 0; i < this->_neurons.size(); i++)
    {
        if (this->_neurons[i].getActivated())
            number++;
    }
    // std::cout << "Number of activated neurons: " << number << std::endl;
}

void Layer::firstHiddenLayerFeed(Layer &previousLayer, std::vector<double> input)
{
    (void)previousLayer;
    for (size_t i = 0; i < this->_neurons.size(); i++)
    {
        double sum = this->_neurons[i].getBias();

        for (size_t j = 1; j < input.size(); j++)
            sum += this->_neurons[i].getWeights()[j] * input[j];

        sigmoid(sum, i);
    }
    debugNeuronsActivated();
}

void Layer::hiddenLayerFeed(Layer &previousLayer)
{
    for (size_t i = 0; i < this->_neurons.size(); i++)
    {
        double sum = this->_neurons[i].getBias();

        std::vector<Neuron> &neurons = previousLayer.getNeurons();

        for (size_t j = 0; j < neurons.size(); j++)
            sum += this->_neurons[i].getWeights()[j] * neurons[j].getOutput();

        sigmoid(sum, i);
    }
    debugNeuronsActivated();
}

std::vector<double> Layer::outputLayerFeed(Layer &previousLayer)
{
    std::vector<double> logits;
    for (size_t i = 0; i < this->_neurons.size(); i++)
    {
        double sum = this->_neurons[i].getBias();

        std::vector<Neuron> &neurons = previousLayer.getNeurons();

        for (size_t j = 0; j < neurons.size(); j++)
            sum += this->_neurons[i].getWeights()[j] * neurons[j].getOutput();
        logits.push_back(sum);
    }
    // std::cout << std::endl;
    return logits;
}

double Layer::feedForward(Layer &previousLayer, int mode, std::vector<std::vector<double>> inputs, std::vector<double> input, std::vector<double> networkWeights)
{
    (void)inputs;
    if (mode == 0)
        this->firstHiddenLayerFeed(previousLayer, input);
    else if (mode == 1)
        this->hiddenLayerFeed(previousLayer);
    else if (mode == 2)
    {
        std::vector<double> probabilities = softmaxFunction(this->outputLayerFeed(previousLayer));
        for (size_t i = 0; i < probabilities.size(); i++)
            this->_neurons[i].setOutput(probabilities[i]);

        double error = crossEntropyLoss(probabilities, input[0], networkWeights);
        this->setLoss(error);
        return error;
        // std::cout << "Loss: " << error << std::endl;
    }
    return 0;
}

double Layer::crossEntropyLoss(std::vector<double> probabilities, int result, std::vector<double> networkWeights)
{
    (void)networkWeights;
    // Categorical cross-entropy loss
    // double loss = 0;
    // double lambda = 0.01;
    double y_hat = probabilities[result]; // Predicted probability
    return -log(y_hat);
    // loss = -log(y_hat + 1e-9);

    // // Regularization
    // double sum = 0;
    // for (size_t i = 0; i < networkWeights.size(); i++)
    //     sum += pow(networkWeights[i], 2);
    // loss += (lambda / (2 * networkWeights.size())) * sum;

    // return loss;
}

double Layer::getValidationLoss(std::vector<std::vector<double>> validationSet, std::vector<Layer> layers)
{
    double loss = 0;
    double accuracy = 0;

    for (size_t i = 0; i < validationSet.size(); i++)
    {
        std::vector<double> probabilities;
        for (size_t j = 1; j < layers.size(); j++)
        {
            if (j == layers.size() - 1)
                layers[j].feedForward(layers[j - 1], 2, validationSet, validationSet[i], {});
            else if (j == 1)
                layers[j].feedForward(layers[j - 1], 0, validationSet, validationSet[i]);
            else
                layers[j].feedForward(layers[j - 1], 1, validationSet, validationSet[i]);
        }
        for (auto &output : layers.back().getNeurons())
            probabilities.push_back(output.getOutput());
        if (std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end())) == static_cast<int>(validationSet[i][0]))
            accuracy++;

        loss += layers.back().getLoss();
    }

    accuracy /= validationSet.size();
    // for (size_t i = 0; i < validationSet.size(); i++)
    // {
    // loss += -log(probabilities[static_cast<int>(validationSet[i][0])]);
    // }
    std::cout << "Accuracy: " << accuracy << " ";
    return loss / validationSet.size();
}

void Layer::backPropagation(std::vector<Layer> &layers, std::vector<double> target, double learningRate)
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
        double biasUpdate = learningRate * deltaOutput[k];
        layers.back()._neurons[k].updateBias(biasUpdate);
    }

    // Calculate delta for hidden layers using relu derivative
    for (size_t i = layers.size() - 2; i > 1; i--)
    {
        std::vector<std::vector<double>> next_layer_gradients = std::move(layer_gradients);
        for (size_t j = 0; j < layers[i]._neurons.size(); j++)
        {
            layer_gradients.push_back(std::vector<double>());
            double derivates_sum = 0;
            for (size_t l = 0; l < next_layer_gradients.size(); l++)
                derivates_sum += next_layer_gradients[l][j] * layers[i + 1]._neurons[l].getWeights()[j];

            double biasUpdate = learningRate * derivates_sum;
            layers[i]._neurons[j].updateBias(biasUpdate);

            double loss_derivative = (layers[i]._neurons[j].getOutput() * (1 - layers[i]._neurons[j].getOutput())) * derivates_sum;

            for (size_t k = 0; k < layers[i - 1]._neurons.size(); k++)
            {
                double gradient = loss_derivative * layers[i - 1]._neurons[k].getOutput();
                layer_gradients.back().push_back(gradient);
                layers[i]._neurons[j].getWeights()[k] -= learningRate * gradient;
            }
        }
    }

    std::vector<std::vector<double>> next_layer_gradients = std::move(layer_gradients);
    for (size_t j = 0; j < layers[1]._neurons.size(); j++)
    {
        double derivates_sum = 0;

        for (size_t l = 0; l < next_layer_gradients.size(); l++)
            derivates_sum += next_layer_gradients[l][j] * layers[2]._neurons[l].getWeights()[j];

        double biasUpdate = learningRate * derivates_sum;
        layers[1]._neurons[j].updateBias(biasUpdate);

        double loss_derivative = (layers[1]._neurons[j].getOutput() * (1 - layers[1]._neurons[j].getOutput())) * derivates_sum;

        for (size_t k = 1; k < target.size(); k++)
        {
            double gradient = loss_derivative * target[k];
            layers[1]._neurons[j].getWeights()[k] -= learningRate * gradient;
        }
    }
}