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
    }
}

double Layer::crossEntropyLoss(std::vector<double> probabilities, std::vector<std::vector<double>> inputs)
{
    // Binary Cross Entropy Loss Function

    double loss = 0;
    for (int i = 0; i < inputs.size(); i++)
    {
        double y = inputs[i][0];         // True label
        double y_hat = probabilities[0]; // Predicted probability
        if (y_hat == 1)
            y_hat = 0.9999999;
        else if (y_hat == 0)
            y_hat = 0.0000001;
        loss += y * log(y_hat) + (1 - y) * log(1 - y_hat);
    }
    loss = -(1.0 / inputs.size()) * loss;
    std::cout << "loss : " << loss << std::endl;
    return loss;

    // double loss = 0;
    // for (int i = 0; i < probabilities.size(); i++)
    // {
    //     for (int j = 0; j < inputs.size(); j++)
    //     {
    //         loss += (inputs[j][0] * log(probabilities[inputs[j][0]] + 1e-15));
    //     }
    // }
    // loss = -(1.0 / inputs.size()) * loss;
    // std::cout << "loss : " << loss << std::endl;
    // return loss;
}

void Layer::backPropagation(std::vector<Layer> &layers, std::vector<std::vector<double>> inputs, double learningRate)
{
    // std::vector<std::vector<double>> weightsOutputLayer;
    // for (int i = 0; i < layers.back().getNeurons().size(); i++)
    // {
    //     weightsOutputLayer.push_back(layers.back().getNeurons()[i].getWeights());
    // }
    // // for (auto &weightsoutputs : weightsOutputLayer)
    // // {
    // //     for (auto &weight : weightsoutputs)
    // //     {
    // //         std::cout << weight << " ";
    // //     }
    // // }
    // std::cout << std::endl;
    // double loss = 0;
    // std::vector<double> gradients;

    // layers[layers.size() - 1].setLoss(0);
    // for (int i = layers.size() - 1; i > 0; i--)
    // {
    //     for (int j = 0; j < layers[i].getNeurons().size(); j++)
    //     {
    //         if (i == layers.size() - 1) // output layer
    //         {
    //             loss = 0;
    //             for (int z = 0; z < inputs.size(); z++)
    //             {
    //                 std::vector<double> predi = singleSoftmax(weightsOutputLayer, inputs[z]);
    //                 for (int k = 0; k < predi.size(); k++)
    //                     loss += (inputs[z][0] * log(predi[k]));
    //                 // loss += -log(predi[k]);
    //             }
    //             layers[i].setLoss(loss + layers[i].getLoss());
    //             this->_neurons[j].setError(-(1.0 / inputs.size()) * loss);
    //             if (j == layers[i].getNeurons().size() - 1)
    //             {
    //                 layers[i].setLoss(-(1.0 / inputs.size()) * layers[i].getLoss());
    //                 // gradient with respect to the weights
    //                 for (int k = 1; k < layers[i].getNeurons()[j].getWeights().size() + 1; k++)
    //                 {
    //                     double gradient = 0;
    //                     for (auto &input : inputs)
    //                     {
    //                         std::vector<double> predi = singleSoftmax(weightsOutputLayer, input);
    //                         int highestProbability = 0;
    //                         for (int l = 0; l < predi.size(); l++)
    //                         {
    //                             if (predi[l] > predi[highestProbability])
    //                                 highestProbability = l;
    //                         }
    //                         // highestProbability += 1;
    //                         // gradient += input[0] * (highestProbability - input[0]) * (highestProbability * (1 - highestProbability));
    //                         gradient += input[0] * (predi[highestProbability] - input[0]) * (predi[highestProbability] * (1 - predi[highestProbability]));
    //                     }
    //                     gradient /= inputs.size();
    //                     gradients.push_back(gradient);
    //                 }

    //                 // update the weights
    //                 for (int k = 1; k < layers[i].getNeurons()[j].getWeights().size() + 1; k++)
    //                 {
    //                     layers[i].getNeurons()[j].setOneWeight(layers[i].getNeurons()[j].getWeights()[k] - (learningRate * gradients[k - 1]), k);
    //                 }
    //                 layers[i].setGradients(gradients);
    //                 gradients.clear();
    //             }
    //         }
    //         else if (i > 0) // hidden layers
    //         {
    //             gradients.clear();
    //             for (int j = 0; j < layers[i].getNeurons().size(); j++)
    //             {
    //                 double error = 0;
    //                 for (int k = 0; k < layers[i + 1].getNeurons().size(); k++)
    //                 {
    //                     for (int m = 1; m < layers[i + 1].getNeurons()[k].getWeights().size() + 1; m++)
    //                     {
    //                         error += layers[i + 1].getNeurons()[k].getError() * layers[i + 1].getNeurons()[k].getWeights()[m];
    //                     }
    //                 }
    //                 error *= (layers[i].getNeurons()[j].getOutput() > 0) ? 1 : 0; // derivative of ReLU

    //                 layers[i].getNeurons()[j].setError(error);

    //                 for (int k = 1; k < layers[i].getNeurons()[j].getWeights().size() + 1; k++)
    //                 {
    //                     double gradient = 0.0;
    //                     for (auto &input : inputs)
    //                     {
    //                         std::vector<double> predi = singleSoftmax(weightsOutputLayer, input);
    //                         int highestProbability = 0;
    //                         for (int l = 0; l < predi.size(); l++)
    //                         {
    //                             if (predi[l] > predi[highestProbability])
    //                                 highestProbability = l;
    //                         }
    //                         if (predi[0] == 0 || predi[1] == 0)
    //                             std::cout << predi[0] << " " << predi[1] << std::endl;
    //                         // highestProbability += 1;
    //                         // gradient += input[0] * (highestProbability - input[0]) * (highestProbability * (1 - highestProbability));
    //                         gradient += input[0] * (predi[highestProbability] - input[0]) * (predi[highestProbability] * (1 - predi[highestProbability]));
    //                     }
    //                     gradient /= inputs.size();
    //                     gradients.push_back(gradient);
    //                 }
    //                 // update the weights
    //                 for (int k = 1; k < layers[i].getNeurons()[j].getWeights().size() + 1; k++)
    //                 {
    //                     layers[i].getNeurons()[j].setOneWeight(layers[i].getNeurons()[j].getWeights()[k] - (learningRate * gradients[k - 1]), k);
    //                 }
    //                 layers[i].setGradients(gradients);
    //                 gradients.clear();
    //             }
    //         }
    //     }
    // }
    // // reset every neurons activated to false
    // for (int i = 1; i < layers.size() - 1; i++)
    // {
    //     for (auto &neuron : layers[i].getNeurons())
    //     {
    //         neuron.setActivated(false);
    //     }
    // }
}