#include "../includes/Layer.hpp"
#include <numeric>
#include <strings.h>
// for the input layer
Layer::Layer(std::vector<std::vector<float>> inputs)
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

// float Layer::reluFunction(float x)
// {
//     if (x > 0)
//         return 1;
//     else
//         return -1;
// }

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

// std::vector<float> Layer::singleSoftmax(std::vector<std::vector<float>> weights, std::vector<float> inputs)
// {
//     std::vector<float> outputs;
//     std::vector<float> exponentials;
//     float sum = 0.0;

//     for (auto &weight : weights)
//     {
//         float exponential = 0;
//         float max_input = 0;
//         for (int i = 1; i < inputs.size() + 1; i++)
//         {
//             if (inputs[i] > max_input)
//                 max_input = inputs[i];
//         }

//         for (int i = 1; i < inputs.size() + 1; i++)
//         {
//             exponential += exp(inputs[i] * weight[i] - max_input);
//         }
//         exponentials.push_back(exponential);
//         sum += exponential;
//     }
//     for (int i = 0; i < exponentials.size(); i++)
//     {
//         outputs.push_back(exponentials[i] / sum);
//     }

//     int index = 0;
//     float max = 0;
//     for (int i = 0; i < outputs.size(); i++)
//     {
//         if (outputs[i] > max)
//         {
//             max = outputs[i];
//             index = i;
//         }
//     }

//     return outputs;
// }

// float Layer::getValidationLoss(std::vector<std::vector<float>> validationSet, std::vector<std::vector<float>> finalWeights)
// {
//     float loss = 0;
//     for (int z = 0; z < validationSet.size(); z++)
//     {
//         std::vector<float> predi = singleSoftmax(finalWeights, validationSet[z]);
//         for (int k = 0; k < predi.size(); k++)
//             loss += (validationSet[z][0] * log(predi[k]));
//     }
//     loss = -(1.0 / validationSet.size()) * loss;
//     return loss;
// }

void Layer::inputLayerFeedForward(Layer &previousLayer)
{
    for (int i = 0; i < this->_neurons.size(); i++)
    {
        float sum = 0;
        std::vector<Neuron> &neurons = previousLayer.getNeurons();

        for (int j = 0; j < neurons.size(); j++)
            for (int k = 1; k < neurons[j].getInputs().size(); k++)
                sum += this->_neurons[i].getWeights(j)[k] * neurons[j].getInputs()[k];

        sum += previousLayer.getBiasNeuron();
        this->_neurons[i].setActivated(false);

        if (sum > 0)
            this->_neurons[i].setActivated(true);
        this->_neurons[i].setOutput(sum);
    }
    debugNeuronsActivated();
}

void Layer::hiddenLayerFeedForward(Layer &previousLayer)
{
    for (int i = 0; i < this->_neurons.size(); i++)
    {
        float sum = 0;
        std::vector<Neuron> &neurons = previousLayer.getNeurons();

        for (int j = 0; j < neurons.size(); j++)
        {
            if (neurons[j].getActivated() == false)
                continue;

            for (int k = 0; k < this->_neurons[i].getWeights(j).size(); k++)
                sum += this->_neurons[i].getWeights(j)[k]; //* neurons[j].getInputs()[k];
        }

        sum += previousLayer.getBiasNeuron();
        this->_neurons[i].setActivated(false);
        if (sum > 0)
            this->_neurons[i].setActivated(true);
        this->_neurons[i].setOutput(sum);
    }
    debugNeuronsActivated();
}

void Layer::outputLayerFeedForward(Layer &previousLayer)
{
    for (int i = 0; i < this->_neurons.size(); i++)
    {
        float sum = 0;
        std::vector<Neuron> &neurons = previousLayer.getNeurons();

        for (int j = 0; j < neurons.size(); j++)
        {
            if (neurons[j].getActivated() == false)
                continue;
            for (int k = 0; k < this->_neurons[i].getWeights(j).size(); k++)
                sum += this->_neurons[i].getWeights(j)[k]; //* neurons[j].getInputs()[k];
        }

        sum += previousLayer.getBiasNeuron();
        this->_neurons[i].setOutput(sum);
    }

    std::cout << std::endl;
}

void Layer::feedForward(Layer &previousLayer, int mode)
{
    if (mode == 0)
        this->inputLayerFeedForward(previousLayer);
    else if (mode == 1)
        this->hiddenLayerFeedForward(previousLayer);
    else if (mode == 2)
    {
        this->outputLayerFeedForward(previousLayer);
        this->applySoftmax();
        // inputs with 32 random values
        std::vector<float> inputs = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                     0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5,
                                     0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3,
                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1,
                                     0.2, 0.3, 0.4};
        std::vector<float> test = softmaxWithInput(inputs);
        for (auto &t : test)
        {
            std::cout << t << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<float> Layer::softmaxWithInput(std::vector<float> inputs)
{
    std::vector<std::vector<float>> weights;
    for (int i = 0; i < this->_neurons.size(); i++)
    {
        for (int j = 0; j < this->_neurons[i].getWeights().size(); j++)
            weights.push_back(this->_neurons[i].getWeights(j));
    }

    std::vector<float> outputs;
    std::vector<float> exponentials;
    float sum = 0.0;

    for (const auto &weight : weights)
    {
        float exponential = 0;
        float max_input = 0;
        for (int i = 1; i < inputs.size() + 1; i++)
        {
            if (inputs[i] > max_input)
                max_input = inputs[i];
        }

        for (int i = 1; i < inputs.size() + 1; i++)
        {
            exponential += exp(inputs[i] * weight[i] - max_input);
        }
        exponentials.push_back(exponential);
        sum += exponential;
    }

    for (int i = 0; i < exponentials.size(); i++)
    {
        outputs.push_back(exponentials[i] / sum);
    }

    return outputs;
}

void Layer::applySoftmax()
{
    std::vector<float> logits;
    for (auto &neuron : this->_neurons)
    {
        logits.push_back(neuron.getOutput());
    }

    std::vector<float> outputs;
    std::vector<float> exponentials;
    float sum = 0.0;

    for (const auto &logit : logits)
    {
        std::cout << logit << " ";
        float exponential = exp(logit);
        exponentials.push_back(exponential);
        sum += exponential;
    }
    std::cout << std::endl;
    for (int i = 0; i < exponentials.size(); i++)
    {
        outputs.push_back(exponentials[i] / sum);
        std::cout << outputs[i] << " ";
    }
    std::cout << std::endl;
}

void Layer::backPropagation(std::vector<Layer> &layers, std::vector<std::vector<float>> inputs, float learningRate)
{
    // std::vector<std::vector<float>> weightsOutputLayer;
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
    // float loss = 0;
    // std::vector<float> gradients;

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
    //                 std::vector<float> predi = singleSoftmax(weightsOutputLayer, inputs[z]);
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
    //                     float gradient = 0;
    //                     for (auto &input : inputs)
    //                     {
    //                         std::vector<float> predi = singleSoftmax(weightsOutputLayer, input);
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
    //                 float error = 0;
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
    //                     float gradient = 0.0;
    //                     for (auto &input : inputs)
    //                     {
    //                         std::vector<float> predi = singleSoftmax(weightsOutputLayer, input);
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

// void Layer::feedForward(Layer &previousLayer, int mode)
// {

//     for (int i = 0; i < this->_neurons.size(); i++) // in every neuron of the actual layer
//     {
//         float sum = 0;
//         std::vector<float> outputs(32, 0);                          // TODO: calculate the size of the outputs
//         for (int j = 0; j < previousLayer.getNeurons().size(); j++) // for every neuron of the previous layer
//         {

//             if (previousLayer.getNeurons()[j].getActivated() == false) // if the previous neuron is not activated, we skip it
//             {
//                 sum += previousLayer.getBiasNeuron();
//                 this->_neurons[i].setOutput(0);
//                 continue;
//             }

//             std::vector<float> previousNeuronsInputs = previousLayer.getNeurons()[j].getInputs();

//             outputs[0] = previousNeuronsInputs[0];                     // the first element is the answer
//             for (int k = 1; k < previousNeuronsInputs.size() + 1; k++) // start from 1 because the first element is the answer
//             {
//                 // float computed = (previousNeuronsInputs[k] * _neurons[i].getWeights()[k]) + previousLayer.getBiasNeuron();
//                 float computed = 0;
//                 computed = _neurons[i].getWeights()[k] * previousNeuronsInputs[k]; //+ previousLayer.getBiasNeuron();
//                 outputs[k] += computed;
//                 sum += computed;
//             }
//         }
//         sum += previousLayer.getBiasNeuron();
//         std::cout << sum << std::endl;
//         if (outputs[1] != 0)
//             this->getNeurons()[i].setInputs(outputs);

//         if (mode == 1) // hidden layers
//         {
//             this->_neurons[i].setOutput(0);
//             this->_neurons[i].setActivated(false);
//             if (sum > 0) // relu activation
//             {
//                 this->_neurons[i].setOutput(sum);
//                 this->_neurons[i].setActivated(true);
//             }
//         }
//         else if (mode == 2 && i == 0) // output layer first neuron
//         // use softMax in every neurons but inputs[0] should be the weights of the actual neuron
//         {
//             std::vector<std::vector<float>> inputs;
//             for (int i = 0; i < this->_neurons.size(); i++)
//             {
//                 inputs.push_back(this->_neurons[i].getWeights());
//             }

//             std::vector<float> result = softmaxFunction(inputs);
//             this->_neurons[i].setSoftmaxResults(result);
//         }
//     }
//     if (mode == 1)
//         debugNeuronsActivated();
// }

// void Layer::backPropagation(std::vector<Layer> &layers, std::vector<std::vector<float>> inputs, float learningRate)
// {
//     std::vector<std::vector<float>> weightsOutputLayer;
//     for (int i = 0; i < layers.back().getNeurons().size(); i++)
//     {
//         weightsOutputLayer.push_back(layers.back().getNeurons()[i].getWeights());
//     }
//     std::cout << std::endl;
//     float loss = 0;
//     std::vector<float> gradients;

//     for (int i = layers.size() - 1; i > -1; i--)
//     {
//         for (int j = 0; j < layers[i].getNeurons().size(); j++)
//         {
//             if (i == layers.size() - 1) // output layer
//             {
//                 if (j != 0)
//                     continue;
//                 for (auto &input : inputs)
//                 {
//                     float predi = singleSoftmax(weightsOutputLayer, input);
//                     // binary cross entropy
//                     loss += -(input[0] * log(predi) + (1 - input[0]) * log(1 - predi));
//                 }
//                 loss = (1.0 / inputs.size()) * loss;
//                 this->_neurons[j].setLoss(loss);
//                 // gradient with respect to the weights
//                 for (int k = 0; k < layers[i].getNeurons()[j].getWeights().size(); k++)
//                 {
//                     float gradient = 0;
//                     for (auto &input : inputs)
//                     {
//                         gradient += (singleSoftmax(weightsOutputLayer, input) - input[0]) * input[k];
//                     }
//                     gradient = (1.0 / inputs.size()) * gradient;
//                     gradients.push_back(gradient);
//                 }
//                 // Update the weights including bias
//                 for (int k = 0; k < layers[i].getNeurons()[j].getWeights().size(); k++)
//                 {
//                     float newWeight = layers[i].getNeurons()[j].getWeights()[k] - learningRate * gradients[k];
//                     layers[i].getNeurons()[j].setOneWeight(newWeight, k);
//                 }
//                 // Update the bias weight
//                 float biasGradient = 0;
//                 for (auto &input : inputs)
//                 {
//                     biasGradient += singleSoftmax(weightsOutputLayer, input) - input[0];
//                 }
//                 biasGradient = (1.0 / inputs.size()) * biasGradient;
//                 float newBiasWeight = layers[i].getNeurons()[j].getBias() - learningRate * biasGradient;
//                 layers[i].getNeurons()[j].setBias(newBiasWeight);
//             }
//             else if (i > 0) // hidden layers
//             {
//                 float gradient = 0;
//                 // gradient with respect to the weights
//                 for (auto &neuron : layers[i].getNeurons())
//                 {
//                     for (int k = 0; k < neuron.getWeights().size(); k++)
//                     {
//                         for (auto &input : inputs)
//                         {
//                             float sum = 0;
//                             for (int l = 0; l < layers[i + 1].getNeurons().size(); l++)
//                             {
//                                 sum += layers[i + 1].getNeurons()[l].getLoss() * layers[i + 1].getNeurons()[l].getWeights()[j];
//                             }
//                             gradient += neuron.getLoss() * sum * input[k];
//                         }
//                         gradient = (1.0 / inputs.size()) * gradient;
//                         gradients.push_back(gradient);
//                         // update the weights
//                         neuron.setOneWeight(neuron.getWeights()[k] - learningRate * gradients[k], k);
//                     }
//                 }
//                 // Update the bias weight
//                 float biasGradient = 0;
//                 for (auto &neuron : layers[i].getNeurons())
//                 {
//                     float sum = 0;
//                     for (int l = 0; l < layers[i + 1].getNeurons().size(); l++)
//                     {
//                         sum += layers[i + 1].getNeurons()[l].getLoss() * layers[i + 1].getNeurons()[l].getWeights()[j];
//                     }
//                     biasGradient += neuron.getLoss() * sum;
//                 }
//                 biasGradient = (1.0 / inputs.size()) * biasGradient;
//                 float newBiasWeight = layers[i].getNeurons()[j].getBias() - learningRate * biasGradient;
//                 layers[i].getNeurons()[j].setBias(newBiasWeight);
//             }
//         }
//     }
// }
