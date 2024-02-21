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
    // for (auto &output : outputs)
    // {
    //     std::cout << output << " ";
    // }
    // std::cout << std::endl;

    return outputs;
}

std::vector<float> Layer::calculatePrediction(std::vector<float> inputs, std::vector<float> weights, int size = 2)
{
    std::vector<float> prediction;
    float sum = 0;

    for (int i = 0; i < inputs.size(); i++)
    {
        sum += inputs[i] * weights[i];
    }
    sum += this->_biasNeuron;
    prediction.push_back(reluFunction(sum));

    return prediction;
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
        else if (mode == 2 && i == 0) // output layer last neuron
        {
            std::vector<std::vector<float>> inputs;
            for (int i = 0; i < this->_neurons.size(); i++)
            {
                inputs.push_back(this->_neurons[i].getWeights());
            }

            std::vector<float> result = softmaxFunction(inputs);
            this->_neurons[i].setSoftmaxResults(result);
        }
    }
    // debugNeuronsActivated();
}

void Layer::debugNeuronsActivated()
{
    int number = 0;
    for (int i = 0; i < this->_neurons.size(); i++)
    {
        if (this->_neurons[i].getActivated())
            number++;
        this->_neurons[i].setActivated(false);
    }
    std::cout << "Number of activated neurons: " << number << std::endl;
}

float Layer::singleSoftmax(std::vector<std::vector<float>> weights, std::vector<float> inputs)
{
    std::vector<float> outputs;
    std::vector<float> exponentials;
    float sum = 0.0;

    for (auto &weight : weights)
    {
        float exponential = 0;
        float max_input = 0;
        for (float val : inputs)
        {
            if (val > max_input)
                max_input = val;
        }

        for (int i = 0; i < inputs.size(); i++)
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

    int index = 0;
    float max = 0;
    for (int i = 0; i < outputs.size(); i++)
    {
        if (outputs[i] > max)
        {
            max = outputs[i];
            index = i;
        }
    }
    return outputs[index];
}

float Layer::getValidationLoss(std::vector<std::vector<float>> validationSet, std::vector<std::vector<float>> finalWeights)
{
    float loss = 0;
    for (auto &input : validationSet)
    {
        float predi = singleSoftmax(finalWeights, input);
        loss += -(input[0] * log(predi) + (1 - input[0]) * log(1 - predi));
    }
    loss = (1.0 / validationSet.size()) * loss;
    return loss;
}

void Layer::backPropagation(std::vector<Layer> &layers, std::vector<std::vector<float>> inputs, float learningRate)
{
    std::vector<std::vector<float>> weightsOutputLayer;
    for (int i = 0; i < layers.back().getNeurons().size(); i++)
    {
        weightsOutputLayer.push_back(layers.back().getNeurons()[i].getWeights());
    }
    std::cout << std::endl;
    float loss = 0;
    std::vector<float> gradients;

    for (int i = layers.size() - 1; i > -1; i--)
    {
        for (int j = 0; j < layers[i].getNeurons().size(); j++)
        {
            if (i == layers.size() - 1) // output layer
            {
                if (j != 0)
                    continue;
                for (auto &input : inputs)
                {
                    float predi = singleSoftmax(weightsOutputLayer, input);
                    // binary cross entropy
                    loss += -(input[0] * log(predi) + (1 - input[0]) * log(1 - predi));
                }
                loss = (1.0 / inputs.size()) * loss;
                this->_neurons[j].setLoss(loss);
                // gradient with respect to the weights
                for (int k = 0; k < layers[i].getNeurons()[j].getWeights().size(); k++)
                {
                    float gradient = 0;
                    for (auto &input : inputs)
                    {
                        gradient += (singleSoftmax(weightsOutputLayer, input) - input[0]) * input[k];
                    }
                    gradient = (1.0 / inputs.size()) * gradient;
                    gradients.push_back(gradient);
                }
                // update the weights
                for (int k = 0; k < layers[i].getNeurons()[j].getWeights().size(); k++)
                {
                    layers[i].getNeurons()[j].setOneWeight(layers[i].getNeurons()[j].getWeights()[k] - learningRate * gradients[k], k);
                }
            }
            else if (i > 0) // hidden layers
            {
                float gradient = 0;
                // gradient with respect to the weights
                for (auto &neuron : layers[i].getNeurons())
                {
                    for (int k = 0; k < neuron.getWeights().size(); k++)
                    {
                        for (auto &input : inputs)
                        {
                            float sum = 0;
                            for (int l = 0; l < layers[i + 1].getNeurons().size(); l++)
                            {
                                sum += layers[i + 1].getNeurons()[l].getLoss() * layers[i + 1].getNeurons()[l].getWeights()[j];
                            }
                            gradient += neuron.getLoss() * sum * input[k];
                        }
                        gradient = (1.0 / inputs.size()) * gradient;
                        gradients.push_back(gradient);
                        // update the weights
                        neuron.setOneWeight(neuron.getWeights()[k] - learningRate * gradients[k], k);
                    }
                }
            }
        }
    }
}

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
