#include <vector>
#include <algorithm>
#include "Neuron.hpp"

class Layer
{

public:
    // for the input layer
    Layer(std::vector<std::vector<float>> inputs);
    // for the hidden layers
    Layer(int neuronsNumber, int sizePreviousLayer, int featureNumber);
    // for the output layer
    Layer(int neuronsNumber, int sizePreviousLayer, int featureNumber, bool isOutputLayer);
    ~Layer();

    void feedForward(Layer &previousLayer, int mode);
    void backPropagation(std::vector<Layer> &layers, std::vector<std::vector<float>> inputs, float learningRate);

    std::vector<Neuron> &getNeurons() { return this->_neurons; };
    float getBiasNeuron() { return this->_biasNeuron; };

    std::vector<float> softmaxFunction(std::vector<std::vector<float>> inputs);
    // float reluFunction(float x);

    std::vector<float> calculatePrediction(std::vector<float> inputs, std::vector<float> weights, int size);

    void debugNeuronsActivated();

    float singleSoftmax(std::vector<std::vector<float>>, std::vector<float> inputs);
    float getValidationLoss(std::vector<std::vector<float>> validationSet, std::vector<std::vector<float>> finalWeights);

    void setLoss(float loss) { this->loss = loss; };
    float getLoss() { return this->loss; };

private:
    float loss;
    float _biasNeuron;
    std::vector<Neuron> _neurons;
    float _numClasses = 2; // TODO caculate this value in the main
};