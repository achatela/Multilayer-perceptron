#include <vector>
#include "Neuron.hpp"

class Layer
{

public:
    // for the input layer
    Layer(std::vector<std::vector<float>> inputs);
    // for the hidden layers
    Layer(int neuronsNumber, int sizePreviousLayer, int featureNumber);
    ~Layer();

    void feedForward(Layer &previousLayer);

    std::vector<Neuron> &getNeurons() { return this->_neurons; };
    float getBiasNeuron() { return this->_biasNeuron; };

    float reluFunction(float x);

private:
    float _biasNeuron;
    std::vector<Neuron> _neurons;
};