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

    void forwardPropagation(Layer previousLayer);

private:
    float _biasNeuron;
    std::vector<Neuron> _neurons;
};