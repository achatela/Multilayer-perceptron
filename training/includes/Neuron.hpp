#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>

class Neuron
{

public:
    // for the input layer
    Neuron(std::vector<float> inputs, int featureNumber);
    // for the hidden layers
    Neuron(int sizePreviousLayer, int featureNumber);
    ~Neuron();

    void heInitialization(int sizePreviousLayer, int featureNumber);

private:
    float _bias;
    std::vector<float> _weights;
    std::vector<float> _inputs;
};