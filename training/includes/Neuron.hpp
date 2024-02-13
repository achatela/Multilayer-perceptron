#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

class Neuron
{

public:
    // for the input layer
    Neuron(std::vector<float> inputs, int featureNumber);
    // for the hidden layers
    Neuron(int sizePreviousLayer, int featureNumber);
    ~Neuron();

    void heInitialization(int sizePreviousLayer, int featureNumber);

    float &getBias() { return this->_bias; };

    std::vector<float> &getWeights() { return this->_weights; };
    void setWeights(std::vector<float> weights) { this->_weights = weights; };

    std::vector<float> &getInputs() { return this->_inputs; };
    void setInputs(std::vector<float> inputs) { this->_inputs = inputs; };

    void setOutput(float output) { this->_output = output; };
    float getOutput() { return this->_output; };

    void setActivated(bool activated) { this->_activated = activated; };
    bool getActivated() { return this->_activated; };

    float getSlope() { return slope; };
    float getIntercept() { return intercept; };

    void setSlope(float slope) { this->slope = slope; };
    void setIntercept(float intercept) { this->intercept = intercept; };

    void setClassPredicted(int classPredicted) { this->classPredicted = classPredicted; };

private:
    float _bias;
    std::vector<float> _weights;
    std::vector<float> _inputs;
    float _output;
    bool _activated = false;

    float slope = 0.01;
    float intercept = 0.01;

    float learningRate = 0.01;

    int classPredicted;
};