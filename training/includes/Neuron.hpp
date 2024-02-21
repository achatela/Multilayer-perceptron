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
    void setBias(float bias) { this->_bias = bias; };

    std::vector<float> &getWeights() { return this->_weights; };
    void setWeights(std::vector<float> weights) { this->_weights = weights; };

    void setOneWeight(float weight, int index) { this->_weights[index] = weight; };

    std::vector<float> &getInputs() { return this->_inputs; };
    void setInputs(std::vector<float> inputs) { this->_inputs = inputs; };

    void setOutput(float output) { this->_output = output; };
    float getOutput() { return this->_output; };

    void setActivated(bool activated) { this->_activated = activated; };
    bool getActivated() { return this->_activated; };

    void setSlope(float slope) { this->slope = slope; };
    float getSlope() { return slope; };

    void setIntercept(float intercept) { this->intercept = intercept; };
    float getIntercept() { return intercept; };

    void setClassPredicted(int classPredicted) { this->classPredicted = classPredicted; };
    int getClassPredicted() { return this->classPredicted; };

    void setSoftmaxResults(std::vector<float> softmaxResults) { this->softmaxResults = softmaxResults; };
    std::vector<float> getSoftmaxResults() { return this->softmaxResults; };

    void setLoss(float loss) { this->loss = loss; };
    float getLoss() { return this->loss; };

private:
    std::vector<float> softmaxResults;
    float loss;

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