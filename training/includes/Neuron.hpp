#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

class Neuron
{

public:
    // for the input layer
    Neuron(std::vector<double> inputs, int featureNumber);
    // for the hidden layers
    Neuron(int sizePreviousLayer, int featureNumber, int weightsNumber);
    // for the output layer
    Neuron();
    ~Neuron();

    void heInitialization(int sizePreviousLayer, int featureNumber);

    double &getBias() { return this->_bias; };
    void setBias(double bias) { this->_bias = bias; };
    void updateBias(double delta) { this->_bias -= delta; };

    std::vector<double> &getWeights() { return this->_weights; };
    // std::vector<double> &getWeights(int index) { return this->_weights[index]; };
    // void setWeights(std::vector<double> weights, int index) { this->_weights[index] = weights; };

    // void setOneWeight(double weight, int index, int offset) { this->_weights[index][offset] = weight; };

    std::vector<double> &getInputs() { return this->_inputs; };
    void setInputs(std::vector<double> inputs) { this->_inputs = inputs; };

    void setOutput(double output) { this->_output = output; };
    double getOutput() { return this->_output; };

    void setActivated(bool activated) { this->_activated = activated; };
    bool getActivated() { return this->_activated; };

    void setSlope(double slope) { this->slope = slope; };
    double getSlope() { return slope; };

    void setIntercept(double intercept) { this->intercept = intercept; };
    double getIntercept() { return intercept; };

    void setClassPredicted(int classPredicted) { this->classPredicted = classPredicted; };
    int getClassPredicted() { return this->classPredicted; };

    void setSoftmaxResults(std::vector<double> softmaxResults) { this->softmaxResults = softmaxResults; };
    std::vector<double> getSoftmaxResults() { return this->softmaxResults; };

    void setError(double error) { this->error = error; };
    double getError() { return this->error; };

    void setDelta(double delta) { this->delta = delta; };
    double getDelta() { return this->delta; };

private:
    std::vector<double> softmaxResults;
    double error;

    double delta;

    double _bias;
    std::vector<double> _weights;
    std::vector<double> _inputs;
    double _output;
    bool _activated = false;

    double slope = 0.01;
    double intercept = 0.01;

    double learningRate = 0.01;

    int classPredicted;
};