#include <vector>
#include <algorithm>
#include "Neuron.hpp"

class Layer
{

public:
    Layer();
    // for the input layer
    Layer(std::vector<std::vector<double>> &inputs);
    // for the hidden layers
    Layer(int neuronsNumber, int weightsNumber);
    // for the output layer
    ~Layer();

    void backPropagation(std::vector<Layer> &layers, std::vector<double> &input, double learningRate);

    std::vector<Neuron> &getNeurons() { return this->_neurons; };
    std::vector<double> softmaxFunction(std::vector<double> &inputs);

    double getValidationLoss(std::vector<std::vector<double>> &validationSet, std::vector<Layer> &layers);

    void setLoss(double loss) { this->_loss = loss; };
    double getLoss() { return this->_loss; };

    void setBiasNeuron(double biasNeuron) { this->_biasNeuron = biasNeuron; };
    double getBiasNeuron() { return _biasNeuron; };

    void firstHiddenLayerFeed(std::vector<double> &input);
    void hiddenLayerFeed(Layer &previousLayer);
    void outputLayerFeed(Layer &previousLayer, std::vector<double> &input);

    void sigmoid(double sum, int i);

    double crossEntropyLoss(std::vector<double> &probabilities, int result);

private:
    double _loss;
    std::vector<Neuron> _neurons;
    double _biasNeuron;
};