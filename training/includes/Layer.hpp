#include <vector>
#include <algorithm>
#include "Neuron.hpp"

class Layer
{

public:
    // for the input layer
    Layer(std::vector<std::vector<double>> inputs);
    // for the hidden layers
    Layer(int neuronsNumber, int sizePreviousLayer, int featureNumber, int weightsNumber);
    // for the output layer
    Layer(int neuronsNumber, int sizePreviousLayer, int featureNumber, int weightsNumber, bool isOutputLayer);
    ~Layer();

    double feedForward(Layer &previousLayer, int mode, std::vector<std::vector<double>> inputs = {}, std::vector<double> input = {}, std::vector<double> networkWeights = {});
    void backPropagation(std::vector<Layer> &layers, std::vector<double> input, double learningRate);

    std::vector<Neuron> &getNeurons() { return this->_neurons; };
    double getBiasNeuron() { return this->_biasNeuron; };

    std::vector<double> softmaxFunction(std::vector<double> inputs);
    // double reluFunction(double x);

    std::vector<double> calculatePrediction(std::vector<double> inputs, std::vector<double> weights, int size);

    void debugNeuronsActivated();

    std::vector<double> singleSoftmax(std::vector<std::vector<double>>, std::vector<double> inputs);
    double getValidationLoss(std::vector<std::vector<double>> validationSet, std::vector<Layer> layers);

    void setLoss(double loss) { this->loss = loss; };
    double getLoss() { return this->loss; };

    void setGradients(std::vector<double> gradients) { this->gradients = gradients; };
    std::vector<double> getGradients() { return this->gradients; };

    void firstHiddenLayerFeed(Layer &previousLayer, std::vector<double> input);
    void hiddenLayerFeed(Layer &previousLayer);
    std::vector<double> outputLayerFeed(Layer &previousLayer);

    void applySoftmax();
    void sigmoid(double sum, int i);

    std::vector<double> softmaxWithInput(std::vector<double> inputs);
    double crossEntropyLoss(std::vector<double> probabilities, int result, std::vector<double> networkWeights);

private:
    double loss;
    double _biasNeuron;
    std::vector<Neuron> _neurons;
    double _numClasses = 2; // TODO caculate this value in the main
    std::vector<double> gradients;
};